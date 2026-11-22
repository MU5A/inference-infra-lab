"""The run loop, its summaries, and the environment record."""

from __future__ import annotations

import json

import pytest

from inferlab import env, runner
from inferlab.client import StreamResult
from inferlab.fakeserver import Behavior, FakeServer


def scripted(ttft_s: float, gaps_s: list[float], *, tokens: int | None = None, error=None):
    """A StreamResult whose timings are exact, with no network involved."""
    start = 1_000_000_000
    arrivals = [start + int(ttft_s * 1e9)]
    for gap in gaps_s:
        arrivals.append(arrivals[-1] + int(gap * 1e9))
    usage = {"completion_tokens": tokens if tokens is not None else len(arrivals) + 1}
    usage["prompt_tokens"] = 30
    return StreamResult(
        start_ns=start,
        end_ns=arrivals[-1] if arrivals else start,
        text_arrivals_ns=arrivals if error is None else [],
        usage=usage,
        error=error,
    )


class TestPayload:
    def test_cold_runs_turn_the_prompt_cache_off(self):
        p = runner.build_payload("hi", max_tokens=5, temperature=0, seed=3, cold=True)
        assert p["cache_prompt"] is False
        assert (p["max_tokens"], p["temperature"], p["seed"]) == (5, 0, 3)
        assert p["messages"] == [{"role": "user", "content": "hi"}]

    def test_warm_runs_leave_the_prompt_cache_alone(self):
        p = runner.build_payload("hi", max_tokens=5, temperature=0, seed=3, cold=False)
        assert "cache_prompt" not in p


class TestRunBench:
    def test_warmup_requests_are_sent_but_not_measured(self):
        calls = []
        ttfts = iter([9.0, 9.0, 0.1, 0.2, 0.3])  # two warm-ups that are deliberately terrible

        def send(payload):
            calls.append(payload)
            return scripted(next(ttfts), [0.01, 0.01])

        report = runner.run_bench(send, {"p": "x"}, runs=3, warmup=2)
        assert len(calls) == 5
        block = report["results"]["p"]
        assert [r["ttft_s"] for r in block["requests"]] == pytest.approx([0.1, 0.2, 0.3])
        assert block["summary"]["ttft_s"]["median"] == pytest.approx(0.2)
        assert block["summary"]["ttft_s"]["max"] == pytest.approx(0.3)

    def test_every_prompt_is_summarized_separately(self):
        report = runner.run_bench(
            lambda p: scripted(0.1, [0.02] * 4), {"a": "1", "b": "2"}, runs=2, warmup=0
        )
        assert set(report["results"]) == {"a", "b"}
        assert report["results"]["a"]["summary"]["requests"] == 2

    def test_itl_is_pooled_across_runs_and_failures_are_counted_and_excluded(self):
        results = iter(
            [
                scripted(0.1, [0.01, 0.03]),
                scripted(0.1, [0.02, 0.02]),
                scripted(0.1, [], error="http 500"),
            ]
        )
        report = runner.run_bench(lambda p: next(results), {"p": "x"}, runs=3, warmup=0)
        summary = report["results"]["p"]["summary"]
        assert summary["failed"] == 1
        assert summary["itl_s"]["n"] == 4
        assert summary["itl_s"]["median"] == pytest.approx(0.02)
        assert summary["ttft_s"]["n"] == 2

    def test_a_run_that_still_hits_the_prompt_cache_is_visible(self):
        def send(payload):
            result = scripted(0.1, [0.01, 0.01])
            result.usage["prompt_tokens_details"] = {"cached_tokens": 12}
            return result

        summary = runner.run_bench(send, {"p": "x"}, runs=2, warmup=0)["results"]["p"]["summary"]
        assert summary["cached_tokens_seen"] == 24

    def test_at_least_one_measured_run_is_required(self):
        with pytest.raises(ValueError):
            runner.run_bench(lambda p: scripted(0.1, [0.01]), {"p": "x"}, runs=0)

    def test_the_report_is_json_serializable(self):
        report = runner.run_bench(
            lambda p: scripted(0.1, [0.01, 0.01]), {"p": "x"}, runs=2, warmup=0
        )
        assert json.loads(json.dumps(report))["settings"]["cold_prompt_cache"] is True


class TestEndToEnd:
    def test_over_http_the_ttft_respects_the_servers_delay_and_cold_runs_send_the_flag(self):
        with FakeServer(Behavior(first_delay_s=0.1, interval_s=0.01, n_tokens=6)) as server:
            send, client = runner.http_sender(server.url)
            try:
                report = runner.run_bench(send, {"p": "hello"}, runs=3, warmup=1)
            finally:
                client.close()
            assert server.last_request["cache_prompt"] is False
        summary = report["results"]["p"]["summary"]
        assert summary["failed"] == 0
        assert summary["ttft_s"]["min"] >= 0.1
        assert summary["prompt_tokens"] == [20]


class TestEnvironment:
    def test_describe_never_raises_when_nothing_can_be_read(self, monkeypatch, tmp_path):
        monkeypatch.setattr(env, "_run", lambda *a, **k: None)
        record = env.describe(tmp_path / "missing.gguf", tmp_path / "no-repo")
        assert record["machine"]["chip"] is None
        assert record["conditions"]["thermal"] is None
        assert record["model"] == {"file": "missing.gguf", "sha256": None}
        assert record["software"]["llama_cpp_commit"] is None

    def test_a_published_digest_beside_the_file_is_used_instead_of_rehashing(self, tmp_path):
        model = tmp_path / "m.gguf"
        model.write_bytes(b"not the digest")
        (tmp_path / "m.gguf.sha256").write_text("abc123  m.gguf\n")
        assert env.sha256_of(model) == "abc123"

    def test_without_a_sidecar_the_file_is_hashed(self, tmp_path):
        model = tmp_path / "m.gguf"
        model.write_bytes(b"abc")
        assert (
            env.sha256_of(model)
            == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        )

    def test_a_missing_file_has_no_digest(self, tmp_path):
        assert env.sha256_of(tmp_path / "nope.gguf") is None

    def test_a_real_describe_has_the_expected_shape(self):
        record = env.describe()
        assert set(record) == {"machine", "conditions", "software", "model"}
        assert record["machine"]["arch"]


class TestPreflight:
    def quiet(
        self,
        monkeypatch,
        *,
        load=0.4,
        power="Now drawing from 'AC Power'",
        thermal=None,
        gpu=None,
    ):
        monkeypatch.setattr(env.os, "cpu_count", lambda: 8)
        monkeypatch.setattr(env.os, "getloadavg", lambda: (load, load, load))
        monkeypatch.setattr(env, "power_source", lambda: power)
        monkeypatch.setattr(env, "gpu_utilization", lambda *a, **k: gpu)
        monkeypatch.setattr(env, "memory_free_percent", lambda: 80)
        monkeypatch.setattr(
            env,
            "thermal_state",
            lambda: thermal or "Note: No thermal warning level has been recorded",
        )

    def test_a_quiet_plugged_in_machine_passes(self, monkeypatch):
        self.quiet(monkeypatch)
        assert env.preflight() == []

    def test_a_busy_machine_is_refused_at_half_a_load_per_core(self, monkeypatch):
        self.quiet(monkeypatch, load=4.01)
        assert "load average 4.0" in env.preflight()[0]
        self.quiet(monkeypatch, load=3.99)
        assert env.preflight() == []

    def test_battery_is_refused_unless_ac_is_not_required(self, monkeypatch):
        self.quiet(monkeypatch, power="Now drawing from 'Battery Power'")
        assert env.preflight() == ["running on battery power"]
        assert env.preflight(require_ac=False) == []

    @pytest.mark.parametrize(
        "thermal", ["CPU_Speed_Limit = 80", "CPU_Scheduler_Limit = 50\nCPU_Speed_Limit = 100"]
    )
    def test_throttling_is_refused(self, monkeypatch, thermal):
        self.quiet(monkeypatch, thermal=thermal)
        assert any("thermal throttling" in p for p in env.preflight())

    def test_full_speed_limits_are_fine(self, monkeypatch):
        self.quiet(monkeypatch, thermal="CPU_Speed_Limit = 100\nCPU_Scheduler_Limit = 100")
        assert env.preflight() == []

    def test_a_recorded_thermal_warning_is_refused(self, monkeypatch):
        self.quiet(monkeypatch, thermal="Note: Thermal warning level has been recorded: 2")
        assert any("thermal warning" in p for p in env.preflight())

    def test_every_problem_is_reported_not_just_the_first(self, monkeypatch):
        self.quiet(monkeypatch, load=9, power="Now drawing from 'Battery Power'")
        assert len(env.preflight()) == 2

    def test_unreadable_power_and_thermal_state_do_not_block_or_crash(self, monkeypatch):
        monkeypatch.setattr(env.os, "getloadavg", lambda: (0.1, 0.1, 0.1))
        monkeypatch.setattr(env, "power_source", lambda: None)
        monkeypatch.setattr(env, "thermal_state", lambda: None)
        monkeypatch.setattr(env, "gpu_utilization", lambda *a, **k: None)
        monkeypatch.setattr(env, "memory_free_percent", lambda: None)
        assert env.preflight() == []


IOREG = (
    '"PerformanceStatistics" = {"Tiler Utilization %"=0,"Renderer Utilization %"=0,'
    '"Device Utilization %"=@@,"SplitSceneCount"=0}'
)


class TestGpuUtilization:
    def feed(self, monkeypatch, values):
        readings = iter(IOREG.replace("@@", str(v)) for v in values)
        monkeypatch.setattr(env, "_run", lambda *a, **k: next(readings))
        monkeypatch.setattr(env.time, "sleep", lambda _s: None)

    def test_peak_mean_and_sample_count_are_computed(self, monkeypatch):
        self.feed(monkeypatch, [0, 51, 0, 0])
        assert env.gpu_utilization(samples=4) == {"peak": 51, "mean": 12.75, "samples": 4}

    def test_an_unreadable_gpu_is_none_not_zero(self, monkeypatch):
        monkeypatch.setattr(env, "_run", lambda *a, **k: None)
        assert env.gpu_utilization(samples=3) is None

    def test_output_without_the_field_is_none(self, monkeypatch):
        monkeypatch.setattr(env, "_run", lambda *a, **k: "something else entirely")
        monkeypatch.setattr(env.time, "sleep", lambda _s: None)
        assert env.gpu_utilization(samples=3) is None


class TestGpuPreflight:
    def quiet(self, monkeypatch, gpu):
        monkeypatch.setattr(env.os, "cpu_count", lambda: 8)
        monkeypatch.setattr(env.os, "getloadavg", lambda: (0.4, 0.4, 0.4))
        monkeypatch.setattr(env, "power_source", lambda: "Now drawing from 'AC Power'")
        monkeypatch.setattr(env, "thermal_state", lambda: "Note: No thermal warning level")
        monkeypatch.setattr(env, "gpu_utilization", lambda *a, **k: gpu)
        monkeypatch.setattr(env, "memory_free_percent", lambda: 80)

    def test_a_brief_burst_above_the_peak_limit_is_refused(self, monkeypatch):
        """Video playback: 51% in one sample, 0% in the rest. The load average cannot see this."""
        self.quiet(monkeypatch, {"peak": 51, "mean": 2.6, "samples": 20})
        assert any("GPU is busy" in p for p in env.preflight())

    def test_a_steadily_busy_gpu_is_refused_even_without_a_high_peak(self, monkeypatch):
        self.quiet(monkeypatch, {"peak": 20, "mean": 9.0, "samples": 20})
        assert any("GPU is busy" in p for p in env.preflight())

    def test_an_idle_gpu_passes(self, monkeypatch):
        self.quiet(monkeypatch, {"peak": 4, "mean": 0.5, "samples": 20})
        assert env.preflight() == []

    def test_the_limits_are_inclusive_at_the_boundary(self, monkeypatch):
        self.quiet(monkeypatch, {"peak": 25, "mean": 5.0, "samples": 20})
        assert env.preflight() == []

    def test_an_unreadable_gpu_does_not_block(self, monkeypatch):
        self.quiet(monkeypatch, None)
        assert env.preflight() == []


class TestStability:
    def runs(self, decode_rates, ttft=0.1):
        out = []
        for rate in decode_rates:
            gap = 1 / rate
            out.append(runner.metrics(scripted(ttft, [gap] * 10)))
        return out

    def test_repeatable_runs_are_stable(self):
        summary = runner.summarize_runs(self.runs([150, 152, 148, 151, 149]))
        assert summary["stable"] is True
        assert summary["cv_decode"] < 0.02

    def test_one_collapsed_run_makes_the_set_unstable(self):
        """The real failure: decode fell from about 150 to 25 to 48 tokens/s mid-benchmark."""
        summary = runner.summarize_runs(self.runs([149, 36, 48, 44, 25]))
        assert summary["stable"] is False
        assert summary["cv_decode"] > runner.MAX_DECODE_CV

    def test_the_limit_is_five_percent_for_decode(self):
        steady = runner.summarize_runs(self.runs([100, 105, 95]))
        assert steady["cv_decode"] < 0.05 and steady["stable"] is True
        loose = runner.summarize_runs(self.runs([100, 115, 85]))
        assert loose["cv_decode"] > 0.05 and loose["stable"] is False

    def test_a_noisy_time_to_first_token_alone_is_enough(self):
        results = iter([scripted(t, [0.01] * 5) for t in (0.05, 0.10, 0.20)])
        report = runner.run_bench(lambda p: next(results), {"p": "x"}, runs=3, warmup=0)
        assert report["results"]["p"]["summary"]["stable"] is False
        assert report["stable"] is False

    def test_a_single_run_cannot_be_judged_and_is_not_called_unstable(self):
        summary = runner.summarize_runs(self.runs([150]))
        assert summary["cv_decode"] is None and summary["stable"] is True

    def test_the_report_is_stable_only_if_every_prompt_is(self):
        good = iter([scripted(0.1, [0.0066] * 10) for _ in range(3)])
        bad = iter([scripted(t, [0.0066] * 10) for t in (0.05, 0.1, 0.3)])
        sends = {"a": good, "b": bad}
        calls = iter(["a", "a", "a", "b", "b", "b"])
        report = runner.run_bench(
            lambda p: next(sends[next(calls)]), {"a": "1", "b": "2"}, runs=3, warmup=0
        )
        assert report["results"]["a"]["summary"]["stable"] is True
        assert report["results"]["b"]["summary"]["stable"] is False
        assert report["stable"] is False


class TestMemoryPreflight:
    def quiet(self, monkeypatch, free):
        monkeypatch.setattr(env.os, "cpu_count", lambda: 8)
        monkeypatch.setattr(env.os, "getloadavg", lambda: (0.4, 0.4, 0.4))
        monkeypatch.setattr(env, "power_source", lambda: "Now drawing from 'AC Power'")
        monkeypatch.setattr(env, "thermal_state", lambda: "Note: No thermal warning level")
        monkeypatch.setattr(env, "gpu_utilization", lambda *a, **k: None)
        monkeypatch.setattr(env, "memory_free_percent", lambda: free)

    def test_the_free_percentage_is_parsed_from_memory_pressure_output(self, monkeypatch):
        text = (
            "The system has 17179869184 (1048576 pages)\nSystem-wide memory free percentage: 56%\n"
        )
        monkeypatch.setattr(env, "_run", lambda *a, **k: text)
        assert env.memory_free_percent() == 56

    def test_unparseable_or_missing_output_is_none(self, monkeypatch):
        monkeypatch.setattr(env, "_run", lambda *a, **k: "no such line")
        assert env.memory_free_percent() is None
        monkeypatch.setattr(env, "_run", lambda *a, **k: None)
        assert env.memory_free_percent() is None

    def test_swap_used_is_parsed(self, monkeypatch):
        line = "vm.swapusage: total = 3072.00M  used = 2311.56M  free = 760.44M  (encrypted)"
        monkeypatch.setattr(env, "_run", lambda *a, **k: line)
        assert env.swap_used_mb() == 2311.56

    def test_low_free_memory_is_refused_at_the_boundary(self, monkeypatch):
        self.quiet(monkeypatch, 29)
        assert any("low free memory: 29%" in p for p in env.preflight())
        self.quiet(monkeypatch, 30)
        assert env.preflight() == []

    def test_an_unreadable_memory_state_does_not_block(self, monkeypatch):
        self.quiet(monkeypatch, None)
        assert env.preflight() == []


class TestBusyProcesses:
    PS = (
        "  100     1   0.5 /sbin/launchd\n"
        "  200   100  80.0 /Applications/Xcode.app/Contents/Developer/clang\n"
        "  300   100   5.0 /usr/sbin/mdworker\n"
        "  400   200  50.0 /tmp/bench/membw\n"
        "  500     1  12.5 /Applications/Some App.app/Contents/MacOS/Some App\n"
        "garbage line\n"
    )

    def test_the_busiest_come_first_with_names_not_paths(self, monkeypatch):
        monkeypatch.setattr(env, "_run", lambda *a, **k: self.PS)
        top = env.busy_processes(n=3)
        assert [p["name"] for p in top] == ["clang", "membw", "Some App"]
        assert top[0]["cpu_percent"] == 80.0

    def test_the_benchmarks_own_process_tree_is_excluded(self, monkeypatch):
        monkeypatch.setattr(env, "_run", lambda *a, **k: self.PS)
        top = env.busy_processes(n=5, exclude_tree_of=200)  # 200 and its child 400
        assert [p["name"] for p in top] == ["Some App", "mdworker", "launchd"]

    def test_an_unreadable_process_list_is_none(self, monkeypatch):
        monkeypatch.setattr(env, "_run", lambda *a, **k: None)
        assert env.busy_processes() is None

    def test_malformed_lines_are_skipped(self, monkeypatch):
        monkeypatch.setattr(env, "_run", lambda *a, **k: "x y z\n  1 1 nope /bin/a\n")
        assert env.busy_processes() == []
