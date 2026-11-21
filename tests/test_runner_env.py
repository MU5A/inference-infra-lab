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
    def quiet(self, monkeypatch, *, load=0.4, power="Now drawing from 'AC Power'", thermal=None):
        monkeypatch.setattr(env.os, "cpu_count", lambda: 8)
        monkeypatch.setattr(env.os, "getloadavg", lambda: (load, load, load))
        monkeypatch.setattr(env, "power_source", lambda: power)
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
        assert env.preflight() == []
