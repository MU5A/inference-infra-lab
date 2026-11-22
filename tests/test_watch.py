"""The interference sampler and the wait-for-quiet loop."""

from __future__ import annotations

import time

from inferlab import env, watch


def busy(*pairs):
    return [{"cpu_percent": cpu, "name": name} for name, cpu in pairs]


class TestSummary:
    def feed(self, monkeypatch, snapshots):
        queue = iter(snapshots)
        monkeypatch.setattr(env, "busy_processes", lambda *a, **k: next(queue))
        monkeypatch.setattr(watch.os, "getloadavg", lambda: (2.5, 2.0, 1.0))
        w = watch.Watch()
        for _ in snapshots:
            w.sample()
        return w

    def test_the_peak_total_and_the_busiest_names_are_reported(self, monkeypatch):
        w = self.feed(
            monkeypatch,
            [busy(("compile", 96.0), ("go", 38.0)), busy(("compile", 20.0)), busy(("mds", 5.0))],
        )
        s = w.summary()
        assert s["samples"] == 3
        assert s["peak_other_cpu_percent"] == 134.0
        assert s["busiest_other_processes"][0] == {"name": "compile", "peak_cpu_percent": 96.0}
        assert s["peak_load1"] == 2.5

    def test_a_peak_above_one_and_a_half_cores_is_marked_contended(self, monkeypatch):
        assert self.feed(monkeypatch, [busy(("a", 151.0))]).summary()["contended"] is True
        assert self.feed(monkeypatch, [busy(("a", 150.0))]).summary()["contended"] is False

    def test_no_samples_means_unknown_not_quiet(self, monkeypatch):
        monkeypatch.setattr(env, "busy_processes", lambda *a, **k: None)
        w = watch.Watch()
        w.sample()
        s = w.summary()
        assert s["samples"] == 0 and s["peak_other_cpu_percent"] is None
        assert s["contended"] is False

    def test_a_process_is_named_once_at_its_worst(self, monkeypatch):
        w = self.feed(monkeypatch, [busy(("go", 10.0)), busy(("go", 90.0)), busy(("go", 30.0))])
        assert w.summary()["busiest_other_processes"] == [{"name": "go", "peak_cpu_percent": 90.0}]

    def test_the_benchmarks_own_tree_is_excluded_from_what_is_asked_for(self, monkeypatch):
        seen = {}

        def fake(n, exclude_tree_of=None):
            seen["exclude"] = exclude_tree_of
            return busy(("x", 1.0))

        monkeypatch.setattr(env, "busy_processes", fake)
        watch.Watch().sample()
        assert seen["exclude"] == watch.os.getpid()


def test_the_sampler_thread_collects_samples_and_stops(monkeypatch):
    monkeypatch.setattr(env, "busy_processes", lambda *a, **k: busy(("x", 1.0)))
    with watch.Watch(interval_s=0.01) as w:
        deadline = time.monotonic() + 3
        while not w.samples and time.monotonic() < deadline:
            time.sleep(0.01)
    count = len(w.samples)
    assert count >= 1
    time.sleep(0.05)
    assert len(w.samples) == count  # stopped: nothing is added after the block ends


class TestWaitQuiet:
    def run(self, answers, limit=100.0):
        calls = {"n": 0, "slept": []}
        now = {"t": 0.0}

        def check():
            calls["n"] += 1
            return answers[min(calls["n"] - 1, len(answers) - 1)]

        def sleep(seconds):
            calls["slept"].append(seconds)
            now["t"] += seconds

        result = watch.wait_quiet(
            limit, poll_s=10.0, check=check, sleep=sleep, clock=lambda: now["t"]
        )
        return result, calls

    def test_it_returns_at_once_when_already_clean(self):
        result, calls = self.run([[]])
        assert result == [] and calls["n"] == 1 and calls["slept"] == []

    def test_it_waits_until_the_machine_settles(self):
        result, calls = self.run([["busy"], ["busy"], []])
        assert result == [] and calls["n"] == 3 and calls["slept"] == [10.0, 10.0]

    def test_it_gives_up_at_the_limit_and_reports_what_is_left(self):
        result, calls = self.run([["still busy"]], limit=30.0)
        assert result == ["still busy"]
        assert sum(calls["slept"]) == 30.0
