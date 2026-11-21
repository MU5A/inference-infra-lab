"""The bench command: it must refuse to benchmark a machine in a bad state, and say so."""

from __future__ import annotations

import json
import sys

import pytest

from inferlab import __main__ as cli
from inferlab import env
from inferlab.fakeserver import Behavior, FakeServer


@pytest.fixture
def argv(tmp_path, monkeypatch):
    prompts = tmp_path / "prompts.json"
    prompts.write_text(json.dumps({"p": "hello"}))
    out = tmp_path / "out" / "result.json"

    def make(url: str, *extra: str) -> tuple:
        monkeypatch.setattr(
            sys,
            "argv",
            ["inferlab", "bench", "--base-url", url, "--prompts", str(prompts), "--out", str(out)]
            + ["--runs", "2", "--warmup", "0", "--max-tokens", "4", *extra],
        )
        return (out,)

    return make


def test_a_machine_in_a_bad_state_is_refused_and_nothing_is_sent(argv, monkeypatch, capsys):
    monkeypatch.setattr(env, "preflight", lambda: ["running on battery power"])
    with FakeServer(Behavior(n_tokens=3)) as server:
        (out,) = argv(server.url)
        with pytest.raises(SystemExit) as exit_info:
            cli.main()
        assert server.last_request is None  # no request ever reached the server
    assert exit_info.value.code == 3
    assert "running on battery power" in capsys.readouterr().err
    assert not out.exists()


def test_the_override_runs_and_records_exactly_what_was_overridden(argv, monkeypatch):
    monkeypatch.setattr(env, "preflight", lambda: ["load average 9.0 is above 4.0", "on battery"])
    with FakeServer(Behavior(n_tokens=3, first_delay_s=0.01, interval_s=0.005)) as server:
        (out,) = argv(server.url, "--allow-noisy")
        cli.main()
    report = json.loads(out.read_text())
    assert report["preflight_problems_overridden"] == [
        "load average 9.0 is above 4.0",
        "on battery",
    ]
    assert report["results"]["p"]["summary"]["failed"] == 0


def test_a_clean_machine_runs_and_records_no_overrides(argv, monkeypatch):
    monkeypatch.setattr(env, "preflight", lambda: [])
    with FakeServer(Behavior(n_tokens=3, first_delay_s=0.01, interval_s=0.005)) as server:
        (out,) = argv(server.url)
        cli.main()
    report = json.loads(out.read_text())
    assert report["preflight_problems_overridden"] == []
    assert set(report["environment"]) == {"before", "after", "finished_at"}
    assert report["settings"]["cold_prompt_cache"] is True


def test_the_prompt_cache_stays_off_unless_asked(argv, monkeypatch):
    monkeypatch.setattr(env, "preflight", lambda: [])
    with FakeServer(Behavior(n_tokens=2, first_delay_s=0.01, interval_s=0.005)) as server:
        argv(server.url)
        cli.main()
        assert server.last_request["cache_prompt"] is False
        argv(server.url, "--warm-cache")
        cli.main()
        assert "cache_prompt" not in server.last_request
