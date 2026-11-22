"""Cgroup parsing, tested against fake directories, and the safety guard of the container script."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from inferlab import cgroup

ROOT = Path(__file__).resolve().parents[1]

STAT = """anon 1048576
file 52428800
file_mapped 4096
shmem 0
active_file 100
inactive_file 52428700
pgmajfault 7
"""
EVENTS = "low 0\nhigh 0\nmax 12\noom 1\noom_kill 1\noom_group_kill 0\n"


def fake(tmp_path, **files):
    for name, text in files.items():
        (tmp_path / name.replace("_", ".", 1)).write_text(text)
    return tmp_path


class TestParsing:
    def test_flat_keyed_lines_become_integers(self):
        parsed = cgroup.parse_flat_keyed(STAT)
        assert parsed["file"] == 52428800 and parsed["pgmajfault"] == 7

    def test_malformed_and_empty_input_is_ignored(self):
        assert cgroup.parse_flat_keyed("") == {}
        assert cgroup.parse_flat_keyed("just words\nname notanumber\nok 5\n") == {"ok": 5}

    def test_negative_values_are_kept(self):
        assert cgroup.parse_flat_keyed("delta -3") == {"delta": -3}


class TestSnapshot:
    def test_every_field_is_read(self, tmp_path):
        fake(
            tmp_path,
            memory_current="1000\n",
            memory_peak="2000\n",
            memory_max="100663296\n",
            memory_stat=STAT,
            memory_events=EVENTS,
        )
        (tmp_path / "memory.swap.max").write_text("0\n")
        snap = cgroup.snapshot(tmp_path)
        assert (snap["current"], snap["peak"], snap["limit"], snap["swap_limit"]) == (
            1000,
            2000,
            100663296,
            0,
        )
        assert snap["events"]["oom_kill"] == 1 and snap["stat"]["anon"] == 1048576

    def test_missing_files_give_none_not_errors(self, tmp_path):
        snap = cgroup.snapshot(tmp_path)
        assert snap["current"] is None and snap["limit"] is None
        assert snap["stat"] == {} and snap["events"] == {}


class TestSafetyGuard:
    def test_a_small_limit_is_accepted(self, tmp_path):
        fake(tmp_path, memory_max="100663296")
        assert cgroup.require_limit(tmp_path) == 100663296

    def test_no_limit_is_refused(self, tmp_path):
        fake(tmp_path, memory_max="max")
        with pytest.raises(RuntimeError, match="no memory limit"):
            cgroup.require_limit(tmp_path)

    def test_a_missing_file_is_refused_as_no_limit(self, tmp_path):
        with pytest.raises(RuntimeError, match="no memory limit"):
            cgroup.require_limit(tmp_path)

    def test_a_limit_above_the_ceiling_is_refused(self, tmp_path):
        fake(tmp_path, memory_max=str(2 * cgroup.GIB))
        with pytest.raises(RuntimeError, match="above"):
            cgroup.require_limit(tmp_path)

    def test_the_boundary_is_inclusive(self, tmp_path):
        fake(tmp_path, memory_max=str(cgroup.GIB))
        assert cgroup.require_limit(tmp_path) == cgroup.GIB


def test_the_container_script_refuses_to_run_on_a_machine_without_a_small_limit():
    """The script allocates until the kernel kills something. Run here, it must refuse instead."""
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "linux_memory_demo.py")],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 3
    assert "refusing to run" in proc.stderr
    assert proc.stdout.strip() == ""


def good_run():
    return {
        "info": {"limit_mib": 96.0},
        "page_cache": {
            "file_mib": 48,
            "stages": {
                "after_write": {"file_mib": 48.0},
                "after_drop": {"file_mib": 0.1},
                "after_cold_read": {"file_mib": 47.9},
            },
        },
        "anonymous": {"killed_by_signal": 9, "oom_kill_events": 1, "last_progress_mib": 80},
        "mmap_read": {
            "completed": True,
            "oom_kill_events": 0,
            "peak_current_mib": 95.9,
            "limit_hit_events": 120,
        },
        "mmap_private_write": {
            "completed": False,
            "killed_by_signal": 9,
            "oom_kill_events": 1,
        },
    }


class TestDirtyWritePrediction:
    def arms(self, *, unsynced_killed=True, synced_killed=False, synced_code=0):
        return (
            {"oom_killed": unsynced_killed, "exit_code": 137 if unsynced_killed else 0},
            {"oom_killed": synced_killed, "exit_code": synced_code},
        )

    def test_the_predicted_outcome_has_no_failures(self):
        assert cgroup.check_dirty_write(*self.arms()) == []

    def test_an_unsynced_write_that_survives_is_a_finding(self):
        failures = cgroup.check_dirty_write(*self.arms(unsynced_killed=False))
        assert any("unsynced write was not OOM-killed" in f for f in failures)

    @pytest.mark.parametrize("kwargs", [{"synced_killed": True}, {"synced_code": 1}])
    def test_a_synced_write_that_fails_is_a_finding(self, kwargs):
        failures = cgroup.check_dirty_write(*self.arms(**kwargs))
        assert any("synced write did not complete" in f for f in failures)


class TestMissingResults:
    def test_a_killed_process_is_reported_as_missing_experiments_not_a_crash(self):
        run = good_run()
        del run["mmap_read"], run["mmap_private_write"]
        failures = cgroup.check_run(run)
        assert any("mmap_read: no result" in f for f in failures)
        assert any("mmap_private_write: no result" in f for f in failures)

    def test_an_empty_run_reports_everything_missing(self):
        assert len(cgroup.check_run({})) == 5


class TestPredictions:
    def test_a_run_that_behaves_as_predicted_has_no_failures(self):
        assert cgroup.check_run(good_run()) == []

    @pytest.mark.parametrize(
        ("path", "value", "expect"),
        [
            (("page_cache", "stages", "after_write", "file_mib"), 10.0, "did not stay"),
            (("page_cache", "stages", "after_drop", "file_mib"), 40.0, "did not empty"),
            (("page_cache", "stages", "after_cold_read", "file_mib"), 5.0, "did not refill"),
            (("anonymous", "killed_by_signal"), None, "not OOM-killed"),
            (("anonymous", "oom_kill_events"), 0, "not OOM-killed"),
            (("anonymous", "last_progress_mib"), 120, "more than the limit"),
            (("mmap_read", "completed"), False, "failed"),
            (("mmap_read", "oom_kill_events"), 1, "failed"),
            (("mmap_read", "peak_current_mib"), 101.0, "exceeded the limit"),
            (("mmap_read", "limit_hit_events"), 0, "never reached"),
            (("mmap_private_write", "completed"), True, "not OOM-killed"),
            (("mmap_private_write", "killed_by_signal"), None, "not OOM-killed"),
        ],
    )
    def test_each_violated_prediction_is_reported(self, path, value, expect):
        run = good_run()
        node = run
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value
        failures = cgroup.check_run(run)
        assert failures and any(expect in f for f in failures), failures
