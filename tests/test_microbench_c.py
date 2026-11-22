"""Compile the C microbenchmarks and check what they compute, not how fast they are.

Speed is measured on Day 2 under controlled conditions. Here every program must be correct: the
STREAM arrays validated, the pointer chain one full cycle, every matrix-multiply variant equal to a
double-precision reference, the memory-mapped checksum equal to one computed independently in
Python, and a copy byte-identical to its source.
"""

from __future__ import annotations

import hashlib
import json
import platform
import random
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "microbench"

pytestmark = pytest.mark.skipif(
    shutil.which("make") is None or shutil.which("cc") is None,
    reason="needs make and a C compiler",
)


@pytest.fixture(scope="module")
def bins(tmp_path_factory) -> Path:
    build = tmp_path_factory.mktemp("microbench-build")
    result = subprocess.run(
        ["make", "-C", str(SOURCES), f"BUILD={build}"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    return build


def run(binary: Path, *args: str, check: bool = True) -> dict:
    proc = subprocess.run([str(binary), *args], capture_output=True, text=True, timeout=120)
    if check:
        assert proc.returncode == 0, (proc.stdout, proc.stderr)
    return json.loads(proc.stdout) if proc.stdout.strip() else {"returncode": proc.returncode}


class TestMemoryBandwidth:
    def test_the_arrays_validate_after_every_kernel_has_run(self, bins):
        out = run(bins / "membw", "--array-mib", "2", "--threads", "3", "--reps", "4")
        assert out["validated"] is True
        assert set(out["kernels"]) == {"copy", "scale", "add", "triad"}
        for rates in out["kernels"].values():
            assert 0 < rates["gbs_min"] <= rates["gbs_median"] <= rates["gbs_best"]

    def test_the_thread_count_and_size_are_echoed(self, bins):
        out = run(bins / "membw", "--selftest", "--threads", "4")
        assert (out["threads"], out["array_mib"]) == (4, 1)

    def test_a_single_repetition_is_refused_because_the_first_is_discarded(self, bins):
        assert run(bins / "membw", "--reps", "1", check=False)["returncode"] == 2


class TestPointerChase:
    def test_the_permutation_is_one_cycle_and_the_chain_visits_every_node(self, bins):
        assert run(bins / "chase", "--selftest")["selftest"] == "ok"

    def test_a_small_sweep_has_increasing_sizes_that_are_whole_nodes(self, bins):
        out = run(bins / "chase", "--min-kib", "4", "--max-mib", "1", "--per-octave", "2")
        sizes = [p["bytes"] for p in out["points"]]
        assert sizes == sorted(set(sizes)) and len(sizes) >= 14
        assert all(size % out["node_bytes"] == 0 for size in sizes)
        assert all(p["ns_min"] > 0 and p["ns_min"] <= p["ns_median"] for p in out["points"])

    def test_the_sweep_is_deterministic_in_its_structure_for_a_given_seed(self, bins):
        a = run(bins / "chase", "--max-mib", "1", "--seed", "5")
        b = run(bins / "chase", "--max-mib", "1", "--seed", "5")
        assert [p["bytes"] for p in a["points"]] == [p["bytes"] for p in b["points"]]


class TestMatmul:
    def test_every_variant_matches_a_double_precision_reference(self, bins):
        out = run(bins / "matmul", "--sizes", "40,97", "--blocks", "8,32", "--reps", "1")
        expected = {"naive", "ikj", "blocked8", "blocked32"}
        if platform.system() == "Darwin":
            expected.add("accelerate")
        for result in out["results"]:
            assert set(result["variants"]) == expected
            for name, v in result["variants"].items():
                assert v["ok"] is True and v["rel_err"] < out["tolerance"], name
                assert v["gflops_best"] > 0

    def test_a_size_that_is_not_a_multiple_of_the_block_still_computes_correctly(self, bins):
        out = run(bins / "matmul", "--sizes", "50", "--blocks", "32", "--variants", "blocked")
        assert out["results"][0]["variants"]["blocked32"]["ok"] is True

    def test_the_naive_variant_can_be_skipped_for_large_sizes(self, bins):
        out = run(bins / "matmul", "--sizes", "64", "--naive-max", "32", "--variants", "naive,ikj")
        assert "naive" not in out["results"][0]["variants"]
        assert "ikj" in out["results"][0]["variants"]


class TestMemoryMappedFile:
    @pytest.fixture
    def data_file(self, tmp_path) -> tuple[Path, bytes]:
        data = random.Random(3).randbytes(4096 * 37 + 123)  # not a whole number of pages
        path = tmp_path / "blob.bin"
        path.write_bytes(data)
        return path, data

    @pytest.mark.parametrize("stride", [1, 7, 4096, 16384])
    def test_the_checksum_equals_one_computed_independently(self, bins, data_file, stride):
        path, data = data_file
        out = run(bins / "mmap_touch", str(path), "--mode", "touch", "--stride", str(stride))
        assert out["checksum"] == sum(data[::stride])
        assert out["bytes"] == len(data)

    @pytest.mark.parametrize("advise", ["none", "willneed", "sequential"])
    def test_a_prefetch_hint_changes_how_pages_arrive_not_what_is_read(
        self, bins, data_file, advise
    ):
        path, data = data_file
        out = run(bins / "mmap_touch", str(path), "--stride", "4096", "--advise", advise)
        assert out["advise"] == advise
        assert out["checksum"] == sum(data[::4096])

    def test_read_mode_reports_every_byte(self, bins, data_file):
        path, data = data_file
        out = run(bins / "mmap_touch", str(path), "--mode", "read")
        assert out["bytes"] == out["checksum"] == len(data)

    def test_a_missing_file_fails(self, bins, tmp_path):
        assert run(bins / "mmap_touch", str(tmp_path / "nope"), check=False)["returncode"] == 2


class TestCopyNoCache:
    def test_the_copy_is_byte_identical(self, bins, tmp_path):
        data = random.Random(9).randbytes((4 << 20) + 4097)  # spans more than one 4 MiB chunk
        src, dst = tmp_path / "src.bin", tmp_path / "dst.bin"
        src.write_bytes(data)
        out = run(bins / "copy_nocache", str(src), str(dst))
        assert out["bytes"] == len(data)
        assert hashlib.sha256(dst.read_bytes()).digest() == hashlib.sha256(data).digest()

    def test_an_existing_destination_is_replaced_not_appended_to(self, bins, tmp_path):
        src, dst = tmp_path / "src.bin", tmp_path / "dst.bin"
        src.write_bytes(b"short")
        dst.write_bytes(b"a much longer previous file")
        run(bins / "copy_nocache", str(src), str(dst))
        assert dst.read_bytes() == b"short"

    def test_a_missing_source_fails_and_creates_nothing_useful(self, bins, tmp_path):
        result = run(
            bins / "copy_nocache", str(tmp_path / "nope"), str(tmp_path / "out"), check=False
        )
        assert result["returncode"] == 2
