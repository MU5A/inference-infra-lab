"""Day 5's model configuration files must be exactly the bytes that were fetched and pinned."""

from __future__ import annotations

import hashlib
import json
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "days" / "05-kv-cache-memory" / "configs"
MANIFEST = CONFIGS / "manifest.json"

pytestmark = pytest.mark.skipif(not MANIFEST.exists(), reason="no Day 5 configs committed")


def entries() -> list[dict]:
    return json.loads(MANIFEST.read_text())["files"]


def test_every_config_is_the_exact_file_that_was_fetched():
    for entry in entries():
        raw = (CONFIGS / entry["file"]).read_bytes()
        assert len(raw) == entry["bytes"], entry["file"]
        assert hashlib.sha256(raw).hexdigest() == entry["sha256"], (
            f"{entry['file']} changed since it was fetched from {entry['url']}"
        )


def test_every_config_is_pinned_to_a_commit_and_a_public_https_url():
    for entry in entries():
        assert re.fullmatch(r"[0-9a-f]{40}", entry["commit"]), entry["file"]
        assert entry["url"].startswith("https://huggingface.co/"), entry["file"]
        assert entry["commit"] in entry["url"], entry["file"]
        assert entry["license"], entry["file"]


def test_no_config_file_is_missing_from_the_manifest_or_the_other_way_round():
    listed = {entry["file"] for entry in entries()}
    on_disk = {p.name for p in CONFIGS.glob("*.config.json")}
    assert listed == on_disk


def test_every_config_is_a_json_object_with_the_fields_a_memory_calculator_reads():
    for entry in entries():
        config = json.loads((CONFIGS / entry["file"]).read_text())
        for field in ("num_hidden_layers", "hidden_size", "num_attention_heads"):
            assert isinstance(config[field], int) and config[field] > 0, (entry["file"], field)
