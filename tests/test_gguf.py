"""The GGUF header reader, on files built here, including files built to be wrong."""

from __future__ import annotations

import struct

import pytest

from inferlab import gguf, kvmem
from inferlab.gguf import GGUFError

F32, Q4_K = 0, 12


def s(text: str) -> bytes:
    raw = text.encode()
    return struct.pack("<Q", len(raw)) + raw


def kv_u32(key: str, value: int) -> bytes:
    return s(key) + struct.pack("<II", 4, value)


def kv_string_array(key: str, items: list[str]) -> bytes:
    body = b"".join(s(i) for i in items)
    return s(key) + struct.pack("<IIQ", 9, 8, len(items)) + body


def kv_int_array(key: str, items: list[int]) -> bytes:
    return s(key) + struct.pack("<IIQ", 9, 4, len(items)) + struct.pack(f"<{len(items)}I", *items)


def tensor_record(name: str, dims: tuple[int, ...], type_id: int, offset: int) -> bytes:
    return (
        s(name)
        + struct.pack("<I", len(dims))
        + struct.pack(f"<{len(dims)}Q", *dims)
        + struct.pack("<IQ", type_id, offset)
    )


def build(
    tensors: list[tuple[str, tuple[int, ...], int]],
    metadata: list[bytes] | None = None,
    *,
    alignment: int = 32,
    version: int = 3,
    tensor_count: int | None = None,
    metadata_count: int | None = None,
    pad_data: bool = True,
) -> bytes:
    metadata = metadata if metadata is not None else [kv_u32("general.alignment", alignment)]
    records, offset, data = [], 0, b""
    for name, dims, type_id in tensors:
        elements = 1
        for d in dims:
            elements *= d
        _, block_elements, block_bytes = gguf.TYPES[type_id]
        size = elements // block_elements * block_bytes
        records.append(tensor_record(name, dims, type_id, offset))
        padded = (size + alignment - 1) // alignment * alignment
        data += bytes(padded)
        offset += padded
    head = (
        b"GGUF"
        + struct.pack(
            "<IQQ",
            version,
            len(tensors) if tensor_count is None else tensor_count,
            len(metadata) if metadata_count is None else metadata_count,
        )
        + b"".join(metadata)
        + b"".join(records)
    )
    if pad_data:
        head += bytes((-len(head)) % alignment)
    return head + data


@pytest.fixture
def write(tmp_path):
    def _write(raw: bytes):
        path = tmp_path / "m.gguf"
        path.write_bytes(raw)
        return path

    return _write


TENSORS = [("a.weight", (256, 3), Q4_K), ("b.bias", (7,), F32), ("c.weight", (5, 2), F32)]


def test_a_well_formed_file_is_read_back_exactly(write):
    raw = build(
        TENSORS,
        [
            kv_u32("general.alignment", 32),
            kv_string_array("tokenizer.tokens", ["a", "bb", "ccc"]),
            kv_int_array("tokenizer.ids", [1, 2, 3, 4]),
            s("general.name") + struct.pack("<I", 8) + s("tiny"),
        ],
    )
    g = gguf.read_header(write(raw))
    assert g.version == 3 and g.alignment == 32
    assert g.metadata["general.name"] == "tiny"
    assert g.metadata["tokenizer.tokens"] == {"array_of": 8, "length": 3}
    assert g.metadata["tokenizer.ids"] == {"array_of": 4, "length": 4}
    assert [t.name for t in g.tensors] == ["a.weight", "b.bias", "c.weight"]
    assert g.by_name["a.weight"].dims == (256, 3)
    assert g.parameters == 256 * 3 + 7 + 10
    assert g.by_name["a.weight"].nbytes == 144 * 3  # one Q4_K block per 256, three rows
    assert g.data_offset % 32 == 0
    assert g.padded_bytes() == g.data_bytes
    assert g.bytes_by_type() == {"Q4_K": 432, "F32": 28 + 40}


def test_the_block_table_sizes_follow_the_padding_rule(write):
    g = gguf.read_header(write(build([("x", (3,), F32)])))
    assert g.by_name["x"].nbytes == 12 and g.padded_bytes() == 32 and g.data_bytes == 32


def test_a_larger_alignment_is_honoured(write):
    g = gguf.read_header(write(build(TENSORS, alignment=64)))
    assert g.alignment == 64 and g.data_offset % 64 == 0 and g.padded_bytes() == g.data_bytes


def test_every_truncation_of_a_valid_file_is_a_clean_refusal(write):
    raw = build(TENSORS, [kv_u32("general.alignment", 32), kv_string_array("t", ["a", "bb"])])
    whole = gguf.read_header(write(raw))
    needed = whole.data_offset + max(t.offset + t.nbytes for t in whole.tensors)
    assert needed < len(raw)  # the file ends in padding, which a reader may do without
    for cut in range(needed):
        with pytest.raises(GGUFError):
            gguf.read_header(write(raw[:cut]))
    for cut in (needed, len(raw)):
        assert gguf.read_header(write(raw[:cut])).parameters == whole.parameters


def test_a_file_without_the_magic_is_refused(write):
    with pytest.raises(GGUFError, match="magic"):
        gguf.read_header(write(b"GGML" + build(TENSORS)[4:]))


@pytest.mark.parametrize("version", [0, 1, 4, 99])
def test_an_unsupported_version_is_refused(write, version):
    with pytest.raises(GGUFError, match="version"):
        gguf.read_header(write(build(TENSORS, version=version)))


@pytest.mark.parametrize("count", [2**40, 2**62, 2**64 - 1])
def test_a_tensor_count_the_file_cannot_hold_is_refused_without_looping(write, count):
    with pytest.raises(GGUFError, match="larger than the file"):
        gguf.read_header(write(build(TENSORS, tensor_count=count)))


@pytest.mark.parametrize("count", [2**40, 2**64 - 1])
def test_a_metadata_count_the_file_cannot_hold_is_refused_without_looping(write, count):
    with pytest.raises(GGUFError, match="larger than the file"):
        gguf.read_header(write(build(TENSORS, metadata_count=count)))


def test_a_string_longer_than_the_file_is_refused(write):
    bad = struct.pack("<Q", 2**40) + b"x"
    with pytest.raises(GGUFError):
        gguf.read_header(write(build(TENSORS, [bad + struct.pack("<II", 4, 1)])))


def test_an_array_that_claims_more_items_than_the_file_can_hold_is_refused(write):
    huge = s("k") + struct.pack("<IIQ", 9, 4, 2**50)
    with pytest.raises(GGUFError, match="array"):
        gguf.read_header(write(build(TENSORS, [huge])))
    huge_strings = s("k") + struct.pack("<IIQ", 9, 8, 2**50)
    with pytest.raises(GGUFError, match="array"):
        gguf.read_header(write(build(TENSORS, [huge_strings])))


def test_arrays_nested_too_deeply_are_refused(write):
    inner = struct.pack("<IQ", 9, 1)  # an array whose one item is an array ...
    value = s("k") + struct.pack("<IIQ", 9, 9, 1) + inner * 4
    with pytest.raises(GGUFError):
        gguf.read_header(write(build(TENSORS, [value])))


def test_an_unknown_metadata_or_array_type_is_refused(write):
    with pytest.raises(GGUFError, match="type"):
        gguf.read_header(write(build(TENSORS, [s("k") + struct.pack("<I", 77) + b"\0" * 8])))
    with pytest.raises(GGUFError, match="element type"):
        gguf.read_header(write(build(TENSORS, [s("k") + struct.pack("<IIQ", 9, 77, 0)])))


def test_a_key_that_is_not_utf8_is_refused(write):
    bad_key = struct.pack("<Q", 2) + b"\xff\xfe" + struct.pack("<II", 4, 1)
    with pytest.raises(GGUFError, match="UTF-8"):
        gguf.read_header(write(build(TENSORS, [bad_key])))


@pytest.mark.parametrize("dims", [(), (1, 2, 3, 4, 5)])
def test_a_tensor_with_no_or_too_many_dimensions_is_refused(write, dims):
    record = tensor_record("t", (4,), F32, 0)
    bad = tensor_record("t", dims, F32, 0)  # with no dimensions this is a zero count then the rest
    raw = build([("t", (4,), F32)]).replace(record, bad)
    with pytest.raises(GGUFError, match="dimensions"):
        gguf.read_header(write(raw))


def test_an_empty_dimension_is_refused(write):
    raw = build([("t", (4,), F32)]).replace(
        tensor_record("t", (4,), F32, 0), tensor_record("t", (0,), F32, 0)
    )
    with pytest.raises(GGUFError, match="empty"):
        gguf.read_header(write(raw))


def test_an_unknown_ggml_type_is_refused(write):
    raw = build([("t", (4,), F32)]).replace(
        tensor_record("t", (4,), F32, 0), tensor_record("t", (4,), 99, 0)
    )
    with pytest.raises(GGUFError, match="ggml type"):
        gguf.read_header(write(raw))


def test_a_row_that_is_not_whole_blocks_is_refused(write):
    raw = build([("t", (256,), Q4_K)]).replace(
        tensor_record("t", (256,), Q4_K, 0), tensor_record("t", (100,), Q4_K, 0)
    )
    with pytest.raises(GGUFError, match="whole number of blocks"):
        gguf.read_header(write(raw))


def test_duplicate_tensor_names_are_refused(write):
    with pytest.raises(GGUFError, match="share a name"):
        gguf.read_header(write(build([("t", (4,), F32), ("t", (4,), F32)])))


def test_a_misaligned_tensor_offset_is_refused(write):
    raw = build([("a", (4,), F32), ("b", (4,), F32)])
    raw = raw.replace(tensor_record("b", (4,), F32, 32), tensor_record("b", (4,), F32, 33))
    with pytest.raises(GGUFError, match="aligned"):
        gguf.read_header(write(raw))


def test_a_tensor_that_runs_past_the_end_of_the_file_is_refused(write):
    raw = build([("a", (4,), F32)])  # 16 bytes of data, padded to 32
    with pytest.raises(GGUFError, match="past the end"):
        gguf.read_header(write(raw[:-17]))


@pytest.mark.parametrize("alignment", [0, 3, 48, -4])
def test_a_bad_alignment_is_refused(write, alignment):
    raw = build(TENSORS, [s("general.alignment") + struct.pack("<Ii", 5, alignment)])
    with pytest.raises(GGUFError, match="alignment"):
        gguf.read_header(write(raw))


# ------------------------------------------------------- expected tensors


def tiny_model(tied: bool) -> kvmem.Model:
    return kvmem.load_model(
        {
            "model_type": "qwen2",
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 64,
            "vocab_size": 50,
            "tie_word_embeddings": tied,
            "use_sliding_window": False,
        }
    )


def as_file(m: kvmem.Model, write, *, drop=None, extra=None, reshape=None):
    shapes = dict(gguf.expected_tensors(m))
    if drop:
        del shapes[drop]
    if reshape:
        shapes[reshape[0]] = reshape[1]
    if extra:
        shapes[extra] = (8,)
    return gguf.read_header(write(build([(n, d, F32) for n, d in shapes.items()])))


@pytest.mark.parametrize("tied", [True, False])
def test_a_file_laid_out_as_the_calculator_expects_compares_clean(write, tied):
    m = tiny_model(tied)
    g = as_file(m, write)
    assert gguf.compare_tensors(gguf.expected_tensors(m), g) == []
    assert g.parameters == kvmem.param_counts(m).total
    assert ("output.weight" in g.by_name) == (not tied)


def test_a_missing_extra_or_reshaped_tensor_is_reported_by_name(write):
    m = tiny_model(True)
    expected = gguf.expected_tensors(m)
    missing = gguf.compare_tensors(expected, as_file(m, write, drop="blk.1.ffn_up.weight"))
    assert missing == ["missing from the file: blk.1.ffn_up.weight"]
    extra = gguf.compare_tensors(expected, as_file(m, write, extra="rope_freqs.weight"))
    assert extra == ["in the file but not expected: rope_freqs.weight (8,)"]
    wrong = gguf.compare_tensors(
        expected, as_file(m, write, reshape=("blk.0.attn_k.weight", (32, 32)))
    )
    assert len(wrong) == 1 and "blk.0.attn_k.weight" in wrong[0]


def test_only_the_qwen2_layout_is_known():
    other = kvmem.load_model(
        {
            "model_type": "mixtral",
            "hidden_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 16,
            "vocab_size": 10,
            "tie_word_embeddings": False,
            "num_local_experts": 2,
            "num_experts_per_tok": 1,
        }
    )
    with pytest.raises(ValueError):
        gguf.expected_tensors(other)


# ------------------------------------------------------------------ summaries


def test_a_summary_is_plain_json_and_round_trips_to_an_equivalent_header(write):
    import json

    raw = build(
        TENSORS,
        [
            kv_u32("general.alignment", 32),
            s("general.name") + struct.pack("<I", 8) + s("tiny"),
            s("general.description") + struct.pack("<I", 8) + s("x" * 500),
            kv_string_array("tokenizer.tokens", ["a", "b"]),
        ],
    )
    g = gguf.read_header(write(raw))
    summary = gguf.to_summary(g)
    assert json.loads(json.dumps(summary)) == summary
    assert summary["metadata"]["general.name"] == "tiny"
    assert summary["metadata"]["general.description"] == {"string_chars": 500}
    again = gguf.from_summary(summary)
    assert again.tensors == g.tensors
    assert (again.data_offset, again.file_size, again.alignment) == (
        g.data_offset,
        g.file_size,
        g.alignment,
    )
    assert again.parameters == g.parameters and again.padded_bytes() == g.padded_bytes()
    assert again.data_bytes == g.data_bytes


def good_summary(write):
    return gguf.to_summary(gguf.read_header(write(build(TENSORS))))


def test_a_summary_that_has_been_edited_into_nonsense_is_refused(write):
    base = good_summary(write)
    bad_cases = []
    for field in ("version", "alignment", "data_offset", "file_size", "tensors", "metadata"):
        missing = dict(base)
        del missing[field]
        bad_cases.append(missing)
    first = base["tensors"][0]
    bad_cases += [
        base | {"tensors": [first | {"type_id": 99}]},
        base | {"tensors": [first | {"dims": []}]},
        base | {"tensors": [first | {"dims": [0]}]},
        base | {"tensors": [first | {"dims": [1, 2, 3, 4, 5]}]},
        base | {"tensors": [first | {"dims": [100]}]},  # not whole Q4_K blocks
        base | {"tensors": [first, first]},
        base | {"tensors": [first | {"offset": 3}]},
        base | {"tensors": [first | {"offset": 10**12}]},
        base | {"tensors": [first | {"name": 5}]},
        base | {"tensors": "nope"},
        base | {"alignment": 48},
        base | {"file_size": 10},
        base | {"data_offset": -1},
        base | {"data_offset": True},
        "not a dict",
    ]
    for case in bad_cases:
        with pytest.raises(GGUFError):
            gguf.from_summary(case)
