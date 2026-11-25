"""Read the header of a GGUF model file: its metadata and the shape, type and size of every tensor.

Day 5 uses this to check the memory calculator against the real file, tensor by tensor, without
loading the weights. Only the header is read. The reader refuses a file that does not add up
(wrong magic, a count larger than the file could hold, a tensor that runs past the end) with
`GGUFError`, instead of looping or allocating on a number the file claims.

GGUF layout (version 2 and 3): magic "GGUF", version u32, tensor count u64, metadata count u64,
then the metadata (key string, type u32, value), then one record per tensor (name string,
dimension count u32, dimensions u64 each, ggml type u32, offset u64), then padding to the file's
alignment, then the tensor data. A string is a u64 length and its bytes. Offsets are relative to
the start of the data.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field
from pathlib import Path

from .kvmem import Model

MAGIC = b"GGUF"
DEFAULT_ALIGNMENT = 32
MAX_DIMS = 4  # ggml's own limit
MAX_ARRAY_DEPTH = 2
MAX_STORED_STRING = 4096

# ggml type id: (name, elements per block, bytes per block)
TYPES = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18),
    3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34),
    9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
    15: ("Q8_K", 256, 292),
    24: ("I8", 1, 1),
    25: ("I16", 1, 2),
    26: ("I32", 1, 4),
    27: ("I64", 1, 8),
    28: ("F64", 1, 8),
    30: ("BF16", 1, 2),
}

_SCALARS = {  # metadata value type: (struct format, size)
    0: ("<B", 1),
    1: ("<b", 1),
    2: ("<H", 2),
    3: ("<h", 2),
    4: ("<I", 4),
    5: ("<i", 4),
    6: ("<f", 4),
    7: ("<?", 1),
    10: ("<Q", 8),
    11: ("<q", 8),
    12: ("<d", 8),
}
_STRING, _ARRAY = 8, 9


class GGUFError(ValueError):
    """The file is not a well-formed GGUF file."""


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dims: tuple[int, ...]
    type_id: int
    offset: int

    @property
    def type_name(self) -> str:
        return TYPES[self.type_id][0]

    @property
    def elements(self) -> int:
        count = 1
        for d in self.dims:
            count *= d
        return count

    @property
    def nbytes(self) -> int:
        _, block_elements, block_bytes = TYPES[self.type_id]
        return self.elements // block_elements * block_bytes


@dataclass
class GGUF:
    version: int
    metadata: dict
    tensors: list[TensorInfo]
    alignment: int
    data_offset: int
    file_size: int
    by_name: dict[str, TensorInfo] = field(default_factory=dict)

    @property
    def data_bytes(self) -> int:
        """Size of the tensor-data region as the file lays it out."""
        return self.file_size - self.data_offset

    @property
    def parameters(self) -> int:
        return sum(t.elements for t in self.tensors)

    def padded_bytes(self) -> int:
        """Every tensor's size from the block table, each rounded up to the alignment."""
        return sum(_align(t.nbytes, self.alignment) for t in self.tensors)

    def bytes_by_type(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for t in self.tensors:
            out[t.type_name] = out.get(t.type_name, 0) + t.nbytes
        return out


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


class _Reader:
    def __init__(self, handle, size: int):
        self.handle = handle
        self.size = size

    def remaining(self) -> int:
        return self.size - self.handle.tell()

    def read(self, n: int) -> bytes:
        if n < 0 or n > self.remaining():
            raise GGUFError("the file ends inside a record")
        return self.handle.read(n)

    def skip(self, n: int) -> None:
        if n < 0 or n > self.remaining():
            raise GGUFError("the file ends inside a record")
        self.handle.seek(n, os.SEEK_CUR)

    def unpack(self, fmt: str, size: int):
        return struct.unpack(fmt, self.read(size))[0]

    def u32(self) -> int:
        return self.unpack("<I", 4)

    def u64(self) -> int:
        return self.unpack("<Q", 8)

    def string(self, *, keep: bool = True):
        length = self.u64()
        if length > self.remaining():
            raise GGUFError(f"a string claims {length} bytes, more than the file has left")
        if not keep or length > MAX_STORED_STRING:
            self.skip(length)
            return {"string_bytes": length}
        try:
            return self.read(length).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise GGUFError("a string is not valid UTF-8") from exc


def _value(r: _Reader, type_id: int, depth: int = 0):
    if type_id in _SCALARS:
        fmt, size = _SCALARS[type_id]
        return r.unpack(fmt, size)
    if type_id == _STRING:
        return r.string()
    if type_id == _ARRAY:
        if depth >= MAX_ARRAY_DEPTH:
            raise GGUFError("arrays nested too deeply")
        element = r.u32()
        count = r.u64()
        if element in _SCALARS:
            width = _SCALARS[element][1]
            if count * width > r.remaining():
                raise GGUFError(f"an array claims {count} items, more than the file can hold")
            r.skip(count * width)
        elif element in (_STRING, _ARRAY):
            if count * 8 > r.remaining():
                raise GGUFError(f"an array claims {count} items, more than the file can hold")
            for _ in range(count):
                if element == _STRING:
                    r.string(keep=False)
                else:
                    _value(r, _ARRAY, depth + 1)
        else:
            raise GGUFError(f"unknown array element type {element}")
        return {"array_of": element, "length": count}
    raise GGUFError(f"unknown metadata value type {type_id}")


def read_header(path: str | Path) -> GGUF:
    path = Path(path)
    size = path.stat().st_size
    with path.open("rb") as handle:
        r = _Reader(handle, size)
        if r.read(4) != MAGIC:
            raise GGUFError("not a GGUF file (wrong magic)")
        version = r.u32()
        if version not in (2, 3):
            raise GGUFError(f"unsupported GGUF version {version}")
        n_tensors, n_metadata = r.u64(), r.u64()
        # Smallest possible records: a metadata entry is 8 + 4 bytes and a tensor 8 + 4 + 4 + 8.
        if n_metadata * 12 > r.remaining() or n_tensors * 24 > r.remaining():
            raise GGUFError("the counts in the header are larger than the file could hold")
        metadata = {}
        for _ in range(n_metadata):
            key = r.string()
            if not isinstance(key, str):
                raise GGUFError("a metadata key is implausibly long")
            metadata[key] = _value(r, r.u32())
        infos = []
        for _ in range(n_tensors):
            name = r.string()
            if not isinstance(name, str):
                raise GGUFError("a tensor name is implausibly long")
            n_dims = r.u32()
            if not 1 <= n_dims <= MAX_DIMS:
                raise GGUFError(f"tensor {name!r} has {n_dims} dimensions")
            dims = tuple(r.u64() for _ in range(n_dims))
            if any(d < 1 for d in dims):
                raise GGUFError(f"tensor {name!r} has an empty dimension")
            type_id, offset = r.u32(), r.u64()
            if type_id not in TYPES:
                raise GGUFError(f"tensor {name!r} has unknown ggml type {type_id}")
            infos.append(TensorInfo(name, dims, type_id, offset))
        alignment = metadata.get("general.alignment", DEFAULT_ALIGNMENT)
        if not isinstance(alignment, int) or alignment < 1 or alignment & (alignment - 1):
            raise GGUFError(f"bad alignment {alignment!r}")
        data_offset = _align(handle.tell(), alignment)
    names = [t.name for t in infos]
    if len(set(names)) != len(names):
        raise GGUFError("two tensors share a name")
    for t in infos:
        _, block_elements, _ = TYPES[t.type_id]
        if t.dims[0] % block_elements:
            raise GGUFError(f"tensor {t.name!r}: row length is not a whole number of blocks")
        if t.offset % alignment:
            raise GGUFError(f"tensor {t.name!r} is not aligned")
        if data_offset + t.offset + t.nbytes > size:
            raise GGUFError(f"tensor {t.name!r} runs past the end of the file")
    return GGUF(
        version=version,
        metadata=metadata,
        tensors=infos,
        alignment=alignment,
        data_offset=data_offset,
        file_size=size,
        by_name={t.name: t for t in infos},
    )


# ------------------------------------------------------------ what to expect


def expected_tensors(m: Model) -> dict[str, tuple[int, ...]]:
    """Tensor names and shapes in GGUF's order (input width first) that a dense GQA model of the
    `qwen2` kind should have. Raises ValueError for any other architecture."""
    if m.model_type != "qwen2":
        raise ValueError("expected_tensors knows only the dense qwen2 layout")
    q_out = m.heads * m.head_dim
    kv_out = m.kv_heads * m.head_dim
    shapes: dict[str, tuple[int, ...]] = {
        "token_embd.weight": (m.hidden, m.vocab),
        "output_norm.weight": (m.hidden,),
    }
    if not m.tied:
        shapes["output.weight"] = (m.hidden, m.vocab)
    for i in range(m.layers):
        b = f"blk.{i}."
        shapes |= {
            b + "attn_norm.weight": (m.hidden,),
            b + "attn_q.weight": (m.hidden, q_out),
            b + "attn_q.bias": (q_out,),
            b + "attn_k.weight": (m.hidden, kv_out),
            b + "attn_k.bias": (kv_out,),
            b + "attn_v.weight": (m.hidden, kv_out),
            b + "attn_v.bias": (kv_out,),
            b + "attn_output.weight": (q_out, m.hidden),
            b + "ffn_norm.weight": (m.hidden,),
            b + "ffn_gate.weight": (m.hidden, m.ffn),
            b + "ffn_up.weight": (m.hidden, m.ffn),
            b + "ffn_down.weight": (m.ffn, m.hidden),
        }
    return shapes


def compare_tensors(expected: dict[str, tuple[int, ...]], g: GGUF) -> list[str]:
    """Differences between the calculator's tensors and the file's. Empty means identical."""
    problems = []
    for name in sorted(expected.keys() - g.by_name.keys()):
        problems.append(f"missing from the file: {name}")
    for name in sorted(g.by_name.keys() - expected.keys()):
        problems.append(f"in the file but not expected: {name} {g.by_name[name].dims}")
    for name in sorted(expected.keys() & g.by_name.keys()):
        if g.by_name[name].dims != expected[name]:
            problems.append(f"{name}: expected {expected[name]}, file has {g.by_name[name].dims}")
    return problems
