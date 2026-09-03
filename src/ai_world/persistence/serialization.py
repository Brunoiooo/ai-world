"""Packing the tile grid to/from a BLOB (raw ``uint8`` bytes + zlib)."""
from __future__ import annotations

import zlib

import numpy as np

from ai_world.world.grid import Grid

_MAGIC = b"AWG1"  # ai-world grid, version 1


def serialize_grid(grid: Grid) -> bytes:
    payload = zlib.compress(grid.cells.tobytes(order="C"), level=6)
    return _MAGIC + payload


def deserialize_grid(blob: bytes, width: int, height: int) -> Grid:
    if blob[:4] != _MAGIC:
        raise ValueError("unknown grid format")
    raw = zlib.decompress(blob[4:])
    expected = width * height
    cells = np.frombuffer(raw, dtype=np.uint8)
    if cells.size != expected:
        raise ValueError(f"grid size {cells.size} != {expected} ({width}x{height})")
    return Grid(cells.reshape((height, width)).copy())
