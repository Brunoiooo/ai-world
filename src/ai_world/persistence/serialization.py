"""Packing world state to/from BLOBs (raw bytes + zlib)."""
from __future__ import annotations

import json
import struct
import zlib
from dataclasses import asdict

import numpy as np

from ai_world.world.genome import Genome, Physiology
from ai_world.world.grid import Grid
from ai_world.world.params import EcoParams
from ai_world.world.weather import WeatherState

_GENOME_VERSION = 2

_GRID_MAGIC = b"AWG1"   # ai-world grid, version 1
_ARRAY_MAGIC = b"AWA1"   # self-describing float32 array, version 1
_ECO_STATE_VERSION = 1


def serialize_grid(grid: Grid) -> bytes:
    payload = zlib.compress(grid.cells.tobytes(order="C"), level=6)
    return _GRID_MAGIC + payload


def deserialize_grid(blob: bytes, width: int, height: int) -> Grid:
    if blob[:4] != _GRID_MAGIC:
        raise ValueError("unknown grid format")
    raw = zlib.decompress(blob[4:])
    expected = width * height
    cells = np.frombuffer(raw, dtype=np.uint8)
    if cells.size != expected:
        raise ValueError(f"grid size {cells.size} != {expected} ({width}x{height})")
    return Grid(cells.reshape((height, width)).copy())


def serialize_array(array: np.ndarray) -> bytes:
    """Self-describing ``float32`` array blob: magic + shape header + zlib payload."""
    a = np.ascontiguousarray(array, dtype=np.float32)
    header = struct.pack("<B", a.ndim) + b"".join(struct.pack("<I", d) for d in a.shape)
    return _ARRAY_MAGIC + header + zlib.compress(a.tobytes(order="C"), level=6)


def deserialize_array(blob: bytes) -> np.ndarray:
    if blob[:4] != _ARRAY_MAGIC:
        raise ValueError("unknown array format")
    ndim = struct.unpack_from("<B", blob, 4)[0]
    offset = 5
    shape = tuple(
        struct.unpack_from("<I", blob, offset + 4 * i)[0] for i in range(ndim)
    )
    offset += 4 * ndim
    raw = zlib.decompress(blob[offset:])
    return np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()


def serialize_genome(genome: Genome) -> bytes:
    doc = {
        "v": _GENOME_VERSION,
        "physiology": asdict(genome.physiology),
        "body_signature": genome.body_signature.astype(float).tolist(),
        "mating_type": genome.mating_type.astype(float).tolist(),
    }
    return json.dumps(doc, separators=(",", ":")).encode("utf-8")


def deserialize_genome(blob: bytes) -> Genome:
    doc = json.loads(blob.decode("utf-8"))
    return Genome(
        physiology=Physiology(**doc["physiology"]),
        body_signature=np.asarray(doc["body_signature"], dtype=np.float32),
        mating_type=np.asarray(doc["mating_type"], dtype=np.float32),
    )


def serialize_eco_state(
    params: EcoParams, weather: WeatherState, rng: np.random.Generator
) -> bytes:
    doc = {
        "version": _ECO_STATE_VERSION,
        "params": asdict(params),
        "weather": asdict(weather),
        "rng": rng.bit_generator.state,
    }
    return zlib.compress(json.dumps(doc).encode("utf-8"), level=6)


def deserialize_eco_state(
    blob: bytes,
) -> tuple[EcoParams, WeatherState, np.random.Generator]:
    doc = json.loads(zlib.decompress(blob).decode("utf-8"))
    params = EcoParams(**doc["params"])
    weather = WeatherState(**doc["weather"])
    rng = np.random.default_rng()
    rng.bit_generator.state = doc["rng"]
    return params, weather, rng
