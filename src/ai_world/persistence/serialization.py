"""Packing world state to/from BLOBs (raw bytes + zlib)."""
from __future__ import annotations

import json
import struct
import zlib
from dataclasses import asdict, fields

import numpy as np

from ai_world.world.food import N_FOOD_TYPES
from ai_world.world.genome import (
    ConnGene,
    Genome,
    Innovations,
    NodeGene,
    Physiology,
    PortGene,
)
from ai_world.world.grid import Grid
from ai_world.world.params import EcoParams
from ai_world.world.weather import WeatherState

_GENOME_VERSION = 5

# Physiology fields added after a genome version bump: fill these in for saves
# from before they existed rather than failing to load. Values reproduce the
# old fixed behaviour (one offspring, the unscaled `EcoParams.repro_cooldown`).
_PHYSIOLOGY_DEFAULTS: dict = {"repro_cooldown_mult": 1.0, "litter_size": 1.0}

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
        "diet": genome.diet.astype(float).tolist(),
        "nodes": [[n.id, n.kind, n.activation] for n in genome.nodes],
        "conns": [[c.innov, c.src, c.dst, c.weight, c.enabled] for c in genome.conns],
        "ports": [
            [p.node_id, p.mode, p.signature.astype(float).tolist(),
             p.angle, p.arc, p.reach, p.gain]
            for p in genome.ports
        ],
    }
    return json.dumps(doc, separators=(",", ":")).encode("utf-8")


def deserialize_genome(blob: bytes) -> Genome:
    doc = json.loads(blob.decode("utf-8"))
    physiology = {**_PHYSIOLOGY_DEFAULTS, **doc["physiology"]}
    return Genome(
        physiology=Physiology(**physiology),
        body_signature=np.asarray(doc["body_signature"], dtype=np.float32),
        mating_type=np.asarray(doc["mating_type"], dtype=np.float32),
        diet=np.asarray(
            doc.get("diet", np.zeros(N_FOOD_TYPES)), dtype=np.float32
        ),
        nodes=[NodeGene(i, k, a) for i, k, a in doc.get("nodes", [])],
        conns=[
            ConnGene(innov, src, dst, w, bool(en))
            for innov, src, dst, w, en in doc.get("conns", [])
        ],
        ports=[
            PortGene(nid, mode, np.asarray(sig, dtype=np.float32), ang, arc, reach, gain)
            for nid, mode, sig, ang, arc, reach, gain in doc.get("ports", [])
        ],
    )


def _pack_innovations(innov: Innovations) -> dict:
    return {
        "conn": [[a, b, i] for (a, b), i in innov.conn.items()],
        "node_split": [[a, b, i] for (a, b), i in innov.node_split.items()],
        "next_conn": innov.next_conn,
        "next_node": innov.next_node,
    }


def _unpack_innovations(doc: dict) -> Innovations:
    return Innovations(
        conn={(a, b): i for a, b, i in doc["conn"]},
        node_split={(a, b): i for a, b, i in doc["node_split"]},
        next_conn=doc["next_conn"],
        next_node=doc["next_node"],
    )


def serialize_eco_state(
    params: EcoParams,
    weather: WeatherState,
    rng: np.random.Generator,
    innovations: Innovations,
    species_meta: dict,
) -> bytes:
    doc = {
        "version": _ECO_STATE_VERSION,
        "params": asdict(params),
        "weather": asdict(weather),
        "rng": rng.bit_generator.state,
        "innovations": _pack_innovations(innovations),
        "species": species_meta,
    }
    return zlib.compress(json.dumps(doc).encode("utf-8"), level=6)


def deserialize_eco_state(
    blob: bytes,
) -> tuple[EcoParams, WeatherState, np.random.Generator, Innovations, dict]:
    doc = json.loads(zlib.decompress(blob).decode("utf-8"))
    params_doc = dict(doc["params"])
    params_doc["climate_event_ticks"] = tuple(params_doc["climate_event_ticks"])
    known = {f.name for f in fields(EcoParams)}
    params = EcoParams(**{k: v for k, v in params_doc.items() if k in known})
    weather = WeatherState(**doc["weather"])
    rng = np.random.default_rng()
    rng.bit_generator.state = doc["rng"]
    innovations = _unpack_innovations(doc["innovations"])
    species_meta = doc.get("species", {"threshold": 3.0, "target": 12, "next_id": 1})
    return params, weather, rng, innovations, species_meta
