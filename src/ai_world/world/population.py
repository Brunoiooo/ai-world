"""The living organisms of a world, stored column-wise for vectorised systems.

Per-organism scalars (pose, vitals, cached physiology traits) live in parallel
numpy arrays so the metabolism / movement systems are array maths rather than a
Python loop over hundreds of objects. Object-shaped data (the genome, and later
the torch brain) lives in index-aligned lists. :class:`Entity` is only a
detached snapshot used at the spawn / inspect / persistence boundaries.
"""
from __future__ import annotations

import numpy as np

from ai_world.world.entity import Entity
from ai_world.world.genome import (
    PHYS_FIELDS,
    Genome,
    physiology_from_vector,
    physiology_vector,
)

_INDEX_CELL = 8       # tiles per spatial-hash bucket
_KEY_STRIDE = 1 << 20  # packs (gx, gy) into one int; > any realistic grid/_INDEX_CELL
TRAIT_IX = {name: i for i, name in enumerate(PHYS_FIELDS)}


class Population:
    def __init__(self, entities: list[Entity] | None = None):
        self.births = 0
        self.deaths = 0
        self._next_id = 1
        self.genomes: list[Genome] = []
        self.brains: list[object | None] = []
        self.brain_state: list[np.ndarray | None] = []

        self.id = np.zeros(0, dtype=np.int64)
        self.x = np.zeros(0, dtype=np.float64)
        self.y = np.zeros(0, dtype=np.float64)
        self.heading = np.zeros(0, dtype=np.float64)
        self.energy = np.zeros(0, dtype=np.float64)
        self.hp = np.zeros(0, dtype=np.float64)
        self.speed = np.zeros(0, dtype=np.float64)
        self.age = np.zeros(0, dtype=np.int64)
        self.birth_tick = np.zeros(0, dtype=np.int64)
        self.generation = np.zeros(0, dtype=np.int64)
        self.species_id = np.zeros(0, dtype=np.int64)
        self.parent_a = np.zeros(0, dtype=np.int64)
        self.parent_b = np.zeros(0, dtype=np.int64)
        self.traits = np.zeros((0, len(PHYS_FIELDS)), dtype=np.float64)
        self._signature = np.zeros((0, 0), dtype=np.float32)
        self._sorted_idx = np.zeros(0, dtype=np.intp)
        self._buckets: dict[int, tuple[int, int]] = {}

        # transient per-tick decisions, (re)written wholesale by the think system
        self.i_turn = np.zeros(0, dtype=np.float64)
        self.i_thrust = np.zeros(0, dtype=np.float64)
        self.i_eat = np.zeros(0, dtype=bool)
        self.i_attack = np.zeros(0, dtype=bool)
        self.i_mate = np.zeros(0, dtype=bool)

        if entities:
            self.add_many(entities)

    # --- size ---------------------------------------------------------
    def __len__(self) -> int:
        return int(self.id.shape[0])

    def new_id(self) -> int:
        value = self._next_id
        self._next_id += 1
        return value

    @property
    def signature(self) -> np.ndarray:
        return self._signature

    # --- mutation of the store -------------------------------------
    def _entity_columns(self, entity: Entity) -> dict:
        return {
            "id": entity.id,
            "x": entity.x,
            "y": entity.y,
            "heading": entity.heading,
            "energy": entity.energy,
            "hp": entity.hp,
            "speed": entity.speed,
            "age": entity.age,
            "birth_tick": entity.birth_tick,
            "generation": entity.generation,
            "species_id": entity.species_id,
            "parent_a": -1 if entity.parent_a is None else entity.parent_a,
            "parent_b": -1 if entity.parent_b is None else entity.parent_b,
        }

    def add_many(self, entities: list[Entity]) -> None:
        if not entities:
            return
        cols = [self._entity_columns(e) for e in entities]
        self.id = np.append(self.id, [c["id"] for c in cols])
        self.x = np.append(self.x, [c["x"] for c in cols])
        self.y = np.append(self.y, [c["y"] for c in cols])
        self.heading = np.append(self.heading, [c["heading"] for c in cols])
        self.energy = np.append(self.energy, [c["energy"] for c in cols])
        self.hp = np.append(self.hp, [c["hp"] for c in cols])
        self.speed = np.append(self.speed, [c["speed"] for c in cols])
        self.age = np.append(self.age, [c["age"] for c in cols])
        self.birth_tick = np.append(self.birth_tick, [c["birth_tick"] for c in cols])
        self.generation = np.append(self.generation, [c["generation"] for c in cols])
        self.species_id = np.append(self.species_id, [c["species_id"] for c in cols])
        self.parent_a = np.append(self.parent_a, [c["parent_a"] for c in cols])
        self.parent_b = np.append(self.parent_b, [c["parent_b"] for c in cols])

        trait_rows = np.array(
            [physiology_vector(e.genome.physiology) for e in entities], dtype=np.float64
        )
        self.traits = np.vstack([self.traits, trait_rows]) if len(self.traits) else trait_rows
        sig_rows = np.array([e.genome.body_signature for e in entities], dtype=np.float32)
        self._signature = (
            np.vstack([self._signature, sig_rows]) if self._signature.size else sig_rows
        )
        self.genomes.extend(e.genome for e in entities)
        self.brains.extend(e.brain for e in entities)
        self.brain_state.extend(e.brain_state for e in entities)
        self._next_id = max(self._next_id, int(self.id.max()) + 1) if len(self) else self._next_id

    def keep(self, mask: np.ndarray) -> None:
        """Retain only rows where ``mask`` is true (used by death)."""
        removed = int((~mask).sum())
        if removed == 0:
            return
        self.deaths += removed
        for name in (
            "id", "x", "y", "heading", "energy", "hp", "speed", "age",
            "birth_tick", "generation", "species_id", "parent_a", "parent_b",
        ):
            setattr(self, name, getattr(self, name)[mask])
        self.traits = self.traits[mask]
        self._signature = self._signature[mask]
        keep_idx = np.flatnonzero(mask)
        self.genomes = [self.genomes[i] for i in keep_idx]
        self.brains = [self.brains[i] for i in keep_idx]
        self.brain_state = [self.brain_state[i] for i in keep_idx]

    # --- snapshots (boundary use only) -----------------------------
    def snapshot(self, i: int) -> Entity:
        pa, pb = int(self.parent_a[i]), int(self.parent_b[i])
        return Entity(
            id=int(self.id[i]),
            x=float(self.x[i]),
            y=float(self.y[i]),
            heading=float(self.heading[i]),
            energy=float(self.energy[i]),
            hp=float(self.hp[i]),
            genome=self.genomes[i],
            age=int(self.age[i]),
            birth_tick=int(self.birth_tick[i]),
            generation=int(self.generation[i]),
            species_id=int(self.species_id[i]),
            parent_a=None if pa < 0 else pa,
            parent_b=None if pb < 0 else pb,
            speed=float(self.speed[i]),
        )

    def snapshots(self) -> list[Entity]:
        return [self.snapshot(i) for i in range(len(self))]

    def index_of(self, entity_id: int) -> int | None:
        hit = np.flatnonzero(self.id == entity_id)
        return int(hit[0]) if hit.size else None

    # --- spatial index -------------------------------------------------
    # A uniform grid hash kept as sorted arrays: ``_sorted_idx`` holds entity
    # rows grouped by bucket, ``_buckets`` maps a packed cell key to the
    # [start, end) slice of that group.
    def rebuild_index(self) -> None:
        n = len(self)
        if n == 0:
            self._sorted_idx = np.zeros(0, dtype=np.intp)
            self._buckets = {}
            return
        gx = self.x.astype(np.int64) // _INDEX_CELL
        gy = self.y.astype(np.int64) // _INDEX_CELL
        key = gx * _KEY_STRIDE + gy
        order = np.argsort(key, kind="stable")
        self._sorted_idx = order.astype(np.intp)
        sorted_key = key[order]
        uniq, starts = np.unique(sorted_key, return_index=True)
        ends = np.append(starts[1:], n)
        self._buckets = {int(k): (int(s), int(e)) for k, s, e in zip(uniq, starts, ends)}

    def neighbours(self, x: float, y: float, radius: float) -> list[int]:
        cx, cy = int(x) // _INDEX_CELL, int(y) // _INDEX_CELL
        span = int(radius // _INDEX_CELL) + 1
        r2 = radius * radius
        out: list[int] = []
        for gx in range(cx - span, cx + span + 1):
            for gy in range(cy - span, cy + span + 1):
                span_slice = self._buckets.get(gx * _KEY_STRIDE + gy)
                if span_slice is None:
                    continue
                for i in self._sorted_idx[span_slice[0]:span_slice[1]]:
                    if (self.x[i] - x) ** 2 + (self.y[i] - y) ** 2 <= r2:
                        out.append(int(i))
        return out

    def nearest_index(
        self, x: float, y: float, max_dist: float, *, exclude: int | None = None
    ) -> int | None:
        best, best_d2 = None, max_dist * max_dist
        for i in self.neighbours(x, y, max_dist):
            if i == exclude:
                continue
            d2 = (self.x[i] - x) ** 2 + (self.y[i] - y) ** 2
            if d2 <= best_d2:
                best, best_d2 = i, d2
        return best

    def nearest(
        self, x: float, y: float, max_dist: float, *, exclude_id: int | None = None
    ) -> Entity | None:
        exclude = None if exclude_id is None else self.index_of(exclude_id)
        i = self.nearest_index(x, y, max_dist, exclude=exclude)
        return None if i is None else self.snapshot(i)


def rebuild_traits(pop: Population, i: int) -> None:
    """Re-cache the trait row + signature for organism ``i`` after a genome change."""
    genome = pop.genomes[i]
    pop.traits[i] = physiology_vector(genome.physiology)
    pop._signature[i] = genome.body_signature


__all__ = ["Population", "TRAIT_IX", "rebuild_traits", "physiology_from_vector"]
