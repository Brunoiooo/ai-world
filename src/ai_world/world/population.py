"""The living organisms of a world, stored column-wise for vectorised systems.

Per-organism scalars (pose, vitals, cached physiology traits) live in parallel
numpy arrays so the metabolism / movement systems are array maths rather than a
Python loop over hundreds of objects. Genomes live in an index-aligned list and
their compiled brains in :class:`ai_world.world.brain.BrainStore`, kept in step
row-for-row. :class:`Entity` is only a detached snapshot used at the spawn /
inspect / persistence boundaries.
"""
from __future__ import annotations

import numpy as np

from ai_world.world.brain import BrainStore
from ai_world.world.entity import Entity
from ai_world.world.genome import PHYS_FIELDS, Genome, physiology_vector
from ai_world.world.params import EcoParams

_INDEX_CELL = 8       # tiles per spatial-hash bucket
_KEY_STRIDE = 1 << 20  # packs (gx, gy) into one int; > any realistic grid / _INDEX_CELL
TRAIT_IX = {name: i for i, name in enumerate(PHYS_FIELDS)}

_SCALAR_COLUMNS = (
    "id", "x", "y", "heading", "energy", "hp", "speed", "age",
    "birth_tick", "generation", "species_id", "parent_a", "parent_b",
)
_COLUMN_DTYPE = {
    "id": np.int64, "age": np.int64, "birth_tick": np.int64, "generation": np.int64,
    "species_id": np.int64, "parent_a": np.int64, "parent_b": np.int64,
}


class Population:
    def __init__(self, entities: list[Entity] | None = None, params: EcoParams | None = None):
        self.births = 0
        self.deaths = 0
        self._next_id = 1
        self.genomes: list[Genome] = []
        self.brains = BrainStore(params) if params is not None else None

        for name in _SCALAR_COLUMNS:
            setattr(self, name, np.zeros(0, dtype=_COLUMN_DTYPE.get(name, np.float64)))
        self.traits = np.zeros((0, len(PHYS_FIELDS)), dtype=np.float64)
        self._signature = np.zeros((0, 0), dtype=np.float32)
        self.mating_type = np.zeros((0, 0), dtype=np.float32)
        self.last_turn = np.zeros(0, dtype=np.float64)  # proprioception feedback
        self.repro_cd = np.zeros(0, dtype=np.int64)     # ticks until this organism can mate again
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

    def __len__(self) -> int:
        return int(self.id.shape[0])

    def new_id(self) -> int:
        value = self._next_id
        self._next_id += 1
        return value

    @property
    def signature(self) -> np.ndarray:
        return self._signature

    # --- growing / shrinking the store ---------------------------
    def add_many(self, entities: list[Entity]) -> None:
        if not entities:
            return
        for name in _SCALAR_COLUMNS:
            values = [_column_value(e, name) for e in entities]
            setattr(self, name, np.append(getattr(self, name), values))

        trait_rows = np.array(
            [physiology_vector(e.genome.physiology) for e in entities], dtype=np.float64
        )
        sig_rows = np.array([e.genome.body_signature for e in entities], dtype=np.float32)
        mt_rows = np.array([e.genome.mating_type for e in entities], dtype=np.float32)
        self.traits = np.vstack([self.traits, trait_rows]) if self.traits.size else trait_rows
        self._signature = (
            np.vstack([self._signature, sig_rows]) if self._signature.size else sig_rows
        )
        self.mating_type = (
            np.vstack([self.mating_type, mt_rows]) if self.mating_type.size else mt_rows
        )
        self.last_turn = np.append(self.last_turn, np.zeros(len(entities)))
        self.repro_cd = np.append(self.repro_cd, np.zeros(len(entities), dtype=np.int64))
        self.genomes.extend(e.genome for e in entities)
        if self.brains is not None:
            self.brains.append([e.genome for e in entities])
        if len(self):
            self._next_id = max(self._next_id, int(self.id.max()) + 1)

    def keep(self, mask: np.ndarray) -> None:
        """Retain only rows where ``mask`` is true (death), by swap-removal so
        the brain store never has to copy its whole weight tensor."""
        n = len(self)
        n_new = int(mask.sum())
        if n_new == n:
            return
        self.deaths += n - n_new
        dead = np.flatnonzero(~mask)
        live = np.flatnonzero(mask)
        fill = dead[dead < n_new]
        src = live[live >= n_new]

        for name in _SCALAR_COLUMNS:
            arr = getattr(self, name)
            arr[fill] = arr[src]
            setattr(self, name, arr[:n_new])
        for matrix_name in ("traits", "_signature", "mating_type"):
            m = getattr(self, matrix_name)
            m[fill] = m[src]
            setattr(self, matrix_name, m[:n_new])
        self.last_turn[fill] = self.last_turn[src]
        self.last_turn = self.last_turn[:n_new]
        self.repro_cd[fill] = self.repro_cd[src]
        self.repro_cd = self.repro_cd[:n_new]
        for dst, source in zip(fill, src):
            self.genomes[dst] = self.genomes[source]
        del self.genomes[n_new:]
        if self.brains is not None:
            self.brains.compact(fill, src, n_new)

    def resync_traits(self, i: int) -> None:
        """Re-cache the trait row + signature for organism ``i`` after a genome edit."""
        self.traits[i] = physiology_vector(self.genomes[i].physiology)
        self._signature[i] = self.genomes[i].body_signature
        self.mating_type[i] = self.genomes[i].mating_type

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
    def rebuild_index(self) -> None:
        n = len(self)
        if n == 0:
            self._sorted_idx = np.zeros(0, dtype=np.intp)
            self._buckets = {}
            return
        gx = self.x.astype(np.int64) // _INDEX_CELL
        gy = self.y.astype(np.int64) // _INDEX_CELL
        key = gx * _KEY_STRIDE + gy
        order = np.argsort(key, kind="stable").astype(np.intp)
        self._sorted_idx = order
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
                cell = self._buckets.get(gx * _KEY_STRIDE + gy)
                if cell is None:
                    continue
                for i in self._sorted_idx[cell[0]:cell[1]]:
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


def _column_value(entity: Entity, name: str):
    if name in ("parent_a", "parent_b"):
        value = getattr(entity, name)
        return -1 if value is None else value
    return getattr(entity, name)


__all__ = ["Population", "TRAIT_IX"]
