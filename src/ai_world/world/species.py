"""Dynamic speciation by NEAT compatibility distance.

Newborns are assigned to the first species whose representative is within
``threshold`` compatibility distance, or found a new species. A periodic pass
recounts members, drops empty species and refreshes representatives. The
threshold nudges itself to keep the number of species near a target so a run
neither collapses to one species nor shatters into hundreds.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ai_world.world.genome import PHYS_FIELDS, Genome, compat_distance

# Per-species mean trait vector recorded alongside each census: the physiology
# fields (row-aligned with ``Population.traits``) plus three brain-complexity
# scalars. Indexes into a ``trait_history`` vector.
TRAIT_CHANNELS: tuple[str, ...] = PHYS_FIELDS + ("brain_nodes", "brain_conns", "ports")

_HISTORY_CAP = 4000


@dataclass
class Species:
    id: int
    representative: Genome
    first_tick: int
    parent_id: int | None = None
    count: int = 0


@dataclass
class SpeciesRegistry:
    threshold: float = 3.0
    target_count: int = 12
    next_id: int = 1
    species: dict[int, Species] = field(default_factory=dict)
    # census: list of (tick, {species_id: count})
    census: list = field(default_factory=list)
    # trait_history: list of (tick, {species_id: np.ndarray(len(TRAIT_CHANNELS))}),
    # index-aligned with ``census`` -- the mean traits of each species over time.
    trait_history: list = field(default_factory=list)

    def assign(self, genome: Genome, tick: int, parent_species: int | None = None) -> int:
        best_id, best_dist = None, self.threshold
        for sp in self.species.values():
            dist = compat_distance(genome, sp.representative)
            if dist < best_dist:
                best_id, best_dist = sp.id, dist
        if best_id is not None:
            return best_id

        sp_id = self.next_id
        self.next_id += 1
        self.species[sp_id] = Species(sp_id, genome.copy(), tick, parent_species)
        return sp_id

    def recount(
        self,
        species_ids: np.ndarray,
        genomes: list[Genome],
        tick: int,
        traits: np.ndarray | None = None,
    ) -> None:
        counts: dict[int, int] = {}
        for sid in species_ids:
            counts[int(sid)] = counts.get(int(sid), 0) + 1

        for sp_id in list(self.species):
            n = counts.get(sp_id, 0)
            self.species[sp_id].count = n
            if n == 0:
                del self.species[sp_id]

        # refresh representatives to a living member
        by_species: dict[int, int] = {}
        for idx, sid in enumerate(species_ids):
            by_species.setdefault(int(sid), idx)
        for sp_id, idx in by_species.items():
            if sp_id in self.species:
                self.species[sp_id].representative = genomes[idx].copy()

        self._adapt_threshold()
        self.census.append((tick, {k: v for k, v in counts.items() if v}))
        if len(self.census) > _HISTORY_CAP:
            self.census = self.census[-_HISTORY_CAP:]

        if traits is not None:
            self.trait_history.append((tick, _trait_means(species_ids, genomes, traits)))
            if len(self.trait_history) > _HISTORY_CAP:
                self.trait_history = self.trait_history[-_HISTORY_CAP:]

    def _adapt_threshold(self) -> None:
        # No population gate: a struggling small population needs speciation
        # (and the niche diversification it drives) to recover, not less of
        # it. The threshold's own clamps (1.0 .. 20.0) keep it from ever
        # shattering into one-member "species" or collapsing to a single blob.
        live = len(self.species)
        if live > self.target_count * 1.3:
            self.threshold = min(20.0, self.threshold * 1.04)
        elif live < self.target_count * 0.7:
            self.threshold = max(1.0, self.threshold * 0.97)

    def sorted_species(self) -> list[Species]:
        return sorted(self.species.values(), key=lambda s: s.count, reverse=True)


def _trait_means(
    species_ids: np.ndarray, genomes: list[Genome], traits: np.ndarray
) -> dict[int, np.ndarray]:
    """Mean trait vector per species: physiology columns from ``traits`` plus
    ``[brain_nodes, brain_conns, ports]`` averaged over that species' genomes."""
    out: dict[int, np.ndarray] = {}
    ids = species_ids.astype(np.int64)
    for sid in np.unique(ids):
        mask = ids == sid
        phys = traits[mask].mean(axis=0)
        brain = np.mean(
            [
                (g.node_count, g.enabled_conn_count, len(g.ports))
                for g, keep in zip(genomes, mask)
                if keep
            ],
            axis=0,
        )
        out[int(sid)] = np.concatenate([phys, brain]).astype(np.float32)
    return out
