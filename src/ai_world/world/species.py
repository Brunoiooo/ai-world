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

from ai_world.world.genome import Genome, compat_distance


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

    def recount(self, species_ids: np.ndarray, genomes: list[Genome], tick: int) -> None:
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

        self._adapt_threshold(int(species_ids.size))
        self.census.append((tick, {k: v for k, v in counts.items() if v}))
        if len(self.census) > 4000:
            self.census = self.census[-4000:]

    def _adapt_threshold(self, population: int) -> None:
        # only nudge when the population is large enough to actually support
        # ``target_count`` species; otherwise a small population would shatter.
        if population < self.target_count * 20:
            return
        live = len(self.species)
        if live > self.target_count * 1.3:
            self.threshold = min(20.0, self.threshold * 1.04)
        elif live < self.target_count * 0.7:
            self.threshold = max(1.0, self.threshold * 0.97)

    def sorted_species(self) -> list[Species]:
        return sorted(self.species.values(), key=lambda s: s.count, reverse=True)
