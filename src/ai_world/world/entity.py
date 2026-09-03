"""A detached snapshot of one organism.

The live population is stored column-wise in :class:`ai_world.world.population`.
``Entity`` is the record type used only where a single organism crosses a
boundary: spawning, the UI inspector, and persistence rows.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ai_world.world.genome import Genome


@dataclass
class Entity:
    id: int
    x: float
    y: float
    heading: float
    energy: float
    hp: float
    genome: Genome
    age: int = 0
    birth_tick: int = 0
    generation: int = 0
    species_id: int = 0
    parent_a: int | None = None
    parent_b: int | None = None
    speed: float = 0.0
    brain: object | None = None
    brain_state: np.ndarray | None = None

    @property
    def tile_x(self) -> int:
        return int(self.x)

    @property
    def tile_y(self) -> int:
        return int(self.y)
