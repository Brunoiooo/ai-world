"""Global climate state: season phase plus slow weather drift.

Deterministic function of the tick counter — no RNG — so a reloaded world with
the same tick reproduces the same climate. Phase 5 layers moving fronts and
climate events on top of this.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ai_world.world.params import EcoParams


@dataclass
class WeatherState:
    season_phase: float = 0.0        # 0..1 around the year
    temperature_anomaly: float = 0.0  # normalized units added everywhere
    regen_multiplier: float = 1.0     # scales enzyme regrowth (drought < 1)

    def advance(self, tick: int, params: EcoParams) -> None:
        self.season_phase = (tick / params.season_period_ticks) % 1.0
        self.temperature_anomaly = (
            0.12 * math.sin(tick / 1737.0) + 0.06 * math.sin(tick / 613.0)
        )
        drought = 0.55 * math.sin(tick / 2903.0) - 0.15
        self.regen_multiplier = max(0.2, 1.0 - max(0.0, drought))

    @property
    def season_offset(self) -> float:
        """Normalized temperature shift from the season alone."""
        return math.sin(2.0 * math.pi * self.season_phase)

    def season_offset_scaled(self, amplitude: float) -> float:
        return amplitude * self.season_offset
