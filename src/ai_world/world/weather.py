"""Global + local climate: season phase, drifting weather, moving fronts and
rare climate events.

The season and the slow drift are deterministic functions of the tick counter.
The moving fronts are a coarse value-noise field scrolled by a wind vector, so a
reloaded world with the same tick reproduces the same weather without storing
any large arrays. Climate events (cold snaps, heat waves, blooms, droughts) are
drawn from the world RNG and carried in the state.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ai_world.world.params import EcoParams

_FRONT_CELLS = 6          # coarse-noise lattice resolution
_WIND = (0.013, 0.008)    # tiles/tick the front field scrolls


@dataclass
class WeatherState:
    season_phase: float = 0.0
    temperature_anomaly: float = 0.0     # global, added everywhere
    regen_multiplier: float = 1.0        # global enzyme-regrowth scale
    event: str = ""                      # "", "cold_snap", "heat_wave", "bloom", "drought"
    event_ticks: int = 0

    def advance(self, tick: int, params: EcoParams, rng=None) -> None:
        self.season_phase = (tick / params.season_period_ticks) % 1.0
        self.temperature_anomaly = (
            0.10 * math.sin(tick / 1737.0) + 0.05 * math.sin(tick / 613.0)
        )
        drought = 0.5 * math.sin(tick / 2903.0) - 0.2
        self.regen_multiplier = max(0.3, 1.0 - max(0.0, drought))

        if self.event_ticks > 0:
            self.event_ticks -= 1
            if self.event_ticks == 0:
                self.event = ""
        elif rng is not None and rng.random() < params.climate_event_chance:
            self.event = str(rng.choice(("cold_snap", "heat_wave", "bloom", "drought")))
            self.event_ticks = int(rng.integers(*params.climate_event_ticks))

        self.temperature_anomaly += {
            "cold_snap": -0.22, "heat_wave": 0.22,
        }.get(self.event, 0.0)
        self.regen_multiplier *= {
            "bloom": 1.8, "drought": 0.35,
        }.get(self.event, 1.0)

    @property
    def season_offset(self) -> float:
        return math.sin(2.0 * math.pi * self.season_phase)

    def season_offset_scaled(self, amplitude: float) -> float:
        return amplitude * self.season_offset

    # --- local fields ------------------------------------------------
    def front_field(self, height: int, width: int, seed: int, tick: int) -> np.ndarray:
        """A smooth (h, w) field in roughly [-1, 1]: drifting warm/cold fronts.

        A coarse value-noise lattice (regenerated deterministically from the
        seed) sampled through a window that the wind vector scrolls across it.
        The lattice is periodic, so the window roams without bound and the field
        keeps drifting for the life of the world.
        """
        rng = np.random.default_rng((seed ^ 0x5A17) & 0xFFFFFFFF)
        lattice = rng.standard_normal((_FRONT_CELLS, _FRONT_CELLS)).astype(np.float32)

        ys = np.linspace(0.0, _FRONT_CELLS, height, endpoint=False) + _WIND[1] * tick
        xs = np.linspace(0.0, _FRONT_CELLS, width, endpoint=False) + _WIND[0] * tick
        y0 = np.floor(ys).astype(np.intp)
        x0 = np.floor(xs).astype(np.intp)
        fy = (ys - y0)[:, None]
        fx = (xs - x0)[None, :]
        fy = fy * fy * (3.0 - 2.0 * fy)
        fx = fx * fx * (3.0 - 2.0 * fx)
        y0m, y1m = y0 % _FRONT_CELLS, (y0 + 1) % _FRONT_CELLS
        x0m, x1m = x0 % _FRONT_CELLS, (x0 + 1) % _FRONT_CELLS
        top = lattice[np.ix_(y0m, x0m)] * (1 - fx) + lattice[np.ix_(y0m, x1m)] * fx
        bot = lattice[np.ix_(y1m, x0m)] * (1 - fx) + lattice[np.ix_(y1m, x1m)] * fx
        return np.tanh(top * (1 - fy) + bot * fy).astype(np.float32)
