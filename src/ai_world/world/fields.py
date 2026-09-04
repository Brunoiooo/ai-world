"""Continuous scalar/vector fields laid over the tile grid.

Three fields make up the physical substrate the ecosystem runs on:

* :class:`FoodField`     -- multi-channel food (see :mod:`ai_world.world.food`);
  biome-grown types spread on their own terrain, derived types are deposited by
  the organisms themselves.
* :class:`TemperatureField` -- per-tile temperature (terrain + latitude + altitude),
  modulated by season and weather each tick.
* :class:`SpectrumField`  -- ``C`` decaying/diffusing channels that carry the
  "signals" organisms sense and emit.

Every field is a plain ``float32`` numpy array in ``[y, x]`` layout, matching
:class:`ai_world.world.grid.Grid`. The classes hold only state and local update
rules; the schedule (which field steps on which tick) lives in the simulation
systems.
"""
from __future__ import annotations

import numpy as np

from ai_world.world.food import FOOD_TYPES, N_FOOD_TYPES
from ai_world.world.generator import heightmap
from ai_world.world.grid import Grid
from ai_world.world.params import EcoParams
from ai_world.world.tiles import Tile

# Static temperature bias per terrain type, in normalized units (0.5 == temperate).
TERRAIN_TEMPERATURE: dict[int, float] = {
    Tile.DEEP_WATER: -0.02,
    Tile.WATER: -0.01,
    Tile.SAND: 0.12,
    Tile.GRASS: 0.0,
    Tile.FOREST: -0.03,
    Tile.DIRT: 0.05,
    Tile.ROCK: -0.02,
    Tile.SNOW: -0.18,
}


def _terrain_temperature_map(grid: Grid) -> np.ndarray:
    out = np.zeros(grid.cells.shape, dtype=np.float32)
    for tile, value in TERRAIN_TEMPERATURE.items():
        out[grid.cells == int(tile)] = value
    return out


def _diffuse(a: np.ndarray, rate: float) -> None:
    """In-place isotropic diffusion of a 2D plane (clamped, non-wrapping boundary).

    Kept per-plane and pad-free on purpose: ``np.pad`` and 3D batching both
    measured slower here (allocation + cache spill).
    """
    if rate <= 0.0:
        return
    lap = np.full_like(a, 0.0)
    lap[1:, :] += a[:-1, :]
    lap[:-1, :] += a[1:, :]
    lap[:, 1:] += a[:, :-1]
    lap[:, :-1] += a[:, 1:]
    # clamped boundary: the off-grid neighbour equals the edge cell itself
    lap[0, :] += a[0, :]
    lap[-1, :] += a[-1, :]
    lap[:, 0] += a[:, 0]
    lap[:, -1] += a[:, -1]
    lap -= 4.0 * a
    a += rate * lap


def _bilinear(a: np.ndarray, x: float, y: float) -> np.ndarray:
    """Sample ``a`` (``[..., h, w]``) at fractional ``(x, y)``, clamped to bounds."""
    h, w = a.shape[-2:]
    x = min(max(x, 0.0), w - 1.001)
    y = min(max(y, 0.0), h - 1.001)
    x0, y0 = int(x), int(y)
    fx, fy = x - x0, y - y0
    top = a[..., y0, x0] * (1.0 - fx) + a[..., y0, x0 + 1] * fx
    bot = a[..., y0 + 1, x0] * (1.0 - fx) + a[..., y0 + 1, x0 + 1] * fx
    return top * (1.0 - fy) + bot * fy


def _grow_ceilings(grid: Grid, capacity: float) -> np.ndarray:
    """``(K, h, w)`` per-channel carrying capacity. A grown type has capacity only
    on its own terrain; derived types (enzyme / carrion) never grow, so 0."""
    ceilings = np.zeros((N_FOOD_TYPES, *grid.cells.shape), dtype=np.float32)
    for k, food in enumerate(FOOD_TYPES):
        if food.source == "grow" and food.terrain is not None:
            ceilings[k][grid.cells == food.terrain] = capacity
    return ceilings


class FoodField:
    """Multi-channel food. ``values[k, y, x]`` in ``[0, capacity]``; one plane per
    :data:`ai_world.world.food.FOOD_TYPES` entry.

    Grown types expand logistically over their own terrain and diffuse into the
    neighbourhood; enzyme / carrion planes only decay and diffuse from whatever
    the organisms deposit.
    """

    __slots__ = (
        "values", "_ceiling", "_grow", "_growth_chance", "_seed", "_decay",
        "_diffusion", "_capacity",
    )

    def __init__(
        self,
        values: np.ndarray,
        ceiling: np.ndarray,
        *,
        growth_rate: float,
        growth_chance: float = 1.0,
        seed_rate: float,
        decay: np.ndarray,
        diffusion: float,
        capacity: float,
    ):
        self.values = np.ascontiguousarray(values, dtype=np.float32)
        self._ceiling = np.ascontiguousarray(ceiling, dtype=np.float32)
        self._grow = growth_rate
        self._growth_chance = growth_chance
        self._seed = seed_rate
        self._decay = np.asarray(decay, dtype=np.float32)[:, None, None]
        self._diffusion = diffusion
        self._capacity = capacity

    @classmethod
    def for_grid(cls, grid: Grid, params: EcoParams, rng: np.random.Generator) -> "FoodField":
        ceiling = _grow_ceilings(grid, params.food_capacity)
        noise = rng.random(ceiling.shape, dtype=np.float32)
        values = ceiling * params.food_initial_fill * noise
        return cls._with_params(values, ceiling, params)

    @classmethod
    def rebuild(cls, grid: Grid, params: EcoParams, values: np.ndarray) -> "FoodField":
        """Reconstruct from persisted ``values``; the rest is derived from the grid."""
        ceiling = _grow_ceilings(grid, params.food_capacity)
        if values.shape != ceiling.shape:  # pre-release: shape changed, start fresh
            values = ceiling * params.food_initial_fill
        return cls._with_params(values, ceiling, params)

    @classmethod
    def _with_params(
        cls, values: np.ndarray, ceiling: np.ndarray, params: EcoParams
    ) -> "FoodField":
        decay = np.array(
            [
                params.food_derived_decay if f.source != "grow" else params.food_decay
                for f in FOOD_TYPES
            ],
            dtype=np.float32,
        )
        return cls(
            values,
            ceiling,
            growth_rate=params.food_growth_rate,
            growth_chance=params.food_growth_chance,
            seed_rate=params.food_seed_rate,
            decay=decay,
            diffusion=params.food_diffusion,
            capacity=params.food_capacity,
        )

    def step(
        self, regen_multiplier: float = 1.0, rng: np.random.Generator | None = None
    ) -> None:
        headroom = self._ceiling - self.values
        logistic = self._grow * self.values * np.maximum(headroom, 0.0) / max(self._capacity, 1e-6)
        seed = self._seed * (self._ceiling > 0.0)
        growth = logistic + seed
        if rng is not None and self._growth_chance < 1.0:
            # Growth lands on a tile only with some probability each tick, scaled
            # up by 1/chance when it does -- same long-run average as applying it
            # every tick, but patchy/bursty tick-to-tick instead of a smooth ramp.
            lucky = rng.random(growth.shape, dtype=np.float32) < self._growth_chance
            growth = np.where(lucky, growth / self._growth_chance, 0.0)
        self.values += regen_multiplier * growth
        self.values -= self._decay * self.values
        for plane in self.values:
            _diffuse(plane, self._diffusion)
        np.clip(self.values, 0.0, self._capacity, out=self.values)

    def total(self) -> np.ndarray:
        """``(h, w)`` summed food density across every channel."""
        return self.values.sum(axis=0)

    def absorb(self, channel: int, x: int, y: int, amount: float) -> float:
        """Remove up to ``amount`` of one food type from a tile; return what was taken."""
        available = float(self.values[channel, y, x])
        taken = min(available, amount)
        self.values[channel, y, x] = available - taken
        return taken

    def deposit(self, channel: int, x: int, y: int, amount: float) -> None:
        self.values[channel, y, x] = min(
            self.values[channel, y, x] + amount, self._capacity
        )


class TemperatureField:
    """Per-tile temperature in normalized units (0.5 == temperate, 0 freezing, 1 scorching)."""

    __slots__ = ("values", "_base", "_latitude_swing")

    def __init__(self, base: np.ndarray, latitude_swing: np.ndarray):
        self._base = np.ascontiguousarray(base, dtype=np.float32)
        self._latitude_swing = np.ascontiguousarray(latitude_swing, dtype=np.float32)
        self.values = self._base.copy()

    @classmethod
    def for_grid(cls, grid: Grid, params: EcoParams, seed: int) -> "TemperatureField":
        h, w = grid.cells.shape
        # Warmest across the vertical middle, coldest at the poles.
        lat = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
        equator_warmth = (1.0 - np.abs(lat)) * np.ones((1, w), dtype=np.float32)

        altitude = heightmap(w, h, seed)
        sea_level = 0.42  # matches the WATER/SAND threshold band in the generator
        land_height = np.clip((altitude - sea_level) / (1.0 - sea_level), 0.0, 1.0)

        base = (
            0.30
            + 0.45 * equator_warmth
            - params.altitude_lapse * land_height
            + _terrain_temperature_map(grid)
        ).astype(np.float32)
        np.clip(base, 0.0, 1.0, out=base)

        # Seasons swing harder toward the poles than at the equator.
        latitude_swing = (0.35 + 0.65 * np.abs(lat)).astype(np.float32) * np.ones(
            (1, w), dtype=np.float32
        )
        return cls(base, latitude_swing)

    def refresh(
        self, season_offset: float, anomaly: float, fronts: np.ndarray | None = None
    ) -> None:
        """Recompute ``values`` from the static base plus the current climate."""
        self.values = self._base + season_offset * self._latitude_swing + anomaly
        if fronts is not None:
            self.values = self.values + fronts
        np.clip(self.values, 0.0, 1.0, out=self.values)

    def at(self, x: float, y: float) -> float:
        return float(_bilinear(self.values, x, y))

    @staticmethod
    def to_celsius(normalized: float) -> float:
        return -20.0 + 60.0 * normalized


class SpectrumField:
    """``C`` decaying, diffusing channels: ``values[c, y, x]``."""

    __slots__ = ("values", "_decay", "_diffusion")

    def __init__(self, values: np.ndarray, *, decay: float, diffusion: float):
        self.values = np.ascontiguousarray(values, dtype=np.float32)
        self._decay = decay
        self._diffusion = diffusion

    @classmethod
    def for_grid(cls, grid: Grid, params: EcoParams) -> "SpectrumField":
        h, w = grid.cells.shape
        values = np.zeros((params.spectrum_channels, h, w), dtype=np.float32)
        return cls.rebuild(params, values)

    @classmethod
    def rebuild(cls, params: EcoParams, values: np.ndarray) -> "SpectrumField":
        return cls(values, decay=params.spectrum_decay, diffusion=params.spectrum_diffusion)

    @property
    def channels(self) -> int:
        return self.values.shape[0]

    def step(self) -> None:
        self.values *= 1.0 - self._decay
        for channel in self.values:
            _diffuse(channel, self._diffusion)

    def set_channel(self, channel: int, plane: np.ndarray) -> None:
        self.values[channel] = plane

    def splat(self, x: int, y: int, vector: np.ndarray) -> None:
        """Add a signature vector to a single tile (organism emission)."""
        self.values[:, y, x] += vector

    def sample(self, x: float, y: float) -> np.ndarray:
        return _bilinear(self.values, x, y).astype(np.float32)
