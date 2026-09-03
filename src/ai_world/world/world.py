"""World aggregate: metadata + grid + tick counter + (optional) ecosystem."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from ai_world.world.fields import EnzymeField, SpectrumField, TemperatureField
from ai_world.world.grid import Grid
from ai_world.world.params import ECOSYSTEM_MAX_DIM, EcoParams
from ai_world.world.weather import WeatherState

_ECO_RNG_SALT = 0xEC05


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class World:
    name: str
    grid: Grid
    seed: int
    tick: int = 0
    id: int | None = None
    created_at: datetime = field(default_factory=_now)
    updated_at: datetime = field(default_factory=_now)

    # --- ecosystem (present only when enabled for this world) -----------
    eco_params: EcoParams | None = None
    eco_rng: np.random.Generator | None = None
    enzymes: EnzymeField | None = None
    spectrum: SpectrumField | None = None
    temperature: TemperatureField | None = None
    weather: WeatherState | None = None

    @property
    def width(self) -> int:
        return self.grid.width

    @property
    def height(self) -> int:
        return self.grid.height

    @property
    def ecosystem_enabled(self) -> bool:
        return self.enzymes is not None

    def touch(self) -> None:
        self.updated_at = _now()

    def refresh_temperature(self) -> None:
        assert self.temperature and self.weather and self.eco_params
        self.temperature.refresh(
            self.weather.season_offset_scaled(self.eco_params.season_amplitude),
            self.weather.temperature_anomaly,
        )


def ecosystem_fits(width: int, height: int) -> bool:
    return width <= ECOSYSTEM_MAX_DIM and height <= ECOSYSTEM_MAX_DIM


def attach_ecosystem(world: World, params: EcoParams) -> None:
    """Create a fresh field substrate + climate for ``world`` in place.

    Organisms and the species registry are attached by later phases; this sets
    up only the physical fields and climate.
    """
    world.eco_params = params
    world.eco_rng = np.random.default_rng((world.seed ^ _ECO_RNG_SALT) & 0xFFFFFFFF)
    world.enzymes = EnzymeField.for_grid(world.grid, params, world.eco_rng)
    world.spectrum = SpectrumField.for_grid(world.grid, params)
    world.temperature = TemperatureField.for_grid(world.grid, params, world.seed)
    world.weather = WeatherState()
    world.weather.advance(world.tick, params)
    world.refresh_temperature()


def rebuild_ecosystem(
    world: World,
    params: EcoParams,
    weather: WeatherState,
    rng: np.random.Generator,
    enzyme_values: np.ndarray,
    spectrum_values: np.ndarray,
) -> None:
    """Restore a persisted ecosystem onto ``world``."""
    world.eco_params = params
    world.eco_rng = rng
    world.weather = weather
    world.enzymes = EnzymeField.rebuild(world.grid, params, enzyme_values)
    world.spectrum = SpectrumField.rebuild(params, spectrum_values)
    world.temperature = TemperatureField.for_grid(world.grid, params, world.seed)
    world.refresh_temperature()
