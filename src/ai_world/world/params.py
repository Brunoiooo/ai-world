"""Tunable constants for the ecosystem.

Lives in ``world/`` (not ``simulation/``) so both the domain model and the
persistence layer can reference it without breaking the one-way dependency
flow. A save stores its own ``EcoParams`` so a reloaded world keeps the exact
rules it was created with.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Ecosystem is only simulated for maps up to this size (field diffusion over
# every tile every tick gets too expensive beyond it). Larger maps stay
# terrain-only.
ECOSYSTEM_MAX_DIM = 512

DEFAULT_ECOSYSTEM_DIM = 192


@dataclass(frozen=True)
class EcoParams:
    # --- world spectrum -------------------------------------------------
    spectrum_channels: int = 6
    # Reserved channel indices written by the world itself (not by organisms).
    enzyme_channel: int = 0
    temperature_channel: int = 1

    # --- enzyme (food) field -----------------------------------------
    enzyme_capacity: float = 1.0
    enzyme_regen_rate: float = 0.03  # pull toward the per-tile ceiling per tick
    enzyme_decay: float = 0.01       # global leak toward zero per tick
    enzyme_diffusion: float = 0.015
    enzyme_initial_fill: float = 0.35

    # --- spectrum field --------------------------------------------------
    spectrum_decay: float = 0.08
    spectrum_diffusion: float = 0.06

    # --- temperature / seasons -------------------------------------------
    season_period_ticks: int = 60_000
    season_amplitude: float = 0.18  # in normalized (0..1) temperature units
    altitude_lapse: float = 0.45    # normalized temperature lost from sea level to peak

    # --- simulation cadence (run these systems every N ticks) -----------
    weather_interval: int = 4
    speciation_interval: int = 10
    census_interval: int = 25

    def channel_count(self) -> int:
        return self.spectrum_channels
