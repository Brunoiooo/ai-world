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
    spectrum_decay: float = 0.16
    spectrum_diffusion: float = 0.12
    spectrum_interval: int = 2  # step the spectrum field every N ticks

    # --- temperature / seasons -------------------------------------------
    season_period_ticks: int = 60_000
    season_amplitude: float = 0.18  # in normalized (0..1) temperature units
    altitude_lapse: float = 0.45    # normalized temperature lost from sea level to peak

    # --- population ----------------------------------------------------
    initial_population: int = 300
    population_soft_cap: int = 800
    spawn_energy: float = 0.5

    # --- metabolism (all drains are energy per tick) -------------------
    base_upkeep: float = 0.0009
    size_upkeep: float = 0.0011          # x physiology.size
    speed_upkeep: float = 0.0016         # x physiology.max_speed (standing cost of the capacity)
    regen_upkeep: float = 0.030          # x physiology.hp_regen_rate (self-repair is expensive)
    tolerance_upkeep: float = 0.0016     # x (1 / comfort_width - 1), cost of a narrow comfort band
    combat_upkeep: float = 0.0009        # x (attack_power + armor)
    brain_node_upkeep: float = 0.00010   # x active brain nodes   (Phase 3)
    brain_conn_upkeep: float = 0.00003   # x enabled connections  (Phase 3)
    port_upkeep: float = 0.0006          # x sum(reach^2 / arc)    (Phase 3)
    move_cost: float = 0.010             # x (speed / max_speed_ref)^2
    attack_cost: float = 0.02
    emit_cost: float = 0.004

    # --- feeding / vitals --------------------------------------------
    eat_rate: float = 0.06               # max enzyme absorbed per tick
    sated_energy: float = 0.7            # at/above this, hp regenerates
    hp_decay_starving: float = 0.02      # hp lost per tick while energy == 0
    thermal_penalty: float = 0.020       # energy/tick per unit of temperature outside the comfort band
    drown_penalty: float = 0.030         # energy/tick in deep water
    water_slow: float = 0.5              # movement multiplier on (shallow) water
    corpse_enzyme_fraction: float = 0.6  # of body mass returned to the tile on death
    max_age: int = 200_000              # hard senescence cutoff (safety)

    # --- reproduction (Phase 2: asexual placeholder; Phase 4: sexual) --
    repro_threshold: float = 0.75        # min energy to reproduce
    repro_cost: float = 0.45             # energy handed to the offspring (per parent in Phase 4)

    # --- simulation cadence (run these systems every N ticks) -----------
    weather_interval: int = 4
    speciation_interval: int = 10
    census_interval: int = 25

    def channel_count(self) -> int:
        return self.spectrum_channels
