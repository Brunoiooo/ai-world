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

DEFAULT_ECOSYSTEM_DIM = 144


@dataclass(frozen=True)
class EcoParams:
    # --- world spectrum -------------------------------------------------
    spectrum_channels: int = 6
    # Reserved channel indices written by the world itself (not by organisms).
    enzyme_channel: int = 0
    temperature_channel: int = 1

    # --- enzyme (food) field -----------------------------------------
    enzyme_capacity: float = 1.0
    enzyme_regen_rate: float = 0.010  # pull toward the per-tile ceiling per tick
    enzyme_decay: float = 0.006       # global leak toward zero per tick
    enzyme_diffusion: float = 0.012
    enzyme_initial_fill: float = 0.5

    # --- spectrum field --------------------------------------------------
    spectrum_decay: float = 0.16
    spectrum_diffusion: float = 0.12
    spectrum_interval: int = 2  # step the spectrum field every N ticks

    # --- temperature / seasons -------------------------------------------
    season_period_ticks: int = 60_000
    season_amplitude: float = 0.18   # in normalized (0..1) temperature units
    altitude_lapse: float = 0.45     # normalized temperature lost from sea level to peak
    front_amplitude: float = 0.12    # local temperature swing from drifting weather fronts
    climate_event_chance: float = 0.00015          # per weather tick
    climate_event_ticks: tuple = (3_000, 12_000)   # [min, max) duration

    # --- population ----------------------------------------------------
    initial_population: int = 500
    population_soft_cap: int = 900    # a safety ceiling; scarce food is the real limit
    spawn_energy: float = 0.55

    # --- brain --------------------------------------------------------
    brain_max_nodes: int = 32      # hard cap on nodes per organism (batch padding width)
    brain_sensor_samples: int = 5  # ray samples per IN port per tick

    # --- metabolism (all drains are energy per tick) -------------------
    base_upkeep: float = 0.00110
    size_upkeep: float = 0.00050         # x physiology.size
    speed_upkeep: float = 0.00070        # x physiology.max_speed (standing cost of the capacity)
    regen_upkeep: float = 0.012          # x physiology.hp_regen_rate (self-repair is expensive)
    tolerance_upkeep: float = 0.00070    # x (1 / comfort_width - 1), cost of a narrow comfort band
    combat_upkeep: float = 0.00040       # x (attack_power + armor)
    brain_node_upkeep: float = 0.000040  # x active brain nodes
    brain_conn_upkeep: float = 0.000020  # x enabled connections
    port_upkeep: float = 0.00020         # x sum(gain * reach^2 / arc)
    move_cost: float = 0.008             # x speed^2
    attack_cost: float = 0.02
    emit_cost: float = 0.003

    # --- feeding / vitals --------------------------------------------
    eat_rate: float = 0.06               # max enzyme absorbed per tick
    sated_energy: float = 0.7            # at/above this, hp regenerates
    hp_decay_starving: float = 0.02      # hp lost per tick while energy == 0
    thermal_penalty: float = 0.020       # energy/tick per unit of temperature outside the comfort band
    drown_penalty: float = 0.030         # energy/tick in deep water
    water_slow: float = 0.5              # movement multiplier on (shallow) water
    corpse_enzyme_fraction: float = 0.6  # of body mass returned to the tile on death
    max_age: int = 200_000              # hard senescence cutoff (safety)

    # --- aging / senescence ---------------------------------------------
    # The max-hp ceiling falls from 1.0 as an organism ages, so even a
    # perpetually well-fed body still weakens and dies of old age -- no
    # perpetual-motion population. `aging_speed` is global and cannot be
    # evolved away; the genome's `senescence_rate` only modulates the slope.
    aging_speed: float = 1.0            # 0 disables aging; 1 => ceiling hits the floor at aging_scale
    aging_scale: float = 45_000.0       # nominal lifespan in ticks (at senescence_rate 0)
    aging_gene_influence: float = 0.6   # how much genome senescence_rate accelerates the decline
    aging_hp_floor: float = 0.0         # the max-hp ceiling never drops below this

    # --- reproduction (sexual) --------------------------------------
    repro_threshold: float = 0.85        # only well-fed organisms can mate
    repro_cost: float = 0.34             # energy each parent contributes to the offspring
    mating_range: float = 6.0            # tiles within which a partner can be found
    mating_type_lo: float = 0.15         # partners must differ by more than this
    mating_type_hi: float = 4.0          # ...and less than this (same broad type)
    species_threshold: float = 8.0       # initial compatibility distance (starts as ~1 species)
    species_target: int = 12             # threshold self-adjusts toward this many species

    # --- simulation cadence (run these systems every N ticks) -----------
    weather_interval: int = 4
    speciation_interval: int = 10
    census_interval: int = 25

    def channel_count(self) -> int:
        return self.spectrum_channels
