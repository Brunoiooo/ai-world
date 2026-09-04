"""Tunable constants for the ecosystem.

Lives in ``world/`` (not ``simulation/``) so both the domain model and the
persistence layer can reference it without breaking the one-way dependency
flow. A save stores its own ``EcoParams`` so a reloaded world keeps the exact
rules it was created with.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# The ecosystem can be simulated for any map the app allows (matches
# ``config.MAX_MAP_DIM``). Field diffusion is O(tiles) per tick, so very large
# maps run slowly -- that is a deliberate trade-off, not a hard limit.
ECOSYSTEM_MAX_DIM = 4096

DEFAULT_ECOSYSTEM_DIM = 144


@dataclass(frozen=True)
class EcoParams:
    # --- world spectrum -------------------------------------------------
    spectrum_channels: int = 6
    # Reserved channel indices written by the world itself (not by organisms).
    food_channel: int = 0            # spectrum mirror of the total food density
    temperature_channel: int = 1

    # --- food field (multi-channel: see ai_world.world.food.FOOD_TYPES) ----
    food_capacity: float = 1.0
    food_growth_rate: float = 0.02     # logistic rate: existing patches expand
    food_seed_rate: float = 0.0006     # trickle onto barren fertile tiles so a
                                       # grazed-out biome can be recolonised
    food_decay: float = 0.004          # global leak toward zero per tick (grown types)
    food_derived_decay: float = 0.02   # faster leak for enzyme / carrion
    food_diffusion: float = 0.02       # spreads a patch into its neighbourhood
    food_initial_fill: float = 0.30    # starting surplus while foraging evolves

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
    spawn_energy: float = 0.55
    # No population cap: the food supply + mortality set the equilibrium.

    # --- brain --------------------------------------------------------
    brain_initial_width: int = 48  # starting batch-padding width G for the brain
                                   # tensor. NOT a cap -- BrainStore widens G on
                                   # demand as brains evolve past it. Fixed block
                                   # is 25 nodes: 15 proprio (9 + one "food here"
                                   # sense per food type) + 10 outputs (turn,
                                   # thrust, attack, mate, one eat gate per type).
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
    move_cost: float = 0.010             # x size x speed^2 (mass in motion -- a
                                         # bigger body pays more to shift itself)
    attack_cost: float = 0.02
    emit_cost: float = 0.003
    eat_attempt_cost: float = 0.0012     # x number of eat gates held open this tick.
                                         # Firing `eat` costs whether or not the tile
                                         # has food, so a brain that never gates it on
                                         # the "food here" sense bleeds energy -- eating
                                         # is a decision, not a free reflex. Raised 4x
                                         # from the original 0.0003: at that level
                                         # holding every gate open all the time was
                                         # cheaper than evolving a narrower diet, so
                                         # nothing ever specialised.
    food_sense_upkeep: float = 0.00060   # x number of food_k proprio senses this
                                         # brain actually reads (Genome.food_sense_count).
                                         # The "food here" sense is hard-wired into every
                                         # genome, but *using* it (wiring it to anything)
                                         # is not free -- same idea as an evolved port,
                                         # priced so ports remain competitive with it
                                         # instead of being strictly dominated.
    reserve_upkeep: float = 0.015        # x max(0, energy - sated_energy). Carrying a
                                         # reserve above satiety costs a fraction of it
                                         # per tick (fat is a load), so a well-fed
                                         # organism's energy settles at an equilibrium
                                         # instead of climbing without bound.
    litter_upkeep: float = 0.00060       # x physiology.litter_size -- standing cost of
                                         # carrying a high-fecundity body plan, paid every
                                         # tick whether or not the organism is currently
                                         # breeding (same pattern as size_upkeep).
    cycling_upkeep: float = 0.00080      # x max(0, 1/repro_cooldown_mult - 1) -- a
                                         # faster-than-baseline reproductive cycle costs
                                         # standing upkeep; cycling slower than baseline
                                         # is free (already priced at the point of mating).

    # --- feeding / vitals --------------------------------------------
    eat_rate: float = 0.03               # max food absorbed per tick, per food type
    food_digest_cap: float = 1.25        # ceiling on how nourishing a well-adapted diet gets
    food_toxicity: float = 0.05          # hp lost per unit of mismatched (diet < 0) intake
    enzyme_yield: float = 0.3            # fraction of intake excreted back as the enzyme type
    sated_energy: float = 0.7            # at/above this, hp regenerates
    hp_decay_starving: float = 0.02      # hp lost per tick while energy == 0
    thermal_penalty: float = 0.020       # energy/tick per unit of temperature outside the comfort band
    drown_penalty: float = 0.030         # energy/tick in deep water
    water_slow: float = 0.5              # movement multiplier on (shallow) water
    corpse_food_fraction: float = 0.6    # of body mass left as carrion on the tile on death
    # No hard age cutoff: death comes only when hp hits 0 (the aging ceiling
    # below is what eventually forces that for a well-fed body).

    # --- aging / senescence ---------------------------------------------
    # The max-hp ceiling falls from 1.0 as an organism ages, so even a
    # perpetually well-fed body still weakens and dies of old age -- no
    # perpetual-motion population. `aging_speed` is global and cannot be
    # evolved away; the genome's `senescence_rate` only modulates the slope.
    # The decline is also proportional to the organism's metabolic load --
    # a big brain / heavy body ages faster, the same way it burns more energy.
    aging_speed: float = 1.0            # 0 disables aging; 1 => a baseline body's
                                        # ceiling hits the floor around aging_scale
    aging_scale: float = 45_000.0       # reference lifespan in ticks for a minimal,
                                        # senescence_rate-0 organism
    aging_gene_influence: float = 0.6   # how much genome senescence_rate accelerates the decline
    aging_load_influence: float = 0.5   # how strongly metabolic load above the
                                        # baseline steepens the decline
    aging_hp_floor: float = 0.0         # the max-hp ceiling never drops below this

    # --- reproduction (sexual) --------------------------------------
    repro_cost: float = 0.5              # energy each parent contributes per offspring
                                         # (also the affordability gate, per offspring --
                                         # see genome.litter_size / _reproduce)
    repro_cooldown: int = 800           # baseline ticks before an organism can mate again;
                                         # the realised cooldown is this x the pair's mean
                                         # `physiology.repro_cooldown_mult` (evolvable --
                                         # see litter_size below), so it is no longer one
                                         # fixed number for the whole population -- it
                                         # bounds population growth without forcing every
                                         # organism's reproductive rhythm to synchronise.
    litter_cap: int = 12                # hard ceiling on offspring from one mating event --
                                         # not a design target (litter_size can evolve past
                                         # what typically lands here), just a safety valve
                                         # against a single Poisson outlier tick spawning an
                                         # absurd number of entities at once.
    mating_range: float = 10.0           # tiles within which a partner can be found
                                         # (wide enough that a sparse population can recover)
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
