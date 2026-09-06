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
    food_growth_chance: float = 0.2    # per-tile, per-tick odds that growth actually
                                       # lands; when it does it's scaled by 1/chance so
                                       # the long-run average matches food_growth_rate,
                                       # but tick-to-tick it looks patchy, not a smooth ramp
    food_seed_rate: float = 0.0006     # trickle onto barren fertile tiles so a
                                       # grazed-out biome can be recolonised
    food_decay: float = 0.004          # global leak toward zero per tick (grown types)
    food_derived_decay: float = 0.02   # faster leak for enzyme / carrion
    food_diffusion: float = 0.02       # spreads a patch into its neighbourhood
    food_initial_fill: float = 0.55    # starting surplus while foraging evolves (raised
                                       # from 0.30 -- a food-rich early world keeps more
                                       # founder lineages alive through the learning phase)
    blind_diet_mean: float = 0.32      # mean of a blind organism's per-food digestion
                                       # weights (was a hard-coded 0.15). At 0.15 early-life
                                       # feeding yielded intake x ~0.15 x metabolic_eff --
                                       # nearly nothing -- so juveniles couldn't break even
                                       # before a real specialisation evolved.

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
    spawn_energy: float = 1.6          # was 0.8, 0.55 before that. The founder die-off
                                        # (~600 -> single digits in 500 ticks) is a ~99%
                                        # cull that leaves ~1 species and almost no genetic
                                        # raw material, which is why post-crash recovery
                                        # was a coin-flip (3/9 seeds). A bigger starting
                                        # buffer carries more of the founders -- and their
                                        # variance -- through the learning phase.
    # No population cap: the food supply + mortality set the equilibrium.

    # --- juveniles: a grace period against the starvation gap -----------
    # ~91% of all deaths are organisms younger than 200 ticks, median age 84,
    # every one at energy ~0 (starvation). Newborns *do* feed (98% draw food
    # before dying) but stay net-negative until their recurrent brain learns
    # to hold station on a patch; the 1.5x-repro_cost starting buffer only
    # covers ~84 ticks of that. These widen the runway: for the first
    # `juvenile_ticks` of life, standing upkeep is scaled by
    # `juvenile_upkeep_mult` and the per-gate `eat_attempt_cost` is waived.
    newborn_energy_mult: float = 2.2    # newborn starting energy = this x repro_cost
                                        # (was a hard-coded 1.5)
    juvenile_ticks: int = 220
    juvenile_upkeep_mult: float = 0.55

    # --- reproduction: density-adaptive cooldown ------------------------
    # Mirror of the mating-range widening below. At low headcount the realised
    # `repro_cooldown` is scaled down toward `repro_cooldown_min_mult` (linearly
    # in n / mating_range_ref_pop), so the survivors that *do* break even get
    # more reproductive attempts per unit time -- and the cooldown gene can't
    # keep the lineage in the low-N trap while density is depressed.
    repro_cooldown_min_mult: float = 0.4

    # --- low-population rescue: fresh genes + forced exploration --------
    immigration_below: int = 16        # while the population is under this, drop in one
    immigration_interval: int = 1800   # fresh random_blind organism every N ticks -- cheap
                                        # insurance against a monoculture of marginal
                                        # generalists (0 interval disables).
    low_pop_mutation_mult: float = 2.5  # x mutation_rate for offspring produced while the
                                        # population is at/below critical_population, so the
                                        # last-resort breeding path explores instead of
                                        # copying a genome that already wasn't good enough.

    # The initial cohort is dropped in a handful of clusters rather than spread
    # uniformly across the whole map. A uniform spawn scatters survivors of the
    # early die-off (see spawn_energy above, and genome.random_blind's seed
    # bias) far beyond mating_range, so a population that drops to a handful of
    # individuals can end up too sparse to ever find a partner again (observed:
    # ~66-tile mean pairwise distance among 8 survivors on a 144x144 map). A
    # clustered spawn keeps founders -- and the founders' descendants -- close
    # enough to recover from a bad cull instead of stalling or dying out.
    spawn_clusters: int = 8
    spawn_cluster_radius: float = 10.0  # tiles; matches mating_range by design

    # --- reproduction: density-adaptive mating range ---------------------
    # mating_range (below) is sized for a healthy population; at very low
    # population it becomes an Allee-effect trap (mate_frac stays high but
    # nobody is ever in range). Below mating_range_ref_pop, the effective
    # search radius widens up to mating_range_max_mult x, tapering back to 1x
    # once the population recovers.
    mating_range_ref_pop: float = 50.0
    mating_range_max_mult: float = 4.0

    # --- reproduction: last-resort rescue at the very bottom -------------
    # At or below this many living organisms, two more rules relax on top of
    # the density scaling above:
    #  - the mating_range_max_mult cap comes off entirely (search the whole
    #    map) -- 4x still wasn't enough at n=2 in testing (survivors 42 tiles
    #    apart, cap tops out at 40);
    #  - the brain's `mate` output is bypassed for anyone who can otherwise
    #    afford a child. A lineage's mate drive can drift to permanently off
    #    through ordinary mutation with nobody left to select against it --
    #    observed directly: two survivors sitting at full energy, zero
    #    cooldown, compatible mating types, for 13,000+ ticks, `i_mate=False`
    #    every single tick. Below this floor there is no population left to
    #    lose by overriding that decision.
    critical_population: int = 3

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
    attack_cost: float = 0.01            # was 0.02 -- with armor cheap to raise to parity
                                         # with attack (same combat_upkeep coefficient for
                                         # both), a swing that fails to penetrate armor
                                         # still paid this flat cost, making attack a
                                         # strictly dominated strategy in practice
                                         # (observed: attack_frac stayed <=0.1% across
                                         # every run). Halved so a losing swing is cheap
                                         # enough that aggression stays a live option
                                         # instead of being taxed out of the gene pool
                                         # before it can be tested against armor.
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
