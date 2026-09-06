"""Ecosystem simulation systems.

Each system is a ``(World) -> None`` callable registered on
:class:`ai_world.simulation.engine.Engine`; they run in list order every tick.
The organisms are stored column-wise (see :class:`ai_world.world.population`)
so metabolism and movement are array maths over the whole population at once.

Phase 2 status: perception is a stub, the brain is a momentum-biased random
walk, and reproduction is an asexual placeholder. Phases 3-4 replace all three.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np

from ai_world.world.brain import ATTACK, EAT_SLICE, MATE, THRUST, TURN
from ai_world.world.entity import Entity
from ai_world.world.food import CARRION_IX, ENZYME_IX, N_FOOD_TYPES
from ai_world.world.genome import Genome, crossover, mutate, physiology_vector
from ai_world.world.params import EcoParams
from ai_world.world.population import TRAIT_IX, Population
from ai_world.world.tiles import Tile
from ai_world.world.world import World

System = Callable[[World], None]

_TWO_PI = 2.0 * math.pi
_MAX_TURN = 0.5           # radians/tick at |turn| == 1
_ATTACK_RANGE = 1.6       # tiles
_SENESCENCE_SCALE = 20_000.0
_WATER = int(Tile.WATER)
_DEEP_WATER = int(Tile.DEEP_WATER)

_IX_SPEED = TRAIT_IX["max_speed"]
_IX_SIZE = TRAIT_IX["size"]
_IX_META = TRAIT_IX["metabolic_efficiency"]
_IX_CCENTER = TRAIT_IX["comfort_center"]
_IX_CWIDTH = TRAIT_IX["comfort_width"]
_IX_ATK = TRAIT_IX["attack_power"]
_IX_ARM = TRAIT_IX["armor"]
_IX_REGEN = TRAIT_IX["hp_regen_rate"]
_IX_SENESCENCE = TRAIT_IX["senescence_rate"]
_IX_LITTER = TRAIT_IX["litter_size"]
_IX_REPRO_MULT = TRAIT_IX["repro_cooldown_mult"]


# --------------------------------------------------------------------------- #
# metabolism maths (vectorised; scalar helpers delegate for unit tests)
# --------------------------------------------------------------------------- #
def _cycling_load(traits: np.ndarray) -> np.ndarray:
    """Extra load from a faster-than-baseline reproductive cycle. A
    ``repro_cooldown_mult`` of 1.0 is baseline (free here -- already priced at
    the point of mating); < 1.0 (cycling faster) costs standing upkeep."""
    mult = np.maximum(traits[:, _IX_REPRO_MULT], 0.05)
    return np.maximum(0.0, 1.0 / mult - 1.0)


def standing_upkeep_vec(traits: np.ndarray, params: EcoParams, age: np.ndarray) -> np.ndarray:
    comfort_w = np.maximum(traits[:, _IX_CWIDTH], 1e-3)
    base = (
        params.base_upkeep
        + params.size_upkeep * traits[:, _IX_SIZE]
        + params.speed_upkeep * traits[:, _IX_SPEED]
        + params.regen_upkeep * traits[:, _IX_REGEN]
        + params.tolerance_upkeep * (1.0 / comfort_w - 1.0)
        + params.combat_upkeep * (traits[:, _IX_ATK] + traits[:, _IX_ARM])
        + params.litter_upkeep * traits[:, _IX_LITTER]
        + params.cycling_upkeep * _cycling_load(traits)
    )
    senescence = 1.0 + traits[:, _IX_SENESCENCE] * (age / _SENESCENCE_SCALE)
    return base * senescence


def thermal_penalty_vec(
    temperature: np.ndarray, traits: np.ndarray, params: EcoParams
) -> np.ndarray:
    excess = np.abs(temperature - traits[:, _IX_CCENTER]) - traits[:, _IX_CWIDTH]
    return params.thermal_penalty * np.maximum(0.0, excess)


def standing_upkeep(genome: Genome, params: EcoParams, age: int) -> float:
    traits = physiology_vector(genome.physiology)[None, :]
    return float(standing_upkeep_vec(traits, params, np.array([age], dtype=float))[0])


def thermal_penalty(temperature: float, genome: Genome, params: EcoParams) -> float:
    traits = physiology_vector(genome.physiology)[None, :]
    return float(thermal_penalty_vec(np.array([temperature]), traits, params)[0])


# Standing metabolic load (energy/tick, minus base_upkeep) of a minimal
# blind-start organism: the level at which the aging load factor is ~1.0.
# size .0005 + speed .00042 + regen .00012 + combat .00016 + ~25 brain nodes
# .001 + ~11 conns .00022 ~= 0.0026.
_AGING_LOAD_REF = 0.0026


def senescence_load_vec(
    traits: np.ndarray,
    params: EcoParams,
    n_nodes: np.ndarray,
    n_conns: np.ndarray,
    port_cost: np.ndarray,
    food_sense_used: np.ndarray | float = 0.0,
) -> np.ndarray:
    """Aging multiplier from an organism's metabolic load. ~1.0 for a baseline
    body; grows as brain / body / ports / fecundity get more expensive, so
    complexity shortens lifespan the same way it drains energy."""
    load = (
        params.size_upkeep * traits[:, _IX_SIZE]
        + params.speed_upkeep * traits[:, _IX_SPEED]
        + params.regen_upkeep * traits[:, _IX_REGEN]
        + params.combat_upkeep * (traits[:, _IX_ATK] + traits[:, _IX_ARM])
        + params.brain_node_upkeep * n_nodes
        + params.brain_conn_upkeep * n_conns
        + params.port_upkeep * port_cost
        + params.food_sense_upkeep * np.asarray(food_sense_used, dtype=np.float64)
        + params.litter_upkeep * traits[:, _IX_LITTER]
        + params.cycling_upkeep * _cycling_load(traits)
    )
    return 1.0 + params.aging_load_influence * np.maximum(0.0, load / _AGING_LOAD_REF - 1.0)


def max_hp_vec(
    traits: np.ndarray,
    params: EcoParams,
    age: np.ndarray,
    n_nodes: np.ndarray | float = 0.0,
    n_conns: np.ndarray | float = 0.0,
    port_cost: np.ndarray | float = 0.0,
    food_sense_used: np.ndarray | float = 0.0,
) -> np.ndarray:
    """Age ceiling on hp.

    Falls linearly from 1.0 with age. ``aging_speed`` is a global knob that
    evolution cannot escape; the genome's ``senescence_rate`` steepens the
    slope, and so does a heavy metabolic load (see :func:`senescence_load_vec`).
    A body that reaches the floor can no longer sustain itself, so the
    population can never be a perpetual-motion machine.
    """
    gene_accel = 1.0 + params.aging_gene_influence * traits[:, _IX_SENESCENCE]
    load_accel = senescence_load_vec(
        traits, params,
        np.asarray(n_nodes, dtype=np.float64),
        np.asarray(n_conns, dtype=np.float64),
        np.asarray(port_cost, dtype=np.float64),
        np.asarray(food_sense_used, dtype=np.float64),
    )
    ceiling = 1.0 - params.aging_speed * gene_accel * load_accel * (age / params.aging_scale)
    return np.clip(ceiling, params.aging_hp_floor, 1.0)


def max_hp(genome: Genome, params: EcoParams, age: int) -> float:
    traits = physiology_vector(genome.physiology)[None, :]
    return float(max_hp_vec(traits, params, np.array([age], dtype=float))[0])


# --------------------------------------------------------------------------- #
# systems
# --------------------------------------------------------------------------- #
def weather_system(world: World) -> None:
    params = world.eco_params
    assert params and world.weather
    if world.tick % params.weather_interval != 0:
        return
    world.weather.advance(world.tick, params, world.eco_rng)
    world.refresh_temperature()


def fields_system(world: World) -> None:
    params = world.eco_params
    assert params and world.food and world.spectrum and world.temperature and world.weather
    assert world.eco_rng is not None
    world.food.step(world.weather.regen_multiplier, rng=world.eco_rng)
    # Signal decay/diffusion is slow on a watchable timescale — stepping it on a
    # stride (with a matching rate bump inside the field) keeps the look while
    # halving the per-tick field cost.
    if world.tick % params.spectrum_interval == 0:
        world.spectrum.step()
    world.spectrum.set_channel(params.food_channel, world.food.total())
    world.spectrum.set_channel(params.temperature_channel, world.temperature.values)


def sense_system(world: World) -> None:
    if world.population is not None and len(world.population):
        world.population.rebuild_index()


def think_system(world: World) -> None:
    """Run the whole population's recurrent brains for one tick."""
    pop = world.population
    if pop is None or not len(pop):
        return
    tx = pop.x.astype(np.intp)
    ty = pop.y.astype(np.intp)
    temp_here = world.temperature.values[ty, tx]
    food_here = world.food.values[:, ty, tx].T          # (n, N_FOOD_TYPES)
    outputs = pop.brains.step(pop, world.spectrum.values, temp_here, food_here)

    pop.i_turn = outputs[:, TURN]                 # tanh node -> -1..1
    pop.i_thrust = outputs[:, THRUST]             # sigmoid node -> 0..1
    # every action is a brain decision -- no reflexes. An organism that never
    # learns to fire an `eat` gate starves; one that never fires `mate` leaves no line.
    pop.i_eat = outputs[:, EAT_SLICE] > 0.5       # (n, N_FOOD_TYPES) per-type gate
    pop.i_attack = outputs[:, ATTACK] > 0.5
    pop.i_mate = outputs[:, MATE] > 0.5


def act_system(world: World) -> None:
    pop = world.population
    if pop is None or not len(pop):
        return
    params = world.eco_params
    grid, food, spectrum, temperature = (
        world.grid, world.food, world.spectrum, world.temperature
    )
    n = len(pop)
    traits = pop.traits
    max_speed = traits[:, _IX_SPEED]
    size = traits[:, _IX_SIZE]
    max_x, max_y = world.width - 1e-3, world.height - 1e-3

    pop.heading = np.mod(pop.heading + pop.i_turn * _MAX_TURN, _TWO_PI)
    speed = np.clip(pop.i_thrust, 0.0, 1.0) * max_speed
    tx, ty = pop.x.astype(np.intp), pop.y.astype(np.intp)
    tile = grid.cells[ty, tx]
    in_water = (tile == _WATER) | (tile == _DEEP_WATER)
    speed = np.where(in_water, speed * params.water_slow, speed)
    pop.speed = speed
    pop.x = np.clip(pop.x + np.cos(pop.heading) * speed, 0.0, max_x)
    pop.y = np.clip(pop.y + np.sin(pop.heading) * speed, 0.0, max_y)

    tx, ty = pop.x.astype(np.intp), pop.y.astype(np.intp)
    tile = grid.cells[ty, tx]

    # feeding: one gate per food type, outcome per type set by the diet gene.
    # A well-adapted diet (>0) nourishes; a mismatched one (<0) poisons.
    want = (params.eat_rate * size)[:, None] * pop.i_eat            # (n, K)
    available = food.values[:, ty, tx].T.astype(np.float64)        # (n, K)
    taken = np.minimum(available, want)
    for k in range(N_FOOD_TYPES):
        np.add.at(food.values[k], (ty, tx), -taken[:, k].astype(np.float32))
    np.clip(food.values, 0.0, None, out=food.values)

    # outcome flags for the UI: who actually drew food off a tile this tick
    pop.acted_eat = taken.sum(axis=1) > 1e-9

    digest = np.clip(pop.diet, 0.0, params.food_digest_cap)        # (n, K)
    toxic = np.clip(-pop.diet, 0.0, 1.0)
    # no hard ceiling on the reserve, but carrying one above satiety costs
    # `reserve_upkeep` per tick (see the drain below), so it settles at an
    # equilibrium rather than climbing forever.
    pop.energy = pop.energy + (taken * digest).sum(axis=1) * traits[:, _IX_META]
    pop.hp = pop.hp - (taken * toxic).sum(axis=1) * params.food_toxicity

    # a fraction of everything eaten is excreted back to the tile as enzyme,
    # food for detritivores that specialise on it
    excreted = taken.sum(axis=1) * params.enzyme_yield
    np.add.at(food.values[ENZYME_IX], (ty, tx), excreted.astype(np.float32))

    passive = (pop.signature * (0.04 * size)[:, None]).astype(np.float32)  # (n, C)
    for channel in range(spectrum.channels):
        np.add.at(spectrum.values[channel], (ty, tx), passive[:, channel])
    emit_cost = pop.brains.emit(pop, spectrum.values)

    temp_here = temperature.values[ty, tx]
    # juveniles get a grace period against the starvation gap: reduced standing
    # upkeep and no per-gate eat-attempt cost for the first `juvenile_ticks` of
    # life, so a slow learner has runway to become a competent forager instead
    # of starving mid-lesson (see EcoParams.juvenile_ticks).
    juvenile = pop.age < params.juvenile_ticks
    upkeep_mult = np.where(juvenile, params.juvenile_upkeep_mult, 1.0)
    eat_attempt_cost = np.where(juvenile, 0.0, params.eat_attempt_cost)
    drain = (
        standing_upkeep_vec(traits, params, pop.age.astype(np.float64)) * upkeep_mult
        + params.move_cost * size * speed ** 2
        + thermal_penalty_vec(temp_here, traits, params)
        + np.where(tile == _DEEP_WATER, params.drown_penalty, 0.0)
        + params.port_upkeep * pop.brains.port_cost
        + params.brain_node_upkeep * pop.brains.n_nodes
        + params.brain_conn_upkeep * pop.brains.n_conns
        + params.food_sense_upkeep * pop.brains.food_sense_used
        + params.emit_cost * emit_cost
        + eat_attempt_cost * pop.i_eat.sum(axis=1)
        + params.reserve_upkeep * np.maximum(0.0, pop.energy - params.sated_energy)
    )
    pop.energy = np.maximum(0.0, pop.energy - drain)
    pop.age += 1
    pop.last_turn = pop.i_turn.copy()
    np.subtract(pop.repro_cd, 1, out=pop.repro_cd, where=pop.repro_cd > 0)

    pop.acted_attack = np.zeros(n, dtype=bool)
    for i in np.flatnonzero(pop.i_attack):
        _resolve_attack(pop, int(i), params)

    _reproduce(world, pop, params)
    _pad_transients(pop)


def vitals_system(world: World) -> None:
    pop = world.population
    if pop is None or not len(pop):
        return
    params = world.eco_params

    ceiling = max_hp_vec(
        pop.traits, params, pop.age.astype(np.float64),
        pop.brains.n_nodes, pop.brains.n_conns, pop.brains.port_cost,
        pop.brains.food_sense_used,
    )
    starving = pop.energy <= 0.0
    sated = pop.energy >= params.sated_energy
    pop.hp = pop.hp - np.where(starving, params.hp_decay_starving, 0.0)
    pop.hp = pop.hp + np.where(sated, pop.traits[:, _IX_REGEN], 0.0)
    # the aging ceiling clamps hp down even when it was already high; it is the
    # only thing that eventually kills a perpetually well-fed body (no max_age).
    pop.hp = np.minimum(pop.hp, ceiling)

    dead = pop.hp <= 0.0
    if dead.any():
        di = np.flatnonzero(dead)
        tx, ty = pop.x[di].astype(np.intp), pop.y[di].astype(np.intp)
        mass = (
            params.corpse_food_fraction * pop.traits[di, _IX_SIZE]
            + np.maximum(0.0, pop.energy[di])
        )
        np.add.at(world.food.values[CARRION_IX], (ty, tx), (mass * 0.25).astype(np.float32))
        pop.keep(~dead)


# --------------------------------------------------------------------------- #
# helpers for the still-scalar corners
# --------------------------------------------------------------------------- #
def _pad_transients(pop: Population) -> None:
    """After a birth wave the per-tick decision / outcome arrays are shorter
    than the population; pad the tail with False (newborns did nothing this
    tick) so every consumer can index them by row without a bounds check."""
    n = len(pop)
    for name in ("i_attack", "i_mate", "acted_attack", "acted_eat", "acted_mate"):
        arr = getattr(pop, name)
        if arr.shape[0] < n:
            setattr(pop, name, np.append(arr, np.zeros(n - arr.shape[0], dtype=bool)))
    if pop.i_eat.shape[0] < n:
        pad = np.zeros((n - pop.i_eat.shape[0], pop.i_eat.shape[1]), dtype=bool)
        pop.i_eat = np.vstack([pop.i_eat, pad])


def _resolve_attack(pop: Population, i: int, params: EcoParams) -> None:
    target = pop.nearest_index(pop.x[i], pop.y[i], _ATTACK_RANGE, exclude=i)
    if target is None:
        return
    pop.acted_attack[i] = True
    power = pop.traits[i, _IX_ATK]
    pop.energy[i] = max(0.0, pop.energy[i] - params.attack_cost)
    damage = max(0.0, power - pop.traits[target, _IX_ARM]) * 0.15
    pop.hp[target] -= damage
    # a successful hit's payoff was raised (0.5 -> 0.6) alongside the lower
    # attack_cost -- landing a real hit needs to be worth more than the risk
    # of a swing that doesn't penetrate armor, or aggression never gets tried.
    stolen = min(pop.energy[target], damage * 0.6)
    pop.energy[target] -= stolen
    pop.energy[i] = pop.energy[i] + stolen


def _cooldown_scale(n: int, params: EcoParams) -> float:
    """Realised reproductive cooldown shrinks toward ``repro_cooldown_min_mult``
    as the population thins below ``mating_range_ref_pop`` -- the same density
    taper the mating range uses. Recovering survivors then breed on a shorter
    cycle, and the heritable cooldown gene can't keep a lineage pinned in the
    low-N trap while density is depressed."""
    ref = max(params.mating_range_ref_pop, 1.0)
    return float(np.clip(n / ref, params.repro_cooldown_min_mult, 1.0))


def _last_rites(world: World, pop: Population, params: EcoParams) -> None:
    """The true end of the line: exactly one organism is left alive. Sexual
    reproduction needs two, so no amount of range-widening or gate-bypassing
    in `_reproduce` can ever fire for it -- it will sit there, well-fed and
    willing, until old age or a bad tick kills it and the world goes silent.
    As a last resort (not a steady-state strategy: capped at one offspring
    per cooldown window), let it clone itself through the ordinary
    crossover-with-itself + mutate pipeline -- a real mutated descendant, not
    a bit-for-bit copy. As soon as a second individual exists, ordinary
    sexual pairing (with its species / mating-type gates) takes back over on
    the very next tick."""
    i = 0
    if not (pop.energy[i] >= params.repro_cost and pop.repro_cd[i] <= 0):
        return
    rng = world.eco_rng
    n_offspring = min(1, int(pop.energy[i] // params.repro_cost))
    cooldown_mult = pop.traits[i, _IX_REPRO_MULT] * _cooldown_scale(len(pop), params)
    pop.repro_cd[i] = max(1, int(round(params.repro_cooldown * cooldown_mult)))
    if n_offspring <= 0:
        return
    pop.energy[i] -= params.repro_cost
    pop.acted_mate[i] = True
    parent_species = int(pop.species_id[i])
    child_genome = mutate(
        crossover(pop.genomes[i], pop.genomes[i], rng, a_is_fitter=True),
        rng, world.innovations, params, rate_mult=params.low_pop_mutation_mult,
    )
    cx = pop.x[i] + rng.normal(0.0, 1.0)
    cy = pop.y[i] + rng.normal(0.0, 1.0)
    newborn = Entity(
        id=pop.new_id(),
        x=min(max(cx, 0.0), world.width - 1e-3),
        y=min(max(cy, 0.0), world.height - 1e-3),
        heading=float(rng.random() * _TWO_PI),
        energy=params.newborn_energy_mult * params.repro_cost,
        hp=1.0,
        genome=child_genome,
        birth_tick=world.tick,
        generation=int(pop.generation[i]) + 1,
        species_id=world.species.assign(child_genome, world.tick, parent_species),
        parent_a=int(pop.id[i]),
        parent_b=int(pop.id[i]),
    )
    pop.add_many([newborn])
    pop.births += 1


def _reproduce(world: World, pop: Population, params: EcoParams) -> None:
    """Sexual reproduction: pair up willing, species-compatible neighbours of a
    compatible mating type; the litter is however many NEAT crossovers (each
    mutated independently) the pair's fecundity genes and energy afford."""
    # a lone survivor has nobody to pair with -- no widened range or gate
    # override changes that, since sex needs two. Rather than let a single
    # well-fed individual's line vanish purely because of that arithmetic,
    # give it one last, asexual option (see `_last_rites`). The instant a
    # second individual exists (even its own clone) this function goes back
    # to ordinary pairing on the next tick.
    pop.acted_mate = np.zeros(len(pop), dtype=bool)
    if len(pop) == 1:
        _last_rites(world, pop, params)
        return

    # the gates are physical, not decisions: a parent must hold at least the
    # energy for one offspring, and must be off its post-mating cooldown.
    # When to mate (within those limits) is entirely the brain's call --
    # except at the very bottom, see `critical` below.
    n = len(pop)
    critical = n <= params.critical_population
    eligible = (pop.energy >= params.repro_cost) & (pop.repro_cd <= 0)
    # at or below critical_population, a lineage's `mate` output can have
    # drifted to permanently off with nobody left to select against it, so
    # bypass the brain's decision for anyone who can otherwise afford a
    # child -- there is no population left to lose by overriding it.
    willing = eligible if critical else (pop.i_mate & eligible)
    candidates = np.flatnonzero(willing)
    if candidates.size < 2:  # no population cap -- food + mortality set the size
        return

    rng = world.eco_rng
    registry = world.species
    mt = pop.mating_type
    paired = np.zeros(n, dtype=bool)
    newborns: list[Entity] = []
    lo2, hi2 = params.mating_type_lo ** 2, params.mating_type_hi ** 2

    if critical:
        # last resort: search the whole map rather than taper by a capped
        # multiplier -- a 4x cap still wasn't enough at n=2 in testing
        # (survivors 42 tiles apart, cap tops out at 40).
        reach = math.hypot(world.width, world.height)
    else:
        # mating_range is sized for a healthy population; at low headcount on
        # a large map it becomes an Allee-effect trap -- brains keep firing
        # `mate` (observed mate_frac 60-95% during a bottleneck) but almost
        # nobody is ever within range, so the population stalls or dies out
        # on pure geometry rather than fitness. Widen the search radius as
        # the population thins out, tapering back to 1x once recovered.
        range_mult = min(
            params.mating_range_max_mult,
            max(1.0, params.mating_range_ref_pop / max(n, 1)),
        )
        reach = params.mating_range * range_mult

    for i in candidates:
        i = int(i)
        if paired[i]:
            continue
        partner = _find_partner(pop, i, willing, paired, mt, lo2, hi2, reach)
        if partner is None:
            continue
        paired[i] = paired[partner] = True
        pop.acted_mate[i] = pop.acted_mate[partner] = True

        # litter size is genetically encoded (mean of both parents' evolvable
        # `litter_size`), realised as a Poisson draw so a high-fecundity
        # genotype usually raises more young but still risks a dud (0) cycle;
        # affordability caps it so a pair can never spend energy it doesn't
        # have. The cooldown is likewise the pair's own evolvable
        # `repro_cooldown_mult` x the baseline -- fecundity is a body plan,
        # not one fixed number for the whole population.
        mean_litter = 0.5 * (pop.traits[i, _IX_LITTER] + pop.traits[partner, _IX_LITTER])
        affordable = int(min(pop.energy[i], pop.energy[partner]) // params.repro_cost)
        n_offspring = min(int(rng.poisson(max(mean_litter, 0.0))), affordable, params.litter_cap)

        # cooldown always applies -- the reproductive cycle takes the same
        # recovery time whether or not it produces young -- but the energy
        # charge is per offspring, so a dud (n=0) cycle costs a shot at the
        # next window, not scarce energy on top of it.
        cost = params.repro_cost * n_offspring
        pop.energy[i] -= cost
        pop.energy[partner] -= cost
        cooldown_mult = (
            0.5 * (pop.traits[i, _IX_REPRO_MULT] + pop.traits[partner, _IX_REPRO_MULT])
            * _cooldown_scale(n, params)
        )
        pop.repro_cd[i] = pop.repro_cd[partner] = max(1, int(round(params.repro_cooldown * cooldown_mult)))
        if n_offspring <= 0:
            continue

        a_fitter = pop.energy[i] >= pop.energy[partner]
        parent_species = int(pop.species_id[i])
        # the geometric midpoint assumes parents are close together, true at
        # the normal mating_range but not under the critical "search the
        # whole map" reach above -- a child born at the midpoint of two
        # parents 40+ tiles apart can land in terrain neither parent is
        # anywhere near (observed: spawned on bare rock/snow between two
        # forest-dwellers, starved before the next census, over and over).
        # Anchor on the fitter parent's own -- already proven viable -- spot
        # instead whenever that gap could matter.
        if critical:
            anchor_x = pop.x[i] if a_fitter else pop.x[partner]
            anchor_y = pop.y[i] if a_fitter else pop.y[partner]
        rate_mult = params.low_pop_mutation_mult if critical else 1.0
        for _ in range(n_offspring):
            child_genome = mutate(
                crossover(pop.genomes[i], pop.genomes[partner], rng, a_is_fitter=a_fitter),
                rng, world.innovations, params, rate_mult=rate_mult,
            )
            if critical:
                cx = anchor_x + rng.normal(0.0, 1.0)
                cy = anchor_y + rng.normal(0.0, 1.0)
            else:
                cx = 0.5 * (pop.x[i] + pop.x[partner]) + rng.normal(0.0, 1.0)
                cy = 0.5 * (pop.y[i] + pop.y[partner]) + rng.normal(0.0, 1.0)
            newborns.append(
                Entity(
                    id=pop.new_id(),
                    x=min(max(cx, 0.0), world.width - 1e-3),
                    y=min(max(cy, 0.0), world.height - 1e-3),
                    heading=float(rng.random() * _TWO_PI),
                    energy=params.newborn_energy_mult * params.repro_cost,
                    hp=1.0,
                    genome=child_genome,
                    birth_tick=world.tick,
                    generation=max(int(pop.generation[i]), int(pop.generation[partner])) + 1,
                    species_id=registry.assign(child_genome, world.tick, parent_species),
                    parent_a=int(pop.id[i]),
                    parent_b=int(pop.id[partner]),
                )
            )
    pop.add_many(newborns)
    pop.births += len(newborns)


def _find_partner(pop, i, willing, paired, mt, lo2, hi2, reach) -> int | None:
    # Reproductive compatibility is judged by the evolvable `mating_type` band
    # alone, not by `species_id` -- species is a compat_distance classification
    # for stats/diversity (SpeciesRegistry), not a reproductive-isolation gate.
    # Requiring an exact species match here can deadlock a small population:
    # the moment it (harmlessly) splits into two species, same-species pairs
    # may no longer exist locally and nobody could ever mate again.
    #
    # Vectorised: `neighbour_rows` already returns the in-reach rows in the same
    # order the old per-row loop visited them, so the closest survivor of the
    # mating-type band is `argmin(d2)` -- same pick, same first-wins tie rule.
    cand = pop.neighbour_rows(pop.x[i], pop.y[i], reach)
    if not cand.size:
        return None
    cand = cand[willing[cand] & ~paired[cand] & (cand != i)]
    if not cand.size:
        return None
    type_d2 = ((mt[cand] - mt[i]) ** 2).sum(axis=1)
    cand = cand[(type_d2 > lo2) & (type_d2 < hi2)]
    if not cand.size:
        return None
    dx = pop.x[cand] - pop.x[i]
    dy = pop.y[cand] - pop.y[i]
    return int(cand[np.argmin(dx * dx + dy * dy)])


def speciation_system(world: World) -> None:
    pop = world.population
    if pop is None or not len(pop):
        return
    params = world.eco_params
    if world.tick % params.speciation_interval != 0:
        return
    world.species.recount(pop.species_id, pop.genomes, world.tick, pop.traits)


_IMMIGRATION_TILES = (
    int(Tile.SAND), int(Tile.GRASS), int(Tile.FOREST), int(Tile.DIRT),
)


def immigration_system(world: World) -> None:
    """While the population is critically small, trickle in one fresh
    ``random_blind`` organism every ``immigration_interval`` ticks. After a
    bottleneck the residents are a monoculture of mediocre foragers with no
    genetic raw material left for selection to work with; a slow drip of fresh
    genes is cheap insurance that *some* lineage carries what recovery needs.
    An immigrant lands next to a resident (so it can still find a mate) or, if
    the world has gone empty, on a spawn tile."""
    params = world.eco_params
    pop = world.population
    assert params is not None
    if (
        pop is None
        or params.immigration_interval <= 0
        or world.tick == 0
        or world.tick % params.immigration_interval != 0
        or len(pop) >= params.immigration_below
    ):
        return
    rng = world.eco_rng
    assert rng is not None and world.innovations is not None
    genome = Genome.random_blind(rng, params, world.innovations)
    if len(pop):
        anchor = int(rng.integers(len(pop)))
        x = float(np.clip(pop.x[anchor] + rng.normal(0.0, 3.0), 0.0, world.width - 1e-3))
        y = float(np.clip(pop.y[anchor] + rng.normal(0.0, 3.0), 0.0, world.height - 1e-3))
    else:
        land = np.argwhere(np.isin(world.grid.cells, _IMMIGRATION_TILES))
        if not len(land):
            return
        row = land[int(rng.integers(len(land)))]
        x, y = float(row[1]) + 0.5, float(row[0]) + 0.5
    newborn = Entity(
        id=pop.new_id(),
        x=x,
        y=y,
        heading=float(rng.random() * _TWO_PI),
        energy=params.spawn_energy,
        hp=1.0,
        genome=genome,
        birth_tick=world.tick,
        species_id=world.species.assign(genome, world.tick),
    )
    pop.add_many([newborn])
    pop.births += 1
    # keep the per-tick decision / outcome arrays indexable by row until
    # think_system refreshes them next tick (the UI reads them in between).
    _pad_transients(pop)


def default_systems() -> list[System]:
    return [
        weather_system,
        fields_system,
        sense_system,
        think_system,
        act_system,
        vitals_system,
        speciation_system,
        immigration_system,
    ]
