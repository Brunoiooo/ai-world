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
    world.food.step(world.weather.regen_multiplier)
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
    drain = (
        standing_upkeep_vec(traits, params, pop.age.astype(np.float64))
        + params.move_cost * size * speed ** 2
        + thermal_penalty_vec(temp_here, traits, params)
        + np.where(tile == _DEEP_WATER, params.drown_penalty, 0.0)
        + params.port_upkeep * pop.brains.port_cost
        + params.brain_node_upkeep * pop.brains.n_nodes
        + params.brain_conn_upkeep * pop.brains.n_conns
        + params.food_sense_upkeep * pop.brains.food_sense_used
        + params.emit_cost * emit_cost
        + params.eat_attempt_cost * pop.i_eat.sum(axis=1)
        + params.reserve_upkeep * np.maximum(0.0, pop.energy - params.sated_energy)
    )
    pop.energy = np.maximum(0.0, pop.energy - drain)
    pop.age += 1
    pop.last_turn = pop.i_turn.copy()
    np.subtract(pop.repro_cd, 1, out=pop.repro_cd, where=pop.repro_cd > 0)

    for i in np.flatnonzero(pop.i_attack):
        _resolve_attack(pop, int(i), params)

    _reproduce(world, pop, params)


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
def _resolve_attack(pop: Population, i: int, params: EcoParams) -> None:
    target = pop.nearest_index(pop.x[i], pop.y[i], _ATTACK_RANGE, exclude=i)
    if target is None:
        return
    power = pop.traits[i, _IX_ATK]
    pop.energy[i] = max(0.0, pop.energy[i] - params.attack_cost)
    damage = max(0.0, power - pop.traits[target, _IX_ARM]) * 0.15
    pop.hp[target] -= damage
    stolen = min(pop.energy[target], damage * 0.5)
    pop.energy[target] -= stolen
    pop.energy[i] = pop.energy[i] + stolen


def _reproduce(world: World, pop: Population, params: EcoParams) -> None:
    """Sexual reproduction: pair up willing, species-compatible neighbours of a
    compatible mating type; the litter is however many NEAT crossovers (each
    mutated independently) the pair's fecundity genes and energy afford."""
    # the gates are physical, not decisions: a parent must hold at least the
    # energy for one offspring, and must be off its post-mating cooldown.
    # When to mate (within those limits) is entirely the brain's call.
    willing = pop.i_mate & (pop.energy >= params.repro_cost) & (pop.repro_cd <= 0)
    candidates = np.flatnonzero(willing)
    if candidates.size < 2:  # no population cap -- food + mortality set the size
        return

    rng = world.eco_rng
    registry = world.species
    mt = pop.mating_type
    paired: set[int] = set()
    newborns: list[Entity] = []
    lo2, hi2 = params.mating_type_lo ** 2, params.mating_type_hi ** 2

    for i in candidates:
        i = int(i)
        if i in paired:
            continue
        partner = _find_partner(pop, i, willing, paired, mt, lo2, hi2, params.mating_range)
        if partner is None:
            continue
        paired.add(i)
        paired.add(partner)

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
        cooldown_mult = 0.5 * (pop.traits[i, _IX_REPRO_MULT] + pop.traits[partner, _IX_REPRO_MULT])
        pop.repro_cd[i] = pop.repro_cd[partner] = max(1, int(round(params.repro_cooldown * cooldown_mult)))
        if n_offspring <= 0:
            continue

        a_fitter = pop.energy[i] >= pop.energy[partner]
        parent_species = int(pop.species_id[i])
        for _ in range(n_offspring):
            child_genome = mutate(
                crossover(pop.genomes[i], pop.genomes[partner], rng, a_is_fitter=a_fitter),
                rng, world.innovations, params,
            )
            cx = 0.5 * (pop.x[i] + pop.x[partner]) + rng.normal(0.0, 1.0)
            cy = 0.5 * (pop.y[i] + pop.y[partner]) + rng.normal(0.0, 1.0)
            newborns.append(
                Entity(
                    id=pop.new_id(),
                    x=min(max(cx, 0.0), world.width - 1e-3),
                    y=min(max(cy, 0.0), world.height - 1e-3),
                    heading=float(rng.random() * _TWO_PI),
                    energy=1.5 * params.repro_cost,  # < 2x: reproduction is slightly lossy
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
    best, best_d2 = None, reach * reach
    for j in pop.neighbours(pop.x[i], pop.y[i], reach):
        if j == i or j in paired or not willing[j]:
            continue
        type_d2 = float(np.sum((mt[i] - mt[j]) ** 2))
        if not (lo2 < type_d2 < hi2):
            continue
        d2 = (pop.x[i] - pop.x[j]) ** 2 + (pop.y[i] - pop.y[j]) ** 2
        if d2 < best_d2:
            best, best_d2 = j, d2
    return best


def speciation_system(world: World) -> None:
    pop = world.population
    if pop is None or not len(pop):
        return
    params = world.eco_params
    if world.tick % params.speciation_interval != 0:
        return
    world.species.recount(pop.species_id, pop.genomes, world.tick, pop.traits)


def default_systems() -> list[System]:
    return [
        weather_system,
        fields_system,
        sense_system,
        think_system,
        act_system,
        vitals_system,
        speciation_system,
    ]
