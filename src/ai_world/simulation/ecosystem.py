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

from ai_world.world.entity import Entity
from ai_world.world.genome import Genome, mutate, physiology_vector
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


# --------------------------------------------------------------------------- #
# metabolism maths (vectorised; scalar helpers delegate for unit tests)
# --------------------------------------------------------------------------- #
def standing_upkeep_vec(traits: np.ndarray, params: EcoParams, age: np.ndarray) -> np.ndarray:
    comfort_w = np.maximum(traits[:, _IX_CWIDTH], 1e-3)
    base = (
        params.base_upkeep
        + params.size_upkeep * traits[:, _IX_SIZE]
        + params.speed_upkeep * traits[:, _IX_SPEED]
        + params.regen_upkeep * traits[:, _IX_REGEN]
        + params.tolerance_upkeep * (1.0 / comfort_w - 1.0)
        + params.combat_upkeep * (traits[:, _IX_ATK] + traits[:, _IX_ARM])
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


# --------------------------------------------------------------------------- #
# systems
# --------------------------------------------------------------------------- #
def weather_system(world: World) -> None:
    params = world.eco_params
    assert params and world.weather
    if world.tick % params.weather_interval != 0:
        return
    world.weather.advance(world.tick, params)
    world.refresh_temperature()


def fields_system(world: World) -> None:
    params = world.eco_params
    assert params and world.enzymes and world.spectrum and world.temperature and world.weather
    world.enzymes.step(world.weather.regen_multiplier)
    # Signal decay/diffusion is slow on a watchable timescale — stepping it on a
    # stride (with a matching rate bump inside the field) keeps the look while
    # halving the per-tick field cost.
    if world.tick % params.spectrum_interval == 0:
        world.spectrum.step()
    world.spectrum.set_channel(params.enzyme_channel, world.enzymes.values)
    world.spectrum.set_channel(params.temperature_channel, world.temperature.values)


def sense_system(world: World) -> None:
    if world.population is not None and len(world.population):
        world.population.rebuild_index()


def think_system(world: World) -> None:
    """Placeholder brain: a momentum-biased random walk that always tries to eat."""
    pop = world.population
    if pop is None or not len(pop):
        return
    n = len(pop)
    rng = world.eco_rng
    params = world.eco_params
    pop.i_turn = np.clip(rng.normal(0.0, 0.35, n), -1.0, 1.0)
    pop.i_thrust = rng.uniform(0.15, 1.0, n)
    pop.i_eat = np.ones(n, dtype=bool)
    pop.i_attack = (pop.traits[:, _IX_ATK] > 0.35) & (rng.random(n) < 0.05)
    pop.i_mate = (pop.energy >= params.repro_threshold) & (rng.random(n) < 0.03)


def act_system(world: World) -> None:
    pop = world.population
    if pop is None or not len(pop):
        return
    params = world.eco_params
    grid, enzymes, spectrum, temperature = (
        world.grid, world.enzymes, world.spectrum, world.temperature
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

    want = params.eat_rate * size * pop.i_eat
    available = enzymes.values[ty, tx].astype(np.float64)
    taken = np.minimum(available, want)
    np.add.at(enzymes.values, (ty, tx), -taken.astype(np.float32))
    enzymes.values[ty, tx] = np.maximum(0.0, enzymes.values[ty, tx])
    pop.energy = np.minimum(1.0, pop.energy + taken * traits[:, _IX_META])

    emission = (pop.signature * (0.04 * size)[:, None]).astype(np.float32)  # (n, C)
    for channel in range(spectrum.channels):
        np.add.at(spectrum.values[channel], (ty, tx), emission[:, channel])

    temp_here = temperature.values[ty, tx]
    drain = (
        standing_upkeep_vec(traits, params, pop.age.astype(np.float64))
        + params.move_cost * speed ** 2
        + thermal_penalty_vec(temp_here, traits, params)
        + np.where(tile == _DEEP_WATER, params.drown_penalty, 0.0)
    )
    pop.energy = np.maximum(0.0, pop.energy - drain)
    pop.age += 1

    for i in np.flatnonzero(pop.i_attack):
        _resolve_attack(pop, int(i), params)

    _reproduce(world, pop, params)


def vitals_system(world: World) -> None:
    pop = world.population
    if pop is None or not len(pop):
        return
    params = world.eco_params

    starving = pop.energy <= 0.0
    sated = pop.energy >= params.sated_energy
    pop.hp = pop.hp - np.where(starving, params.hp_decay_starving, 0.0)
    pop.hp = np.minimum(
        1.0, pop.hp + np.where(sated, pop.traits[:, _IX_REGEN], 0.0)
    )

    dead = (pop.hp <= 0.0) | (pop.age >= params.max_age)
    if dead.any():
        di = np.flatnonzero(dead)
        tx, ty = pop.x[di].astype(np.intp), pop.y[di].astype(np.intp)
        mass = (
            params.corpse_enzyme_fraction * pop.traits[di, _IX_SIZE]
            + np.maximum(0.0, pop.energy[di])
        )
        np.add.at(world.enzymes.values, (ty, tx), (mass * 0.25).astype(np.float32))
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
    pop.energy[i] = min(1.0, pop.energy[i] + stolen)


def _reproduce(world: World, pop: Population, params: EcoParams) -> None:
    ready = np.flatnonzero(pop.i_mate & (pop.energy >= params.repro_threshold))
    room = params.population_soft_cap - len(pop)
    if room <= 0 or ready.size == 0:
        return
    rng = world.eco_rng
    newborns: list[Entity] = []
    for i in ready[:room]:
        i = int(i)
        pop.energy[i] -= params.repro_cost
        angle = rng.random() * _TWO_PI
        newborns.append(
            Entity(
                id=pop.new_id(),
                x=min(max(pop.x[i] + math.cos(angle), 0.0), world.width - 1e-3),
                y=min(max(pop.y[i] + math.sin(angle), 0.0), world.height - 1e-3),
                heading=float(rng.random() * _TWO_PI),
                energy=params.repro_cost,
                hp=1.0,
                genome=mutate(pop.genomes[i], rng),
                birth_tick=world.tick,
                generation=int(pop.generation[i]) + 1,
                parent_a=int(pop.id[i]),
            )
        )
    pop.add_many(newborns)
    pop.births += len(newborns)


def default_systems() -> list[System]:
    return [
        weather_system,
        fields_system,
        sense_system,
        think_system,
        act_system,
        vitals_system,
    ]
