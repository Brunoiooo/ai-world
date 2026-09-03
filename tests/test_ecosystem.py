import numpy as np
import pytest

from ai_world.simulation.ecosystem import (
    act_system,
    default_systems,
    standing_upkeep,
    thermal_penalty,
    vitals_system,
)
from ai_world.world.generator import generate_grid
from ai_world.world.genome import Genome, Physiology
from ai_world.world.params import EcoParams
from ai_world.world.population import TRAIT_IX
from ai_world.world.world import World, attach_ecosystem


def make_world(*, population=60, seed=3, dim=64) -> World:
    params = EcoParams(initial_population=population, population_soft_cap=population * 4)
    world = World(name="eco", grid=generate_grid(dim, dim, seed), seed=seed)
    attach_ecosystem(world, params)
    return world


def run(world: World, ticks: int) -> None:
    systems = default_systems()
    for _ in range(ticks):
        for system in systems:
            system(world)
        world.tick += 1


def test_run_stays_within_invariants():
    world = make_world()
    run(world, 1200)
    pop = world.population
    assert len(pop) >= 0
    if len(pop):
        assert pop.energy.min() >= 0.0 and pop.energy.max() <= 1.0
        assert pop.hp.min() >= 0.0 and pop.hp.max() <= 1.0
        assert pop.x.min() >= 0.0 and pop.x.max() < world.width
        assert pop.y.min() >= 0.0 and pop.y.max() < world.height
    assert pop.births > 0
    assert pop.deaths > 0


def test_speed_never_exceeds_max_speed():
    world = make_world()
    run(world, 200)
    pop = world.population
    max_speed = pop.traits[:, TRAIT_IX["max_speed"]]
    assert np.all(pop.speed <= max_speed + 1e-9)


def test_hp_drops_only_while_starving_and_regens_when_sated():
    world = make_world(population=3)
    pop = world.population
    pop.energy[:] = [0.0, 0.4, 0.95]
    pop.hp[:] = 0.5
    vitals_system(world)
    assert pop.hp[0] < 0.5          # starving -> decays
    assert pop.hp[1] == pytest.approx(0.5)  # neither starving nor sated -> flat
    assert pop.hp[2] > 0.5          # sated -> regenerates


def test_starvation_kills_without_food():
    world = make_world(population=40)
    world.enzymes.values[:] = 0.0
    world.enzymes._regen_ceiling[:] = 0.0  # no regrowth either
    start = len(world.population)
    run(world, 600)
    assert len(world.population) < start
    assert world.population.deaths > 0


def test_bigger_body_costs_more_energy():
    params = EcoParams()
    small = Genome(Physiology.random(np.random.default_rng(1)), np.zeros(6, np.float32), np.zeros(3, np.float32))
    big = Genome(
        Physiology(**{**small.physiology.__dict__, "size": small.physiology.size + 1.5}),
        np.zeros(6, np.float32), np.zeros(3, np.float32),
    )
    assert standing_upkeep(big, params, age=0) > standing_upkeep(small, params, age=0)


def test_thermal_penalty_only_outside_comfort_band():
    params = EcoParams()
    ph = Physiology.random(np.random.default_rng(0))
    ph.comfort_center, ph.comfort_width = 0.5, 0.1
    genome = Genome(ph, np.zeros(6, np.float32), np.zeros(3, np.float32))
    assert thermal_penalty(0.5, genome, params) == 0.0
    assert thermal_penalty(0.55, genome, params) == 0.0
    assert thermal_penalty(0.9, genome, params) > 0.0


def test_reproduction_produces_a_mutated_child():
    world = make_world(population=4)
    pop = world.population
    pop.energy[:] = 0.95
    pop.i_turn = np.zeros(4)
    pop.i_thrust = np.zeros(4)
    pop.i_eat = np.zeros(4, dtype=bool)
    pop.i_attack = np.zeros(4, dtype=bool)
    pop.i_mate = np.ones(4, dtype=bool)

    before = len(pop)
    parent_gen = int(pop.generation[0])
    parent_energy = float(pop.energy[0])
    act_system(world)

    assert len(pop) > before
    child = pop.snapshot(len(pop) - 1)
    assert child.generation == parent_gen + 1
    assert pop.energy[0] < parent_energy
    # a mutation actually happened somewhere in the child's body
    assert not np.array_equal(child.genome.body_signature, pop.genomes[0].body_signature) or \
        child.genome.physiology != pop.genomes[0].physiology


def test_corpse_returns_enzymes_to_the_tile():
    world = make_world(population=1)
    pop = world.population
    pop.x[0], pop.y[0] = 20.5, 20.5
    pop.hp[0] = 0.0
    world.enzymes.values[20, 20] = 0.0
    vitals_system(world)
    assert len(pop) == 0
    assert world.enzymes.values[20, 20] > 0.0


def test_determinism_same_seed_same_trajectory():
    a, b = make_world(seed=7), make_world(seed=7)
    run(a, 400)
    run(b, 400)
    assert np.array_equal(a.population.id, b.population.id)
    assert np.allclose(a.population.x, b.population.x)
    assert np.allclose(a.population.energy, b.population.energy)
