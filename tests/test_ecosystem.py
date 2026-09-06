import numpy as np
import pytest

from ai_world.simulation.ecosystem import (
    act_system,
    default_systems,
    max_hp,
    standing_upkeep,
    thermal_penalty,
    vitals_system,
)
from ai_world.world.food import CARRION_IX, N_FOOD_TYPES
from ai_world.world.generator import generate_grid
from ai_world.world.genome import Genome, Physiology, physiology_vector
from ai_world.world.params import EcoParams
from ai_world.world.population import TRAIT_IX
from ai_world.world.world import World, attach_ecosystem


def make_world(*, population=90, seed=3, dim=80) -> World:
    params = EcoParams(initial_population=population)
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
    run(world, 500)
    pop = world.population
    assert len(pop) >= 0
    if len(pop):
        assert pop.energy.min() >= 0.0  # no upper cap on the energy reserve
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
    pop.traits[:, TRAIT_IX["hp_regen_rate"]] = 0.02  # don't lean on the random draw
    vitals_system(world)
    assert pop.hp[0] < 0.5          # starving -> decays
    assert pop.hp[1] == pytest.approx(0.5)  # neither starving nor sated -> flat
    assert pop.hp[2] > 0.5          # sated -> regenerates


def test_max_hp_ceiling_falls_with_age():
    params = EcoParams()
    ph = Physiology.random(np.random.default_rng(0))
    ph.senescence_rate = 0.0  # a no-senescence genome still ages (global aging_speed)
    genome = Genome(ph, np.zeros(6, np.float32), np.zeros(3, np.float32))
    young = max_hp(genome, params, age=0)
    middle = max_hp(genome, params, age=10_000)
    old = max_hp(genome, params, age=int(params.aging_scale))
    assert young == pytest.approx(1.0)
    assert middle < young
    assert old == pytest.approx(params.aging_hp_floor, abs=1e-6)


def test_senescence_rate_steepens_aging():
    params = EcoParams()
    base = Physiology.random(np.random.default_rng(1)).__dict__
    slow = Genome(Physiology(**{**base, "senescence_rate": 0.0}),
                  np.zeros(6, np.float32), np.zeros(3, np.float32))
    fast = Genome(Physiology(**{**base, "senescence_rate": 2.0}),
                  np.zeros(6, np.float32), np.zeros(3, np.float32))
    assert max_hp(fast, params, 15_000) < max_hp(slow, params, 15_000)


def test_heavy_metabolic_load_ages_faster():
    from ai_world.simulation.ecosystem import max_hp_vec

    params = EcoParams()
    traits = physiology_vector(
        Physiology(max_speed=0.6, size=1.0, metabolic_efficiency=0.8,
                   comfort_center=0.5, comfort_width=0.25, attack_power=0.2,
                   armor=0.2, hp_regen_rate=0.01, mutation_rate=0.5,
                   senescence_rate=0.0, repro_cooldown_mult=1.0, litter_size=1.0)
    )[None, :]
    age = np.array([12_000.0])
    lean = max_hp_vec(traits, params, age, n_nodes=25, n_conns=11, port_cost=0.0)[0]
    heavy = max_hp_vec(traits, params, age, n_nodes=400, n_conns=1500, port_cost=200.0)[0]
    assert heavy < lean  # same body + age, but the expensive brain wears out sooner


def test_aging_speed_zero_keeps_bodies_immortal():
    params = EcoParams(aging_speed=0.0)
    genome = Genome(Physiology.random(np.random.default_rng(2)),
                    np.zeros(6, np.float32), np.zeros(3, np.float32))
    assert max_hp(genome, params, age=1_000_000) == pytest.approx(1.0)


def test_old_well_fed_organism_still_dies_of_old_age():
    world = make_world(population=6)
    pop = world.population
    pop.energy[:] = 1.0   # perfectly fed: no starvation, hp would otherwise regen forever
    pop.hp[:] = 1.0
    pop.age[:] = int(world.eco_params.aging_scale) + 5_000
    before = len(pop)
    vitals_system(world)
    assert len(pop) < before
    assert pop.deaths > 0


def test_starvation_kills_without_food():
    world = make_world(population=40)
    world.food.values[:] = 0.0
    world.food._ceiling[:] = 0.0  # no regrowth either
    start = len(world.population)
    run(world, 400)
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


def test_sexual_reproduction_crosses_two_parents():
    world = make_world(population=6)
    pop = world.population
    # put two organisms next to each other, same species, willing, well-fed
    pop.x[0], pop.y[0] = 40.0, 40.0
    pop.x[1], pop.y[1] = 40.6, 40.0
    pop.species_id[0] = pop.species_id[1] = 1
    pop.mating_type[0] = [0.0, 0.0, 0.0]
    pop.mating_type[1] = [1.0, 0.0, 0.0]  # distance 1.0, inside the band
    pop.energy[:] = 1.1  # comfortably above 2x repro_cost even after a tick of upkeep
    # a high litter-size gene all but guarantees the Poisson draw clears the
    # affordability cap below, so the realised litter is deterministic: 2.
    pop.traits[0, TRAIT_IX["litter_size"]] = 50.0
    pop.traits[1, TRAIT_IX["litter_size"]] = 50.0
    pop.rebuild_index()
    n = len(pop)
    pop.i_turn = np.zeros(n)
    pop.i_thrust = np.zeros(n)
    pop.i_eat = np.zeros((n, N_FOOD_TYPES), dtype=bool)
    pop.i_attack = np.zeros(n, dtype=bool)
    # only the intended pair (0, 1) is willing -- at this population size the
    # density-adaptive mating range (see _reproduce) widens well past 10
    # tiles, so leaving the other filler organisms willing too is no longer
    # safely isolated by distance alone.
    pop.i_mate = np.zeros(n, dtype=bool)
    pop.i_mate[0] = pop.i_mate[1] = True

    before = len(pop)
    act_system(world)

    assert len(pop) == before + 2  # energy affords exactly 2 at repro_cost=0.5
    for offset in (0, 1):
        child = pop.snapshot(before + offset)
        assert child.parent_a in (int(pop.id[0]), int(pop.id[1]))
        assert child.parent_b in (int(pop.id[0]), int(pop.id[1]))
        assert child.parent_a != child.parent_b
        assert child.species_id >= 1
    assert pop.energy[0] < 0.2 and pop.energy[1] < 0.2  # spent 2x repro_cost


def test_corpse_leaves_carrion_on_the_tile():
    world = make_world(population=1)
    pop = world.population
    pop.x[0], pop.y[0] = 20.5, 20.5
    pop.hp[0] = 0.0
    pop.energy[0] = 0.3  # not sated -> no hp regen this tick
    world.food.values[CARRION_IX, 20, 20] = 0.0
    vitals_system(world)
    assert len(pop) == 0
    assert world.food.values[CARRION_IX, 20, 20] > 0.0


def _grass_tile(world):
    from ai_world.world.tiles import Tile

    ys, xs = np.where(world.grid.cells == int(Tile.GRASS))
    return int(xs[0]), int(ys[0])


def _grass_food_ix():
    from ai_world.world.food import FOOD_TYPES
    from ai_world.world.tiles import Tile

    return next(i for i, f in enumerate(FOOD_TYPES) if f.terrain == int(Tile.GRASS))


def _feed_once(world, diet_value):
    """Park one organism on a grass tile, force it to fire only the grass eat
    gate, run act_system once, return (d_energy, d_hp)."""
    pop = world.population
    k = _grass_food_ix()
    tx, ty = _grass_tile(world)
    pop.x[0], pop.y[0] = tx + 0.5, ty + 0.5
    pop.energy[0], pop.hp[0] = 0.4, 0.9
    pop.diet[0] = 0.0
    pop.diet[0, k] = diet_value
    world.food.values[:, ty, tx] = 0.0
    world.food.values[k, ty, tx] = 1.0

    n = len(pop)
    pop.i_turn = np.zeros(n)
    pop.i_thrust = np.zeros(n)
    pop.i_attack = np.zeros(n, dtype=bool)
    pop.i_mate = np.zeros(n, dtype=bool)
    pop.i_eat = np.zeros((n, N_FOOD_TYPES), dtype=bool)
    pop.i_eat[0, k] = True
    pop.rebuild_index()

    e0, h0 = pop.energy[0], pop.hp[0]
    act_system(world)
    return pop.energy[0] - e0, pop.hp[0] - h0


def test_adapted_diet_feeds_mismatched_diet_poisons():
    well = make_world(population=4)
    d_energy, d_hp = _feed_once(well, diet_value=1.0)
    assert d_energy > 0.0          # a specialist gains energy
    assert d_hp > -1e-6            # ...and takes no toxic damage

    ill = make_world(population=4)
    d_energy, d_hp = _feed_once(ill, diet_value=-1.0)
    assert d_hp < 0.0             # eating what you can't digest hurts
    assert d_energy <= 0.0        # ...and feeds you nothing (upkeep only)


def test_feeding_excretes_enzyme_onto_the_tile():
    from ai_world.world.food import ENZYME_IX

    world = make_world(population=4)
    k = _grass_food_ix()
    tx, ty = _grass_tile(world)
    world.food.values[ENZYME_IX, ty, tx] = 0.0
    _feed_once(world, diet_value=1.0)
    assert world.food.values[ENZYME_IX, ty, tx] > 0.0


def test_determinism_same_seed_same_trajectory():
    a, b = make_world(seed=7), make_world(seed=7)
    run(a, 400)
    run(b, 400)
    assert np.array_equal(a.population.id, b.population.id)
    assert np.allclose(a.population.x, b.population.x)
    assert np.allclose(a.population.energy, b.population.energy)
