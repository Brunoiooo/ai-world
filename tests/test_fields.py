import numpy as np
import pytest

from ai_world.persistence.database import connect, init_db
from ai_world.persistence.serialization import (
    deserialize_eco_state,
    serialize_eco_state,
)
from ai_world.persistence.worlds import WorldRepository
from ai_world.world.fields import EnzymeField, SpectrumField, TemperatureField
from ai_world.world.generator import generate_grid
from ai_world.world.params import EcoParams
from ai_world.world.tiles import Tile
from ai_world.world.weather import WeatherState
from ai_world.world.world import attach_ecosystem


@pytest.fixture()
def params():
    return EcoParams()


@pytest.fixture()
def grid():
    return generate_grid(96, 64, seed=7)


def test_enzyme_regen_is_bounded_and_terrain_gated(grid, params):
    rng = np.random.default_rng(0)
    field = EnzymeField.for_grid(grid, params, rng)
    water = grid.cells <= int(Tile.WATER)
    rock = grid.cells == int(Tile.ROCK)

    for _ in range(2000):
        field.step()

    grass_mean = field.values[grid.cells == int(Tile.GRASS)].mean()
    assert field.values.min() >= 0.0
    assert field.values.max() <= params.enzyme_capacity + 1e-5
    assert grass_mean > 0.5
    # infertile terrain only holds what diffuses in from fertile neighbours
    assert field.values[rock].mean() < grass_mean * 0.5
    assert field.values[grid.cells == int(Tile.DEEP_WATER)].mean() < 0.1
    assert field.values[water].mean() < grass_mean


def test_enzyme_absorb_and_deposit(grid, params):
    field = EnzymeField.for_grid(grid, params, np.random.default_rng(1))
    field.values[10, 10] = 0.5
    assert field.absorb(10, 10, 0.2) == pytest.approx(0.2)
    assert field.values[10, 10] == pytest.approx(0.3)
    assert field.absorb(10, 10, 5.0) == pytest.approx(0.3)  # clamped to what's there
    field.deposit(10, 10, 0.4)
    assert field.values[10, 10] == pytest.approx(0.4)


def test_enzyme_diffusion_spreads_a_spike(grid, params):
    field = EnzymeField.for_grid(grid, params, np.random.default_rng(2))
    field.values[:] = 0.0
    field.values[32, 48] = 1.0
    neighbour_before = field.values[32, 49]
    for _ in range(5):
        field.step(regen_multiplier=0.0)
    assert field.values[32, 49] > neighbour_before
    assert field.values[32, 48] < 1.0


def test_temperature_gradient(grid, params):
    temp = TemperatureField.for_grid(grid, params, seed=7)
    h = temp.values.shape[0]
    equator = temp.values[h // 2].mean()
    pole = temp.values[:3].mean()
    assert equator > pole


def test_temperature_refresh_stays_normalized(grid, params):
    temp = TemperatureField.for_grid(grid, params, seed=7)
    baseline = temp.values.copy()
    temp.refresh(season_offset=params.season_amplitude, anomaly=0.1)
    assert temp.values.min() >= 0.0 and temp.values.max() <= 1.0
    assert not np.allclose(temp.values, baseline)


def test_spectrum_decays_toward_zero(grid, params):
    spectrum = SpectrumField.for_grid(grid, params)
    spectrum.splat(20, 20, np.ones(spectrum.channels, dtype=np.float32))
    peak = spectrum.values.max()
    for _ in range(200):
        spectrum.step()
    assert spectrum.values.max() < peak * 0.01


def test_eco_state_roundtrip(params):
    weather = WeatherState()
    weather.advance(12_345, params)
    rng = np.random.default_rng(99)
    rng.random(10)  # advance it

    blob = serialize_eco_state(params, weather, rng)
    rp, rw, rr = deserialize_eco_state(blob)

    assert rp == params
    assert rw == weather
    assert rr.random() == rng.random()


def test_world_create_attaches_ecosystem_within_size_limit():
    conn = connect(":memory:")
    init_db(conn)
    repo = WorldRepository(conn)

    small = repo.create("eco", 128, 128, seed=3)
    assert small.ecosystem_enabled
    assert small.enzymes.values.shape == (128, 128)
    assert small.spectrum.values.shape[0] == EcoParams().spectrum_channels

    big = repo.create("terrain", 1024, 1024, seed=3)
    assert not big.ecosystem_enabled
    conn.close()


def test_ecosystem_survives_save_load_roundtrip():
    conn = connect(":memory:")
    init_db(conn)
    repo = WorldRepository(conn)

    world = repo.create("eco", 96, 96, seed=11)
    world.enzymes.step()
    world.spectrum.splat(5, 5, np.full(world.spectrum.channels, 0.7, dtype=np.float32))
    world.eco_rng.random(4)
    world.tick = 8000
    repo.save(world)

    reloaded = repo.load(world.id)
    assert reloaded.ecosystem_enabled
    assert np.array_equal(reloaded.enzymes.values, world.enzymes.values)
    assert np.array_equal(reloaded.spectrum.values, world.spectrum.values)
    assert reloaded.eco_rng.random() == world.eco_rng.random()
    conn.close()


def test_ecosystem_seed_is_deterministic():
    grid = generate_grid(64, 64, seed=5)
    from ai_world.world.world import World

    a = World(name="a", grid=grid, seed=5)
    b = World(name="b", grid=grid, seed=5)
    attach_ecosystem(a, EcoParams())
    attach_ecosystem(b, EcoParams())
    assert np.array_equal(a.enzymes.values, b.enzymes.values)
    assert np.array_equal(a.temperature.values, b.temperature.values)
