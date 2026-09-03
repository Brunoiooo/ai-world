import numpy as np

from ai_world.persistence.database import connect, init_db
from ai_world.persistence.worlds import WorldRepository
from ai_world.simulation.ecosystem import default_systems
from ai_world.world.genome import Genome, Innovations, crossover, compat_distance, mutate
from ai_world.world.params import EcoParams
from ai_world.world.species import SpeciesRegistry

PARAMS = EcoParams()


def _blind(seed: int) -> Genome:
    return Genome.random_blind(np.random.default_rng(seed), PARAMS, Innovations())


def test_compat_distance_is_zero_for_identical_and_symmetric():
    g = _blind(1)
    assert compat_distance(g, g.copy()) == 0.0
    a, b = _blind(2), _blind(3)
    assert compat_distance(a, b) == compat_distance(b, a)
    assert compat_distance(a, b) > 0.0


def test_crossover_child_shares_genes_with_both_parents():
    rng = np.random.default_rng(0)
    innov = Innovations()
    a = _blind(10)
    b = _blind(11)
    for _ in range(20):
        a = mutate(a, rng, innov, PARAMS)
        b = mutate(b, rng, innov, PARAMS)

    child = crossover(a, b, rng, a_is_fitter=True)
    child_ids = {n.id for n in child.nodes}
    for conn in child.conns:
        assert conn.src in child_ids and conn.dst in child_ids
    a_innovs = {c.innov for c in a.conns}
    b_innovs = {c.innov for c in b.conns}
    child_innovs = {c.innov for c in child.conns}
    # every child connection came from one of the parents
    assert child_innovs <= (a_innovs | b_innovs)


def test_registry_splits_and_prunes():
    reg = SpeciesRegistry(threshold=0.5, target_count=3)
    genomes = [_blind(s) for s in range(30)]
    ids = np.array([reg.assign(g, tick=0) for g in genomes])
    assert len(set(ids.tolist())) > 1  # dissimilar blinds -> multiple species

    # keep only members of the first species, recount -> others pruned
    survivors_mask = ids == ids[0]
    reg.recount(ids[survivors_mask], [g for g, keep in zip(genomes, survivors_mask) if keep], tick=10)
    assert set(reg.species) == {int(ids[0])}
    assert reg.census[-1][0] == 10


def test_speciation_survives_save_load():
    conn = connect(":memory:")
    init_db(conn)
    repo = WorldRepository(conn)
    world = repo.create("sp", 96, 96, seed=5)

    systems = default_systems()
    for _ in range(400):
        for system in systems:
            system(world)
        world.tick += 1

    n_species = len(world.species.species)
    n_census = len(world.species.census)
    repo.save(world)

    reloaded = repo.load(world.id)
    assert len(reloaded.species.species) == n_species
    assert len(reloaded.species.census) == n_census
    assert reloaded.species.threshold == world.species.threshold
    conn.close()
