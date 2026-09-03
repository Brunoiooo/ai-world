import numpy as np
import pytest

from ai_world.world.genome import (
    FIRST_DYNAMIC_NODE,
    Genome,
    Innovations,
    N_FIXED_OUT,
    N_PROPRIO,
    mutate,
)
from ai_world.world.params import EcoParams

PARAMS = EcoParams()


def _assert_valid(genome: Genome) -> None:
    node_ids = [n.id for n in genome.nodes]
    assert len(node_ids) == len(set(node_ids)), "duplicate node id"
    valid = set(node_ids)
    for conn in genome.conns:
        assert conn.src in valid and conn.dst in valid
        assert np.isfinite(conn.weight)
    for port in genome.ports:
        assert port.node_id in valid
        assert any(n.id == port.node_id and n.kind in ("in_port", "out_port")
                   for n in genome.nodes)
    # the 14 fixed nodes are always present with stable ids
    for fixed in range(N_PROPRIO + N_FIXED_OUT):
        assert fixed in valid
    assert genome.node_count <= PARAMS.brain_max_nodes + 8


def test_random_blind_is_valid():
    rng = np.random.default_rng(0)
    innov = Innovations()
    for _ in range(50):
        genome = Genome.random_blind(rng, PARAMS, innov)
        _assert_valid(genome)
        assert len(genome.conns) >= 1


def test_mutation_preserves_invariants_over_many_generations():
    rng = np.random.default_rng(1)
    innov = Innovations()
    genome = Genome.random_blind(rng, PARAMS, innov)
    for _ in range(400):
        genome = mutate(genome, rng, innov, PARAMS)
        _assert_valid(genome)


def test_mutation_is_deterministic():
    base = Genome.random_blind(np.random.default_rng(3), PARAMS, Innovations())

    a = mutate(base, np.random.default_rng(7), Innovations(next_node=FIRST_DYNAMIC_NODE + 5), PARAMS)
    b = mutate(base, np.random.default_rng(7), Innovations(next_node=FIRST_DYNAMIC_NODE + 5), PARAMS)
    assert [n.id for n in a.nodes] == [n.id for n in b.nodes]
    assert [c.innov for c in a.conns] == [c.innov for c in b.conns]
    assert np.allclose([c.weight for c in a.conns], [c.weight for c in b.conns])


def test_innovation_numbers_are_reused_for_the_same_edge():
    innov = Innovations()
    first = innov.conn_innov(0, 10)
    assert innov.conn_innov(0, 10) == first
    assert innov.conn_innov(1, 10) != first
    node = innov.split_node(0, 10)
    assert innov.split_node(0, 10) == node


def test_copy_is_deep():
    genome = Genome.random_blind(np.random.default_rng(4), PARAMS, Innovations())
    clone = genome.copy()
    clone.physiology.size = 99.0
    clone.body_signature[0] = 42.0
    if clone.conns:
        clone.conns[0].weight = 123.0
    assert genome.physiology.size != 99.0
    assert genome.body_signature[0] != 42.0
    if genome.conns:
        assert genome.conns[0].weight != 123.0
