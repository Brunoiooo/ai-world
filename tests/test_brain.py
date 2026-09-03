import numpy as np
import torch

from ai_world.world.brain import Brain, BrainStore, compile_genome
from ai_world.world.genome import (
    ConnGene,
    Genome,
    Innovations,
    N_FIXED_OUT,
    NodeGene,
    Physiology,
    _fixed_nodes,
)
from ai_world.world.params import EcoParams

PARAMS = EcoParams()
G = PARAMS.brain_max_nodes


def _genomes(count: int, seed: int = 0) -> list[Genome]:
    rng = np.random.default_rng(seed)
    innov = Innovations()
    out = [Genome.random_blind(rng, PARAMS, innov) for _ in range(count)]
    # evolve them a bit so topologies differ
    return [_evolve(g, rng, innov) for g in out]


def _evolve(genome: Genome, rng, innov) -> Genome:
    from ai_world.world.genome import mutate

    for _ in range(int(rng.integers(0, 12))):
        genome = mutate(genome, rng, innov, PARAMS)
    return genome


def test_compile_shapes():
    genome = _genomes(1, seed=1)[0]
    cb = compile_genome(genome, G)
    assert cb.W.shape == (G, G)
    assert cb.act_ids.shape == (G,)
    assert cb.input_mask[:9].all()  # proprio always inputs
    assert cb.n_nodes >= 14


def test_brain_forward_shape_and_no_grad():
    brain = Brain(compile_genome(_genomes(1, seed=2)[0], G))
    state = brain.initial_state()
    inputs = torch.zeros(G)
    inputs[:9] = torch.rand(9)
    new_state, outputs = brain(state, inputs)
    assert outputs.shape == (N_FIXED_OUT,)
    assert torch.isfinite(outputs).all()
    assert not new_state.requires_grad


def test_brain_carries_recurrent_state():
    # bare genome: proprio+outputs, plus one hidden node (id 14) with a self-loop
    genome = Genome(
        physiology=Physiology.random(np.random.default_rng(0)),
        body_signature=np.zeros(PARAMS.spectrum_channels, np.float32),
        mating_type=np.zeros(3, np.float32),
        nodes=_fixed_nodes() + [NodeGene(14, "hidden", "tanh")],
        conns=[
            ConnGene(900, 8, 14, 0.6, True),    # bias -> hidden
            ConnGene(901, 14, 14, 0.9, True),   # hidden -> hidden (recurrent)
            ConnGene(902, 14, 9, 1.0, True),    # hidden -> turn output
        ],
    )
    brain = Brain(compile_genome(genome, G))
    state = brain.initial_state()
    inputs = torch.zeros(G)
    inputs[8] = 1.0  # bias

    states = [state.clone()]
    for _ in range(3):
        state, _ = brain(state, inputs)
        states.append(state.clone())
    # first step responds to bias only; later steps integrate the recurrent term
    assert not torch.allclose(states[1], states[2])


def test_brainstore_step_and_compact():
    store = BrainStore(PARAMS)
    store.append(_genomes(20, seed=3))
    assert len(store) == 20

    pop = _FakePop(20)
    spectrum = np.random.default_rng(0).random((PARAMS.spectrum_channels, 40, 40)).astype(np.float32)
    temp = np.full(20, 0.5)
    out = store.step(pop, spectrum, temp)
    assert out.shape == (20, N_FIXED_OUT)
    assert np.isfinite(out).all()

    keep = np.ones(20, dtype=bool)
    keep[[2, 7, 11]] = False
    fill = np.array([2, 7, 11])[np.array([2, 7, 11]) < 17]
    live = np.flatnonzero(keep)
    src = live[live >= 17]
    store.compact(fill, src, 17)
    assert len(store) == 17
    pop2 = _FakePop(17)
    out2 = store.step(pop2, spectrum, np.full(17, 0.5))
    assert out2.shape == (17, N_FIXED_OUT)


def test_brainstore_step_is_deterministic():
    a, b = BrainStore(PARAMS), BrainStore(PARAMS)
    genomes = _genomes(15, seed=9)
    a.append(genomes)
    b.append([g.copy() for g in genomes])
    pop = _FakePop(15)
    spectrum = np.zeros((PARAMS.spectrum_channels, 30, 30), dtype=np.float32)
    temp = np.full(15, 0.5)
    assert np.allclose(a.step(pop, spectrum, temp), b.step(pop, spectrum, temp))


class _FakePop:
    """Minimal stand-in for Population — only the columns BrainStore reads."""

    def __init__(self, n: int):
        rng = np.random.default_rng(n)
        self.energy = rng.random(n)
        self.hp = rng.random(n)
        self.age = rng.integers(0, 500, n)
        self.speed = rng.random(n) * 0.5
        self.last_turn = rng.normal(0, 0.3, n)
        self.x = rng.random(n) * 20 + 5
        self.y = rng.random(n) * 20 + 5
        self.heading = rng.random(n) * 6.28
        self.traits = np.abs(rng.normal(0.6, 0.2, (n, 10))) + 0.1
