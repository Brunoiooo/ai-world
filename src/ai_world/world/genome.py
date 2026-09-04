"""Organism genome: body traits + evolvable perception ports + a NEAT brain graph.

The genome has three evolving parts:

* **physiology** -- scalar body traits, each of which both drives a mechanic and
  costs energy in proportion to its magnitude;
* **ports** -- genetically encoded senses/emitters ``{signature, mode, angle,
  arc, reach, gain}``; an IN port feeds one scalar to the brain, an OUT port
  radiates ``signature * value`` into the world (OUT wired in Phase 4);
* **brain** -- an arbitrary-topology, possibly recurrent graph of nodes and
  connections in the style of NEAT, compiled to a dense weight matrix at runtime
  (see :mod:`ai_world.world.brain`).

Historical markings for connections and node splits come from a per-world
:class:`Innovations` registry so sexual crossover (Phase 4) can align genes.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from ai_world.world.params import EcoParams

MATING_TYPE_DIM = 3

# --- fixed brain nodes (stable ids across every genome) -------------------
PROPRIO_INPUTS = (
    "energy", "hunger", "hp", "age", "speed", "last_turn", "oscillator",
    "thermal", "bias",
)
FIXED_OUTPUTS = ("turn", "thrust", "eat", "attack", "mate")
N_PROPRIO = len(PROPRIO_INPUTS)          # 9  -> node ids 0..8
N_FIXED_OUT = len(FIXED_OUTPUTS)         # 5  -> node ids 9..13
FIRST_DYNAMIC_NODE = N_PROPRIO + N_FIXED_OUT  # 14

ACTIVATIONS = ("identity", "tanh", "sigmoid", "relu", "sin", "gauss", "abs")
_HIDDEN_ACTIVATIONS = ("tanh", "sigmoid", "relu", "sin", "gauss", "abs")


# ---------------------------------------------------------------------------
# physiology
# ---------------------------------------------------------------------------
@dataclass
class Physiology:
    max_speed: float
    size: float
    metabolic_efficiency: float
    comfort_center: float
    comfort_width: float
    attack_power: float
    armor: float
    hp_regen_rate: float
    mutation_rate: float
    senescence_rate: float

    @classmethod
    def random(cls, rng: np.random.Generator) -> "Physiology":
        def jitter(mean: float, spread: float, lo: float, hi: float) -> float:
            return float(np.clip(rng.normal(mean, spread), lo, hi))

        return cls(
            max_speed=jitter(0.6, 0.2, 0.05, 2.0),
            size=jitter(1.0, 0.25, 0.3, 3.0),
            metabolic_efficiency=jitter(0.8, 0.15, 0.2, 1.5),
            comfort_center=jitter(0.5, 0.15, 0.05, 0.95),
            comfort_width=jitter(0.25, 0.1, 0.05, 0.6),
            attack_power=jitter(0.2, 0.15, 0.0, 1.0),
            armor=jitter(0.2, 0.15, 0.0, 1.0),
            hp_regen_rate=jitter(0.01, 0.005, 0.0, 0.05),
            mutation_rate=jitter(0.5, 0.15, 0.05, 2.0),
            senescence_rate=jitter(0.5, 0.2, 0.0, 2.0),
        )


PHYS_FIELDS: tuple[str, ...] = (
    "max_speed", "size", "metabolic_efficiency", "comfort_center", "comfort_width",
    "attack_power", "armor", "hp_regen_rate", "mutation_rate", "senescence_rate",
)

PHYS_BOUNDS: dict[str, tuple[float, float]] = {
    "max_speed": (0.05, 3.0),
    "size": (0.3, 4.0),
    "metabolic_efficiency": (0.2, 2.0),
    "comfort_center": (0.0, 1.0),
    "comfort_width": (0.03, 0.7),
    "attack_power": (0.0, 2.0),
    "armor": (0.0, 1.0),
    "hp_regen_rate": (0.0, 0.08),
    "mutation_rate": (0.02, 3.0),
    "senescence_rate": (0.0, 3.0),
}


def physiology_vector(ph: Physiology) -> np.ndarray:
    return np.array([getattr(ph, name) for name in PHYS_FIELDS], dtype=np.float64)


def physiology_from_vector(values: np.ndarray) -> Physiology:
    return Physiology(**{name: float(v) for name, v in zip(PHYS_FIELDS, values)})


# ---------------------------------------------------------------------------
# brain graph + ports
# ---------------------------------------------------------------------------
@dataclass
class NodeGene:
    id: int
    kind: str        # 'proprio' | 'fixed_out' | 'in_port' | 'out_port' | 'hidden'
    activation: str


@dataclass
class ConnGene:
    innov: int
    src: int
    dst: int
    weight: float
    enabled: bool = True


@dataclass
class PortGene:
    node_id: int
    mode: str          # 'in' | 'out'
    signature: np.ndarray  # (C,)
    angle: float
    arc: float
    reach: float
    gain: float

    def copy(self) -> "PortGene":
        return PortGene(
            self.node_id, self.mode, self.signature.copy(),
            self.angle, self.arc, self.reach, self.gain,
        )


@dataclass
class Innovations:
    """Per-world historical markings, so crossover can line genes up by ancestry."""

    conn: dict[tuple[int, int], int] = field(default_factory=dict)
    node_split: dict[tuple[int, int], int] = field(default_factory=dict)
    next_conn: int = 0
    next_node: int = FIRST_DYNAMIC_NODE

    def conn_innov(self, src: int, dst: int) -> int:
        key = (src, dst)
        if key not in self.conn:
            self.conn[key] = self.next_conn
            self.next_conn += 1
        return self.conn[key]

    def split_node(self, src: int, dst: int) -> int:
        key = (src, dst)
        if key not in self.node_split:
            self.node_split[key] = self.next_node
            self.next_node += 1
        return self.node_split[key]

    def fresh_node(self) -> int:
        value = self.next_node
        self.next_node += 1
        return value


# ---------------------------------------------------------------------------
# genome
# ---------------------------------------------------------------------------
@dataclass
class Genome:
    physiology: Physiology
    body_signature: np.ndarray
    mating_type: np.ndarray
    nodes: list[NodeGene] = field(default_factory=list)
    conns: list[ConnGene] = field(default_factory=list)
    ports: list[PortGene] = field(default_factory=list)

    # --- derived counts (used for brain-complexity upkeep) -------------
    @property
    def node_count(self) -> int:
        return N_PROPRIO + N_FIXED_OUT + len(self.ports) + self._hidden_count()

    @property
    def enabled_conn_count(self) -> int:
        return sum(1 for c in self.conns if c.enabled)

    def _hidden_count(self) -> int:
        return sum(1 for n in self.nodes if n.kind == "hidden")

    def port_cost(self) -> float:
        return sum(p.gain * p.reach * p.reach / max(p.arc, 0.15) for p in self.ports)

    def in_port_node_ids(self) -> list[int]:
        return [p.node_id for p in self.ports if p.mode == "in"]

    def out_port_node_ids(self) -> list[int]:
        return [p.node_id for p in self.ports if p.mode == "out"]

    def copy(self) -> "Genome":
        return Genome(
            physiology=replace(self.physiology),
            body_signature=self.body_signature.copy(),
            mating_type=self.mating_type.copy(),
            nodes=[replace(n) for n in self.nodes],
            conns=[replace(c) for c in self.conns],
            ports=[p.copy() for p in self.ports],
        )

    @classmethod
    def random_blind(
        cls, rng: np.random.Generator, params: EcoParams, innov: Innovations
    ) -> "Genome":
        """A minimal starting organism: a body, a wired-but-uninformed brain,
        and at most one random port. Perception has to evolve from here.

        The brain also gets two guaranteed seed connections ``bias -> eat`` and
        ``bias -> mate`` with random (possibly negative) weights. These are not
        reflexes -- feeding and mating fire *only* from the brain -- they just
        give a blind organism a non-zero chance of trying either, so evolution
        has something to select on. Mutation can strengthen, invert, rewire or
        bury them under learned control like any other gene."""
        c = params.spectrum_channels
        nodes = _fixed_nodes()
        ports: list[PortGene] = []
        genome = cls(
            physiology=Physiology.random(rng),
            body_signature=(rng.random(c) * 0.6).astype(np.float32),
            mating_type=rng.normal(0.0, 1.0, MATING_TYPE_DIM).astype(np.float32),
            nodes=nodes,
            conns=[],
            ports=ports,
        )
        if rng.random() < 0.5:
            _add_port(genome, rng, params, innov, mode="in" if rng.random() < 0.6 else "out")

        # a few random connections from any input to any fixed output
        inputs = list(range(N_PROPRIO)) + genome.in_port_node_ids()
        outputs = list(range(N_PROPRIO, N_PROPRIO + N_FIXED_OUT))
        for _ in range(int(rng.integers(1, 4))):
            src = int(rng.choice(inputs))
            dst = int(rng.choice(outputs))
            _connect(genome, src, dst, float(rng.normal(0.0, 1.2)), innov)

        bias = PROPRIO_INPUTS.index("bias")
        for name in ("eat", "mate"):
            # mildly positive mean so most seed organisms at least try; still
            # often negative, and free for mutation to flip either way.
            _connect(genome, bias, N_PROPRIO + FIXED_OUTPUTS.index(name),
                     float(rng.normal(0.5, 1.0)), innov)
        return genome


# ---------------------------------------------------------------------------
# construction helpers
# ---------------------------------------------------------------------------
def _fixed_nodes() -> list[NodeGene]:
    nodes = [NodeGene(i, "proprio", "identity") for i in range(N_PROPRIO)]
    defaults = ("tanh", "sigmoid", "sigmoid", "sigmoid", "sigmoid")
    nodes += [
        NodeGene(N_PROPRIO + i, "fixed_out", defaults[i]) for i in range(N_FIXED_OUT)
    ]
    return nodes


def _connect(
    genome: Genome, src: int, dst: int, weight: float, innov: Innovations
) -> None:
    if any(c.src == src and c.dst == dst for c in genome.conns):
        return
    genome.conns.append(ConnGene(innov.conn_innov(src, dst), src, dst, weight, True))


def _random_signature(rng: np.random.Generator, channels: int) -> np.ndarray:
    return rng.normal(0.0, 1.0, channels).astype(np.float32)


def _add_port(
    genome: Genome,
    rng: np.random.Generator,
    params: EcoParams,
    innov: Innovations,
    *,
    mode: str,
) -> PortGene:
    node_id = innov.fresh_node()
    genome.nodes.append(
        NodeGene(node_id, "in_port" if mode == "in" else "out_port",
                 "identity" if mode == "in" else "tanh")
    )
    port = PortGene(
        node_id=node_id,
        mode=mode,
        signature=_random_signature(rng, params.spectrum_channels),
        angle=float(rng.uniform(-np.pi, np.pi)),
        arc=float(rng.uniform(0.3, 2.0)),
        reach=float(rng.uniform(2.0, 12.0)),
        gain=float(rng.uniform(0.5, 2.0)),
    )
    genome.ports.append(port)
    return port


# ---------------------------------------------------------------------------
# mutation
# ---------------------------------------------------------------------------
def mutate(
    genome: Genome,
    rng: np.random.Generator,
    innov: Innovations,
    params: EcoParams,
) -> Genome:
    child = genome.copy()
    scale = genome.physiology.mutation_rate

    _mutate_physiology(child, rng, scale)
    _mutate_signatures(child, rng, scale)
    _mutate_weights(child, rng, scale)

    if rng.random() < 0.16 * scale:
        _mutate_add_connection(child, rng, innov)
    if rng.random() < 0.06 * scale and child.node_count < params.brain_max_nodes:
        _mutate_add_node(child, rng, innov)
    if rng.random() < 0.05 * scale:
        _mutate_toggle_connection(child, rng)
    if rng.random() < 0.05 * scale and child.node_count < params.brain_max_nodes:
        _add_port(child, rng, params, innov,
                  mode="in" if rng.random() < 0.6 else "out")
        _wire_new_port(child, rng, innov)
    if rng.random() < 0.04 * scale and child.ports:
        _mutate_remove_port(child, rng)
    if rng.random() < 0.15 * scale and child.ports:
        _mutate_port_params(child, rng, scale)
    if rng.random() < 0.03 * scale and child.ports:
        _mutate_flip_port(child, rng)
    if rng.random() < 0.05 * scale:
        _mutate_activation(child, rng)
    return child


def _mutate_physiology(child: Genome, rng: np.random.Generator, scale: float) -> None:
    for name, (lo, hi) in PHYS_BOUNDS.items():
        current = getattr(child.physiology, name)
        step = rng.normal(0.0, 0.06 * (hi - lo) * scale)
        setattr(child.physiology, name, float(np.clip(current + step, lo, hi)))


def _mutate_signatures(child: Genome, rng: np.random.Generator, scale: float) -> None:
    child.body_signature = np.clip(
        child.body_signature + rng.normal(0.0, 0.05 * scale, child.body_signature.shape),
        0.0, 1.0,
    ).astype(np.float32)
    child.mating_type = (
        child.mating_type + rng.normal(0.0, 0.08 * scale, child.mating_type.shape)
    ).astype(np.float32)


def _mutate_weights(child: Genome, rng: np.random.Generator, scale: float) -> None:
    for conn in child.conns:
        if rng.random() < 0.8:
            conn.weight += float(rng.normal(0.0, 0.15 * scale))
        elif rng.random() < 0.1:
            conn.weight = float(rng.normal(0.0, 1.0))


def _candidate_nodes(child: Genome) -> tuple[list[int], list[int]]:
    ids = [n.id for n in child.nodes]
    sources = [i for i, n in zip(ids, child.nodes) if n.kind != "fixed_out"]
    targets = [i for i, n in zip(ids, child.nodes)
               if n.kind not in ("proprio", "in_port")]
    return sources, targets


def _mutate_add_connection(child: Genome, rng: np.random.Generator, innov: Innovations) -> None:
    sources, targets = _candidate_nodes(child)
    if not sources or not targets:
        return
    src, dst = int(rng.choice(sources)), int(rng.choice(targets))
    _connect(child, src, dst, float(rng.normal(0.0, 1.0)), innov)


def _mutate_add_node(child: Genome, rng: np.random.Generator, innov: Innovations) -> None:
    enabled = [c for c in child.conns if c.enabled]
    if not enabled:
        return
    conn = enabled[int(rng.integers(len(enabled)))]
    conn.enabled = False
    new_id = innov.split_node(conn.src, conn.dst)
    if any(n.id == new_id for n in child.nodes):
        return
    child.nodes.append(NodeGene(new_id, "hidden", str(rng.choice(_HIDDEN_ACTIVATIONS))))
    child.conns.append(ConnGene(innov.conn_innov(conn.src, new_id), conn.src, new_id, 1.0, True))
    child.conns.append(
        ConnGene(innov.conn_innov(new_id, conn.dst), new_id, conn.dst, conn.weight, True)
    )


def _mutate_toggle_connection(child: Genome, rng: np.random.Generator) -> None:
    if child.conns:
        conn = child.conns[int(rng.integers(len(child.conns)))]
        conn.enabled = not conn.enabled


def _wire_new_port(child: Genome, rng: np.random.Generator, innov: Innovations) -> None:
    port = child.ports[-1]
    if port.mode == "in":
        dst_pool = [n.id for n in child.nodes if n.kind not in ("proprio", "in_port")]
        if dst_pool:
            _connect(child, port.node_id, int(rng.choice(dst_pool)),
                     float(rng.normal(0.0, 1.0)), innov)
    else:
        src_pool = [n.id for n in child.nodes if n.kind != "fixed_out" and n.id != port.node_id]
        if src_pool:
            _connect(child, int(rng.choice(src_pool)), port.node_id,
                     float(rng.normal(0.0, 1.0)), innov)


def _mutate_remove_port(child: Genome, rng: np.random.Generator) -> None:
    port = child.ports.pop(int(rng.integers(len(child.ports))))
    child.nodes = [n for n in child.nodes if n.id != port.node_id]
    child.conns = [c for c in child.conns
                   if c.src != port.node_id and c.dst != port.node_id]


def _mutate_port_params(child: Genome, rng: np.random.Generator, scale: float) -> None:
    port = child.ports[int(rng.integers(len(child.ports)))]
    port.signature = (
        port.signature + rng.normal(0.0, 0.15 * scale, port.signature.shape)
    ).astype(np.float32)
    port.angle = float((port.angle + rng.normal(0.0, 0.3 * scale) + np.pi) % (2 * np.pi) - np.pi)
    port.arc = float(np.clip(port.arc + rng.normal(0.0, 0.2 * scale), 0.15, np.pi))
    port.reach = float(np.clip(port.reach + rng.normal(0.0, 1.0 * scale), 1.0, 24.0))
    port.gain = float(np.clip(port.gain + rng.normal(0.0, 0.2 * scale), 0.1, 4.0))


def _mutate_flip_port(child: Genome, rng: np.random.Generator) -> None:
    port = child.ports[int(rng.integers(len(child.ports)))]
    port.mode = "out" if port.mode == "in" else "in"
    for node in child.nodes:
        if node.id == port.node_id:
            node.kind = "in_port" if port.mode == "in" else "out_port"
            node.activation = "identity" if port.mode == "in" else "tanh"


def _mutate_activation(child: Genome, rng: np.random.Generator) -> None:
    mutable = [n for n in child.nodes if n.kind in ("hidden", "fixed_out", "out_port")]
    if mutable:
        mutable[int(rng.integers(len(mutable)))].activation = str(
            rng.choice(_HIDDEN_ACTIVATIONS)
        )


# ---------------------------------------------------------------------------
# sexual reproduction
# ---------------------------------------------------------------------------
COMPAT_COEFFS = (1.0, 1.0, 0.4, 0.6, 0.3)  # excess, disjoint, weight, port-sig, physiology


def crossover(
    a: Genome, b: Genome, rng: np.random.Generator, *, a_is_fitter: bool
) -> Genome:
    """NEAT crossover: matching genes come from either parent at random, genes
    only one parent has come from the fitter parent."""
    fit, other = (a, b) if a_is_fitter else (b, a)

    conns_fit = {c.innov: c for c in fit.conns}
    conns_other = {c.innov: c for c in other.conns}
    child_conns: list[ConnGene] = []
    for innov, cf in conns_fit.items():
        co = conns_other.get(innov)
        pick = cf if (co is None or rng.random() < 0.5) else co
        enabled = pick.enabled if (co is None) else (
            True if rng.random() < 0.75 else (cf.enabled and co.enabled)
        )
        child_conns.append(ConnGene(innov, pick.src, pick.dst, pick.weight, enabled))

    needed = {c.src for c in child_conns} | {c.dst for c in child_conns}
    nodes_fit = {n.id: n for n in fit.nodes}
    nodes_other = {n.id: n for n in other.nodes}
    child_nodes = [replace(nodes_fit[i]) for i in range(N_PROPRIO + N_FIXED_OUT)]
    for node_id in sorted(needed):
        if node_id < N_PROPRIO + N_FIXED_OUT:
            continue
        source = nodes_fit.get(node_id) or nodes_other.get(node_id)
        if source is not None:
            child_nodes.append(replace(source))

    child_node_ids = {n.id for n in child_nodes}
    ports_other = {p.node_id: p for p in other.ports}
    child_ports: list[PortGene] = []
    for port in fit.ports:
        if port.node_id not in child_node_ids:
            continue
        mate = ports_other.get(port.node_id)
        if mate is None or rng.random() < 0.5:
            child_ports.append(port.copy())
        else:
            child_ports.append(
                PortGene(
                    port.node_id, port.mode,
                    0.5 * (port.signature + mate.signature),
                    _circ_mean(port.angle, mate.angle),
                    0.5 * (port.arc + mate.arc),
                    0.5 * (port.reach + mate.reach),
                    0.5 * (port.gain + mate.gain),
                )
            )

    phys = Physiology(**{
        name: float(getattr(a.physiology, name) if rng.random() < 0.5
                    else getattr(b.physiology, name))
        for name in PHYS_FIELDS
    })
    return Genome(
        physiology=phys,
        body_signature=(0.5 * (a.body_signature + b.body_signature)).astype(np.float32),
        mating_type=(0.5 * (a.mating_type + b.mating_type)).astype(np.float32),
        nodes=child_nodes,
        conns=child_conns,
        ports=child_ports,
    )


def _circ_mean(x: float, y: float) -> float:
    return float(np.arctan2(
        0.5 * (np.sin(x) + np.sin(y)), 0.5 * (np.cos(x) + np.cos(y))
    ))


def compat_distance(a: Genome, b: Genome, coeffs: tuple = COMPAT_COEFFS) -> float:
    c_excess, c_disjoint, c_weight, c_port, c_phys = coeffs
    ia = {c.innov: c for c in a.conns}
    ib = {c.innov: c for c in b.conns}
    if ia or ib:
        max_a = max(ia) if ia else -1
        max_b = max(ib) if ib else -1
        boundary = min(max_a, max_b)
        matching = ia.keys() & ib.keys()
        weight_diff = (
            np.mean([abs(ia[k].weight - ib[k].weight) for k in matching])
            if matching else 0.0
        )
        only = (ia.keys() ^ ib.keys())
        disjoint = sum(1 for k in only if k <= boundary)
        excess = len(only) - disjoint
        norm = max(len(ia), len(ib), 1)
    else:
        weight_diff = disjoint = excess = 0.0
        norm = 1

    port_diff = _port_signature_diff(a, b)
    phys_diff = float(np.linalg.norm(
        physiology_vector(a.physiology) - physiology_vector(b.physiology)
    ))
    return (
        c_excess * excess / norm
        + c_disjoint * disjoint / norm
        + c_weight * weight_diff
        + c_port * port_diff
        + c_phys * phys_diff
    )


def _port_signature_diff(a: Genome, b: Genome) -> float:
    if not a.ports or not b.ports:
        return float(abs(len(a.ports) - len(b.ports)))
    sa = np.mean([p.signature for p in a.ports], axis=0)
    sb = np.mean([p.signature for p in b.ports], axis=0)
    return float(np.linalg.norm(sa - sb))
