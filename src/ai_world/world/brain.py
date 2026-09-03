"""Compiling NEAT genomes to torch and running the whole population's brains.

A genome's graph becomes a dense ``G x G`` weight matrix ``W`` (``G`` =
``brain_max_nodes``), zero-padded so every organism's matrix is the same shape
and the population can be evaluated with a single batched ``torch.bmm``. Each
row of the batch is still one organism's own set of weights -- inherited,
mutated and (Phase 4) crossed over independently.

Node layout inside a genome's matrix is fixed so slices are predictable:

    [0 .. 8]    proprioception inputs (+ bias at 8)
    [9 .. 13]   fixed outputs: turn, thrust, eat, attack, mate
    [14 ..]     IN-port nodes, then OUT-port nodes, then hidden nodes
    [.. G-1]    zero padding

The step is recurrent: ``state <- activate(W @ state)`` once per tick with the
input nodes clamped to the current sensor values, hidden/output activations
carried over between ticks.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ai_world.world.genome import (
    ACTIVATIONS,
    FIXED_OUTPUTS,
    N_FIXED_OUT,
    N_PROPRIO,
    Genome,
)

torch.set_num_threads(1)

_ACT_ID = {name: i for i, name in enumerate(ACTIVATIONS)}
FIXED_OUT_SLICE = slice(N_PROPRIO, N_PROPRIO + N_FIXED_OUT)  # 9..13
TURN, THRUST, EAT, ATTACK, MATE = range(N_FIXED_OUT)


def _activate(x: torch.Tensor, act_ids: torch.Tensor) -> torch.Tensor:
    """Per-node activation. Computes every function on the whole tensor and
    selects — cheap at these sizes and fully vectorised."""
    out = x.clone()                                   # identity (id 0)
    out = torch.where(act_ids == _ACT_ID["tanh"], torch.tanh(x), out)
    out = torch.where(act_ids == _ACT_ID["sigmoid"], torch.sigmoid(x), out)
    out = torch.where(act_ids == _ACT_ID["relu"], torch.relu(x), out)
    out = torch.where(act_ids == _ACT_ID["sin"], torch.sin(x), out)
    out = torch.where(act_ids == _ACT_ID["gauss"], torch.exp(-(x * x)), out)
    out = torch.where(act_ids == _ACT_ID["abs"], torch.abs(x), out)
    return out


@dataclass
class CompiledBrain:
    W: np.ndarray            # (G, G) float32
    act_ids: np.ndarray      # (G,)   int64
    input_mask: np.ndarray   # (G,)   bool  -- proprio + IN-port slots
    n_nodes: int
    n_conns: int
    port_cost: float
    in_port_slots: np.ndarray   # (k,) int64  positions of IN-port nodes
    out_port_slots: np.ndarray  # (m,) int64  positions of OUT-port nodes


class _PortColumns:
    """Accumulates flat port arrays across the population during a rebuild."""

    @dataclass
    class Frozen:
        owner: np.ndarray
        slot: np.ndarray
        signature: np.ndarray
        angle: np.ndarray
        reach: np.ndarray
        gain: np.ndarray

    def __init__(self, channels: int):
        self.channels = channels
        self.owner: list[int] = []
        self.slot: list[int] = []
        self.signature: list[np.ndarray] = []
        self.angle: list[float] = []
        self.reach: list[float] = []
        self.gain: list[float] = []

    def add(self, row: int, slot: int, port) -> None:
        self.owner.append(row)
        self.slot.append(slot)
        self.signature.append(port.signature)
        self.angle.append(port.angle)
        self.reach.append(port.reach)
        self.gain.append(port.gain)

    def finish(self) -> "Frozen":
        sig = (
            np.array(self.signature, dtype=np.float32)
            if self.signature else np.zeros((0, self.channels), dtype=np.float32)
        )
        return _PortColumns.Frozen(
            np.array(self.owner, dtype=np.intp),
            np.array(self.slot, dtype=np.intp),
            sig,
            np.array(self.angle, dtype=np.float64),
            np.array(self.reach, dtype=np.float64),
            np.array(self.gain, dtype=np.float64),
        )


def _node_order(genome: Genome) -> list[int]:
    in_ports = sorted(n.id for n in genome.nodes if n.kind == "in_port")
    out_ports = sorted(n.id for n in genome.nodes if n.kind == "out_port")
    hidden = sorted(n.id for n in genome.nodes if n.kind == "hidden")
    return list(range(N_PROPRIO + N_FIXED_OUT)) + in_ports + out_ports + hidden


def compile_genome(genome: Genome, g: int) -> CompiledBrain:
    order = _node_order(genome)
    n = len(order)
    if n > g:
        raise ValueError(f"genome has {n} nodes, brain_max_nodes is {g}")
    pos = {node_id: i for i, node_id in enumerate(order)}
    act_by_id = {node.id: node.activation for node in genome.nodes}

    weight = np.zeros((g, g), dtype=np.float32)
    n_conns = 0
    for conn in genome.conns:
        if conn.enabled and conn.src in pos and conn.dst in pos:
            weight[pos[conn.dst], pos[conn.src]] += conn.weight
            n_conns += 1

    act_ids = np.zeros(g, dtype=np.int64)
    for node_id, slot in pos.items():
        act_ids[slot] = _ACT_ID[act_by_id[node_id]]

    input_mask = np.zeros(g, dtype=bool)
    input_mask[:N_PROPRIO] = True
    in_slots = np.array([pos[p] for p in genome.in_port_node_ids()], dtype=np.int64)
    out_slots = np.array([pos[p] for p in genome.out_port_node_ids()], dtype=np.int64)
    input_mask[in_slots] = True
    return CompiledBrain(
        weight, act_ids, input_mask, n, n_conns, genome.port_cost(), in_slots, out_slots
    )


class Brain(torch.nn.Module):
    """One organism's own recurrent model — used for single-organism inference,
    inspection and tests. The population runs through :class:`BrainStore`."""

    def __init__(self, compiled: CompiledBrain):
        super().__init__()
        self.register_buffer("W", torch.from_numpy(compiled.W))
        self.register_buffer("act_ids", torch.from_numpy(compiled.act_ids))
        self.register_buffer("input_mask", torch.from_numpy(compiled.input_mask))
        self.n_nodes = compiled.n_nodes

    def initial_state(self) -> torch.Tensor:
        return torch.zeros(self.W.shape[0])

    @torch.inference_mode()
    def forward(self, state: torch.Tensor, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        clamped = torch.where(self.input_mask, inputs, state)
        nxt = _activate(self.W @ clamped, self.act_ids)
        nxt = torch.where(self.input_mask, inputs, nxt)
        return nxt, nxt[FIXED_OUT_SLICE]


_AGE_SCALE = 5000.0
_OSC_FREQ = 0.15


class BrainStore:
    """The whole population's brains as one padded batch, plus IN-port sensing.

    Rows are kept in step with :class:`ai_world.world.population.Population`:
    ``append`` / ``compact`` mirror births and deaths.
    """

    def __init__(self, params):
        self.g = params.brain_max_nodes
        self.channels = params.spectrum_channels
        self.samples = params.brain_sensor_samples
        self._n = 0
        self._cap = 0
        self.W = torch.zeros(0)
        self.act = torch.zeros(0, dtype=torch.int64)
        self.mask = torch.zeros(0, dtype=torch.bool)
        self.state = torch.zeros(0)
        self.n_nodes = np.zeros(0, dtype=np.int64)
        self.n_conns = np.zeros(0, dtype=np.int64)
        self.port_cost = np.zeros(0, dtype=np.float64)
        self._compiled: list[CompiledBrain] = []
        self._genomes: list[Genome] = []
        self._last_state: np.ndarray | None = None
        self.ports_in = _PortColumns(self.channels).finish()
        self.ports_out = _PortColumns(self.channels).finish()

    def __len__(self) -> int:
        return self._n

    # --- capacity ---------------------------------------------------
    def _ensure(self, size: int) -> None:
        if size <= self._cap:
            return
        cap = max(64, self._cap * 2, size)
        g = self.g

        def grow(src: torch.Tensor, shape, dtype):
            dst = torch.zeros(shape, dtype=dtype)
            if self._n:
                dst[: self._n] = src[: self._n]
            return dst

        self.W = grow(self.W, (cap, g, g), torch.float32)
        self.act = grow(self.act, (cap, g), torch.int64)
        self.mask = grow(self.mask, (cap, g), torch.bool)
        self.state = grow(self.state, (cap, g), torch.float32)
        self._cap = cap

    # --- population sync ------------------------------------------
    def append(self, genomes: list[Genome]) -> None:
        if not genomes:
            return
        compiled = [compile_genome(gen, self.g) for gen in genomes]
        start = self._n
        self._ensure(start + len(compiled))
        for k, cb in enumerate(compiled):
            row = start + k
            self.W[row] = torch.from_numpy(cb.W)
            self.act[row] = torch.from_numpy(cb.act_ids)
            self.mask[row] = torch.from_numpy(cb.input_mask)
            self.state[row].zero_()
        self._compiled.extend(compiled)
        self._genomes.extend(genomes)
        self._n += len(compiled)
        self._refresh_derived()
        self._rebuild_ports()

    def compact(self, fill: np.ndarray, src: np.ndarray, n_new: int) -> None:
        """Swap-remove: dead rows below the new boundary are overwritten by live
        rows from above it. Only ``len(fill)`` rows move, so the big weight
        tensor is never fully copied."""
        if len(fill):
            f, s = torch.from_numpy(fill), torch.from_numpy(src)
            self.W[f] = self.W[s].clone()
            self.act[f] = self.act[s].clone()
            self.mask[f] = self.mask[s].clone()
            self.state[f] = self.state[s].clone()
            for dst, source in zip(fill, src):
                self._compiled[dst] = self._compiled[source]
                self._genomes[dst] = self._genomes[source]
        del self._compiled[n_new:]
        del self._genomes[n_new:]
        self._n = n_new
        self._refresh_derived()
        self._rebuild_ports()

    def reset(self, genomes: list[Genome]) -> None:
        self._n = 0
        self._compiled = []
        self._genomes = []
        self.append(genomes)

    def _refresh_derived(self) -> None:
        self.n_nodes = np.array([c.n_nodes for c in self._compiled], dtype=np.int64)
        self.n_conns = np.array([c.n_conns for c in self._compiled], dtype=np.int64)
        self.port_cost = np.array([c.port_cost for c in self._compiled], dtype=np.float64)

    def _rebuild_ports(self) -> None:
        cols_in, cols_out = _PortColumns(self.channels), _PortColumns(self.channels)
        for row, (genome, cb) in enumerate(zip(self._genomes, self._compiled)):
            in_ports = [p for p in genome.ports if p.mode == "in"]
            out_ports = [p for p in genome.ports if p.mode == "out"]
            for port, slot in zip(in_ports, cb.in_port_slots):
                cols_in.add(row, int(slot), port)
            for port, slot in zip(out_ports, cb.out_port_slots):
                cols_out.add(row, int(slot), port)
        self.ports_in = cols_in.finish()
        self.ports_out = cols_out.finish()

    # test/introspection helper: how many IN ports across the population
    @property
    def p_owner(self) -> np.ndarray:
        return self.ports_in.owner

    # --- per-tick step -------------------------------------------
    def _proprio(self, pop, temp_here: np.ndarray, traits) -> np.ndarray:
        n = self._n
        out = np.zeros((n, N_PROPRIO), dtype=np.float32)
        out[:, 0] = pop.energy
        out[:, 1] = 1.0 - pop.energy
        out[:, 2] = pop.hp
        out[:, 3] = np.tanh(pop.age / _AGE_SCALE)
        out[:, 4] = pop.speed / np.maximum(traits[:, 0], 1e-3)
        out[:, 5] = pop.last_turn
        out[:, 6] = np.sin(pop.age * _OSC_FREQ)
        comfort_c = traits[:, 3]
        comfort_w = np.maximum(traits[:, 4], 1e-3)
        out[:, 7] = np.clip((temp_here - comfort_c) / comfort_w, -3.0, 3.0)
        out[:, 8] = 1.0
        return out

    def _sample_wedge(self, ports, pop):
        """(P, s) world positions + distances sampled along each port's centre ray."""
        frac = np.linspace(0.6, 1.0, self.samples)[None, :]
        dist = ports.reach[:, None] * frac
        theta = pop.heading[ports.owner][:, None] + ports.angle[:, None]
        sx = pop.x[ports.owner][:, None] + np.cos(theta) * dist
        sy = pop.y[ports.owner][:, None] + np.sin(theta) * dist
        return sx, sy, dist

    def _sense(self, pop, spectrum_values: np.ndarray) -> np.ndarray:
        contrib = np.zeros((self._n, self.g), dtype=np.float32)
        ports = self.ports_in
        if ports.owner.size == 0:
            return contrib
        _, h, w = spectrum_values.shape
        sx, sy, dist = self._sample_wedge(ports, pop)
        ix = np.clip(sx.astype(np.intp), 0, w - 1)
        iy = np.clip(sy.astype(np.intp), 0, h - 1)
        local = spectrum_values[:, iy, ix]                            # (C, P, s)
        sig = ports.signature / (
            np.linalg.norm(ports.signature, axis=1, keepdims=True) + 1e-6
        )
        dotv = np.einsum("cps,pc->ps", local, sig)                    # (P, s)
        intensity = np.tanh(ports.gain * (dotv / (1.0 + dist)).sum(axis=1))
        contrib[ports.owner, ports.slot] = intensity.astype(np.float32)
        return contrib

    @torch.inference_mode()
    def step(self, pop, spectrum_values: np.ndarray, temp_here: np.ndarray) -> np.ndarray:
        n = self._n
        if n == 0:
            return np.zeros((0, N_FIXED_OUT), dtype=np.float32)
        proprio = self._proprio(pop, temp_here, pop.traits)
        inputs_np = self._sense(pop, spectrum_values)
        inputs_np[:, :N_PROPRIO] = proprio

        mask = self.mask[:n]
        inputs = torch.from_numpy(inputs_np)
        state = torch.where(mask, inputs, self.state[:n])
        net = torch.bmm(self.W[:n], state.unsqueeze(-1)).squeeze(-1)
        nxt = _activate(net, self.act[:n])
        nxt = torch.where(mask, inputs, nxt)
        self.state[:n] = nxt
        self._last_state = nxt.numpy()
        return nxt[:, FIXED_OUT_SLICE].numpy().copy()

    def emit(self, pop, spectrum_values: np.ndarray) -> np.ndarray:
        """Splat each OUT port's ``signature * output`` into the spectrum field
        along its wedge. Returns the per-organism emission energy cost."""
        cost = np.zeros(self._n, dtype=np.float64)
        ports = self.ports_out
        if ports.owner.size == 0 or self._last_state is None:
            return cost
        _, h, w = spectrum_values.shape
        values = self._last_state[ports.owner, ports.slot]            # (P,)
        sx, sy, _ = self._sample_wedge(ports, pop)
        cx = np.clip(sx[:, self.samples // 2].astype(np.intp), 0, w - 1)
        cy = np.clip(sy[:, self.samples // 2].astype(np.intp), 0, h - 1)
        emission = (
            values[:, None] * ports.gain[:, None] * ports.signature
        ).astype(np.float32)                                          # (P, C)
        for channel in range(self.channels):
            np.add.at(spectrum_values[channel], (cy, cx), emission[:, channel])
        strength = np.abs(values) * ports.gain * ports.reach * ports.reach
        np.add.at(cost, ports.owner, strength)
        return cost
