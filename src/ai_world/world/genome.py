"""Organism genome.

Phase 2 carries the *body*: physiology traits, the passive body signature (also
the render colour) and the mating-type vector. Phase 3 adds the evolvable
perception ports and the NEAT brain graph; Phase 4 adds crossover.

Every physiology trait is a plain float that both drives a mechanic and costs
energy in proportion to its magnitude — there is no free lunch for being big,
fast or tough.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from ai_world.world.params import EcoParams

MATING_TYPE_DIM = 3


@dataclass
class Physiology:
    max_speed: float          # ceiling on tiles/tick the brain can request
    size: float               # bigger = stronger + hungrier
    metabolic_efficiency: float  # enzyme -> energy conversion (0..1+)
    comfort_center: float     # preferred normalized temperature
    comfort_width: float      # half-width of the no-penalty band
    attack_power: float
    armor: float              # incoming-damage reduction
    hp_regen_rate: float      # hp/tick regained while sated (costly to maintain)
    mutation_rate: float      # self-adapting; scales all mutation magnitudes
    senescence_rate: float    # how fast upkeep climbs with age

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


def physiology_vector(ph: "Physiology") -> np.ndarray:
    return np.array([getattr(ph, name) for name in PHYS_FIELDS], dtype=np.float64)


def physiology_from_vector(values: np.ndarray) -> "Physiology":
    return Physiology(**{name: float(v) for name, v in zip(PHYS_FIELDS, values)})


# Bounds applied after every mutation, keyed by field name.
_PHYS_BOUNDS: dict[str, tuple[float, float]] = {
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


@dataclass
class Genome:
    physiology: Physiology
    body_signature: np.ndarray   # (C,) passive emission into the spectrum + colour
    mating_type: np.ndarray      # (MATING_TYPE_DIM,)

    @classmethod
    def random_blind(cls, rng: np.random.Generator, params: EcoParams) -> "Genome":
        """A minimal starting organism: a body, no evolved perception yet."""
        c = params.spectrum_channels
        return cls(
            physiology=Physiology.random(rng),
            body_signature=rng.random(c).astype(np.float32) * 0.6,
            mating_type=rng.normal(0.0, 1.0, MATING_TYPE_DIM).astype(np.float32),
        )

    def copy(self) -> "Genome":
        return Genome(
            physiology=replace(self.physiology),
            body_signature=self.body_signature.copy(),
            mating_type=self.mating_type.copy(),
        )


def mutate(genome: Genome, rng: np.random.Generator) -> Genome:
    """Return a mutated copy. Magnitude scales with the genome's own mutation_rate."""
    child = genome.copy()
    scale = genome.physiology.mutation_rate

    for name, (lo, hi) in _PHYS_BOUNDS.items():
        current = getattr(child.physiology, name)
        step = rng.normal(0.0, 0.06 * (hi - lo) * scale)
        setattr(child.physiology, name, float(np.clip(current + step, lo, hi)))

    child.body_signature = np.clip(
        child.body_signature + rng.normal(0.0, 0.05 * scale, child.body_signature.shape),
        0.0, 1.0,
    ).astype(np.float32)
    child.mating_type = (
        child.mating_type + rng.normal(0.0, 0.08 * scale, child.mating_type.shape)
    ).astype(np.float32)
    return child
