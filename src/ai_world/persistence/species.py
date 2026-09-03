"""Persistence for the species registry and its census history."""
from __future__ import annotations

import sqlite3

from ai_world.persistence.serialization import deserialize_genome, serialize_genome
from ai_world.world.species import Species, SpeciesRegistry


def save_species(conn: sqlite3.Connection, world_id: int, registry: SpeciesRegistry) -> None:
    conn.execute("DELETE FROM species WHERE world_id = ?", (world_id,))
    conn.executemany(
        "INSERT INTO species (world_id, id, first_tick, parent_id, representative) "
        "VALUES (?, ?, ?, ?, ?)",
        [
            (world_id, sp.id, sp.first_tick, sp.parent_id,
             serialize_genome(sp.representative))
            for sp in registry.species.values()
        ],
    )
    conn.execute("DELETE FROM species_census WHERE world_id = ?", (world_id,))
    conn.executemany(
        "INSERT INTO species_census (world_id, tick, species_id, count) VALUES (?, ?, ?, ?)",
        [
            (world_id, tick, sid, count)
            for tick, counts in registry.census
            for sid, count in counts.items()
        ],
    )


def load_species(
    conn: sqlite3.Connection, world_id: int, threshold: float, target: int, next_id: int
) -> SpeciesRegistry:
    registry = SpeciesRegistry(threshold=threshold, target_count=target, next_id=next_id)
    for row in conn.execute(
        "SELECT id, first_tick, parent_id, representative FROM species WHERE world_id = ?",
        (world_id,),
    ):
        registry.species[row["id"]] = Species(
            id=row["id"],
            representative=deserialize_genome(row["representative"]),
            first_tick=row["first_tick"],
            parent_id=row["parent_id"],
        )

    grouped: dict[int, dict[int, int]] = {}
    for row in conn.execute(
        "SELECT tick, species_id, count FROM species_census WHERE world_id = ? ORDER BY tick",
        (world_id,),
    ):
        grouped.setdefault(row["tick"], {})[row["species_id"]] = row["count"]
    registry.census = [(tick, counts) for tick, counts in grouped.items()]
    return registry
