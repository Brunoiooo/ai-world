"""Row <-> :class:`Entity` mapping for the ``entities`` table."""
from __future__ import annotations

import sqlite3

from ai_world.persistence.serialization import deserialize_genome, serialize_genome
from ai_world.world.entity import Entity
from ai_world.world.population import Population

_COLUMNS = (
    "world_id", "id", "x", "y", "heading", "energy", "hp", "age", "birth_tick",
    "generation", "species_id", "parent_a", "parent_b", "genome", "brain_state",
)
INSERT_SQL = (
    f"INSERT INTO entities ({', '.join(_COLUMNS)}) "
    f"VALUES ({', '.join('?' * len(_COLUMNS))})"
)


def entity_to_row(world_id: int, entity: Entity) -> tuple:
    return (
        world_id,
        entity.id,
        entity.x,
        entity.y,
        entity.heading,
        entity.energy,
        entity.hp,
        entity.age,
        entity.birth_tick,
        entity.generation,
        entity.species_id,
        entity.parent_a,
        entity.parent_b,
        serialize_genome(entity.genome),
        None,  # brain_state — Phase 3
    )


def row_to_entity(row: sqlite3.Row) -> Entity:
    return Entity(
        id=row["id"],
        x=row["x"],
        y=row["y"],
        heading=row["heading"],
        energy=row["energy"],
        hp=row["hp"],
        genome=deserialize_genome(row["genome"]),
        age=row["age"],
        birth_tick=row["birth_tick"],
        generation=row["generation"],
        species_id=row["species_id"],
        parent_a=row["parent_a"],
        parent_b=row["parent_b"],
    )


def load_population(conn: sqlite3.Connection, world_id: int) -> Population:
    rows = conn.execute(
        "SELECT * FROM entities WHERE world_id = ? ORDER BY id", (world_id,)
    ).fetchall()
    return Population([row_to_entity(r) for r in rows])


def save_population(conn: sqlite3.Connection, world_id: int, population: Population) -> None:
    conn.execute("DELETE FROM entities WHERE world_id = ?", (world_id,))
    conn.executemany(
        INSERT_SQL, [entity_to_row(world_id, e) for e in population.snapshots()]
    )
