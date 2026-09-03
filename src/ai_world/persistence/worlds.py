"""World repository — all SQL touching the ``worlds`` table lives here."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone

from ai_world.config import MAX_MAP_DIM, MIN_MAP_DIM
from ai_world.persistence.entities import load_population, save_population
from ai_world.persistence.species import load_species, save_species
from ai_world.persistence.serialization import (
    deserialize_array,
    deserialize_eco_state,
    deserialize_grid,
    serialize_array,
    serialize_eco_state,
    serialize_grid,
)
from ai_world.world.generator import generate_grid
from ai_world.world.params import EcoParams
from ai_world.world.world import World, attach_ecosystem, ecosystem_fits, rebuild_ecosystem


@dataclass(frozen=True)
class WorldSummary:
    """Lightweight row for the world list (without the grid BLOB)."""

    id: int
    name: str
    width: int
    height: int
    seed: int
    tick: int
    created_at: datetime
    updated_at: datetime


def _parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _fmt_dt(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()


def _eco_blobs(world: World) -> tuple[bytes | None, bytes | None, bytes | None]:
    if not world.ecosystem_enabled:
        return None, None, None
    assert world.enzymes and world.spectrum and world.eco_params and world.species
    assert world.weather and world.eco_rng is not None and world.innovations is not None
    species_meta = {
        "threshold": world.species.threshold,
        "target": world.species.target_count,
        "next_id": world.species.next_id,
    }
    return (
        serialize_array(world.enzymes.values),
        serialize_array(world.spectrum.values),
        serialize_eco_state(
            world.eco_params, world.weather, world.eco_rng, world.innovations, species_meta
        ),
    )


def _restore_ecosystem(conn: sqlite3.Connection, world: World, row: sqlite3.Row) -> None:
    if row["eco_state"] is None:
        return
    params, weather, rng, innovations, species_meta = deserialize_eco_state(row["eco_state"])
    registry = load_species(
        conn, world.id, species_meta["threshold"], species_meta["target"],
        species_meta["next_id"],
    )
    rebuild_ecosystem(
        world,
        params,
        weather,
        rng,
        innovations,
        registry,
        deserialize_array(row["enzymes"]),
        deserialize_array(row["spectrum"]),
        load_population(conn, world.id, params),
    )


class WorldRepository:
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    # --- reads --------------------------------------------------------
    def list_worlds(self) -> list[WorldSummary]:
        rows = self._conn.execute(
            "SELECT id, name, width, height, seed, tick, created_at, updated_at "
            "FROM worlds ORDER BY updated_at DESC"
        ).fetchall()
        return [
            WorldSummary(
                id=r["id"],
                name=r["name"],
                width=r["width"],
                height=r["height"],
                seed=r["seed"],
                tick=r["tick"],
                created_at=_parse_dt(r["created_at"]),
                updated_at=_parse_dt(r["updated_at"]),
            )
            for r in rows
        ]

    def load(self, world_id: int) -> World:
        row = self._conn.execute(
            "SELECT * FROM worlds WHERE id = ?", (world_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"world {world_id} does not exist")
        grid = deserialize_grid(row["grid"], row["width"], row["height"])
        world = World(
            id=row["id"],
            name=row["name"],
            grid=grid,
            seed=row["seed"],
            tick=row["tick"],
            created_at=_parse_dt(row["created_at"]),
            updated_at=_parse_dt(row["updated_at"]),
        )
        _restore_ecosystem(self._conn, world, row)
        return world

    # --- writes ------------------------------------------------------
    def create(
        self, name: str, width: int, height: int, seed: int, *, ecosystem: bool = True
    ) -> World:
        name = name.strip() or "New world"
        width = _clamp_dim(width)
        height = _clamp_dim(height)
        seed = int(seed) & 0xFFFFFFFF

        grid = generate_grid(width, height, seed)
        world = World(name=name, grid=grid, seed=seed)
        if ecosystem and ecosystem_fits(width, height):
            attach_ecosystem(world, EcoParams())

        enzymes, spectrum, eco_state = _eco_blobs(world)
        cur = self._conn.execute(
            "INSERT INTO worlds "
            "(name, width, height, seed, tick, created_at, updated_at, grid, "
            " enzymes, spectrum, eco_state) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                world.name,
                width,
                height,
                seed,
                0,
                _fmt_dt(world.created_at),
                _fmt_dt(world.updated_at),
                serialize_grid(grid),
                enzymes,
                spectrum,
                eco_state,
            ),
        )
        world.id = int(cur.lastrowid)
        if world.population is not None:
            save_population(self._conn, world.id, world.population)
        if world.species is not None:
            save_species(self._conn, world.id, world.species)
        self._conn.commit()
        return world

    def save(self, world: World) -> None:
        if world.id is None:
            raise ValueError("world has no id — use create()")
        world.touch()
        enzymes, spectrum, eco_state = _eco_blobs(world)
        self._conn.execute(
            "UPDATE worlds SET name = ?, tick = ?, updated_at = ?, grid = ?, "
            "enzymes = ?, spectrum = ?, eco_state = ? WHERE id = ?",
            (
                world.name,
                world.tick,
                _fmt_dt(world.updated_at),
                serialize_grid(world.grid),
                enzymes,
                spectrum,
                eco_state,
                world.id,
            ),
        )
        if world.population is not None:
            save_population(self._conn, world.id, world.population)
        if world.species is not None:
            save_species(self._conn, world.id, world.species)
        self._conn.commit()

    def rename(self, world_id: int, name: str) -> None:
        self._conn.execute(
            "UPDATE worlds SET name = ?, updated_at = ? WHERE id = ?",
            (name.strip() or "World", _fmt_dt(datetime.now(timezone.utc)), world_id),
        )
        self._conn.commit()

    def delete(self, world_id: int) -> None:
        self._conn.execute("DELETE FROM worlds WHERE id = ?", (world_id,))
        self._conn.commit()


def _clamp_dim(value: int) -> int:
    return max(MIN_MAP_DIM, min(MAX_MAP_DIM, int(value)))
