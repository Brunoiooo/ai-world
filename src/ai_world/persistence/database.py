"""SQLite connection + schema migrations based on ``PRAGMA user_version``."""
from __future__ import annotations

import sqlite3
from pathlib import Path

# Each list item is the next schema version. Never edit existing entries —
# append new ones at the end.
_MIGRATIONS: list[str] = [
    # v1
    """
    CREATE TABLE worlds (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        name       TEXT    NOT NULL,
        width      INTEGER NOT NULL,
        height     INTEGER NOT NULL,
        seed       INTEGER NOT NULL,
        tick       INTEGER NOT NULL DEFAULT 0,
        created_at TEXT    NOT NULL,
        updated_at TEXT    NOT NULL,
        grid       BLOB    NOT NULL
    );
    """,
    # v2 -- ecosystem field substrate (nullable: worlds may be terrain-only)
    """
    ALTER TABLE worlds ADD COLUMN enzymes   BLOB;
    ALTER TABLE worlds ADD COLUMN spectrum  BLOB;
    ALTER TABLE worlds ADD COLUMN eco_state BLOB;
    """,
]


def connect(path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    for index in range(version, len(_MIGRATIONS)):
        conn.executescript(_MIGRATIONS[index])
        conn.execute(f"PRAGMA user_version = {index + 1}")
    conn.commit()
