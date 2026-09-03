"""Persistence layer: SQLite + world repository."""
from ai_world.persistence.database import connect, init_db
from ai_world.persistence.worlds import WorldRepository, WorldSummary

__all__ = ["connect", "init_db", "WorldRepository", "WorldSummary"]
