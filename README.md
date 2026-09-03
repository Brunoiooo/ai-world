# ai-world

A 2D world sandbox: a tile grid of variable size (X/Y), full control over the
passage of time, and management of multiple saves in SQLite. At this stage the
world is terrain only — no entities. The infrastructure is ready for adding them.

## Running

```bash
python -m venv .venv
.venv\Scripts\activate             # Windows
pip install -e .                     # or: pip install pygame-ce numpy
python -m ai_world                   # or: python run.py
```

## Controls (inside a world)

| Key / mouse                | Action                                  |
|----------------------------|-----------------------------------------|
| `Space`                    | pause / resume time                     |
| `+` / `-`                  | faster / slower (0.2x … MAX)            |
| `0`                        | reset speed to 1x                       |
| `W` `A` `S` `D` / arrows   | pan the camera                          |
| mouse wheel                | zoom toward the cursor                  |
| middle / right button drag | pan the camera                          |
| `F`                        | fit the view to the whole map           |
| `F5` / `Ctrl+S`            | save the world                          |
| `Esc`                      | pause menu (save / exit)                |

The `MAX` speed is uncapped — the engine runs as many ticks as the machine can
fit into a ~12 ms per-frame budget (the rest goes to rendering). The `TPS`
counter in the top bar shows the ticks-per-second actually achieved.

## Architecture

```
src/ai_world/
├── config.py            map limits, database path (AI_WORLD_DATA_DIR)
├── world/               domain model — independent of pygame and SQLite
│   ├── tiles.py         tile types + palette
│   ├── grid.py          wrapper over a numpy uint8 array (cells[y, x])
│   ├── generator.py     deterministic terrain from value noise (seed → map)
│   └── world.py         aggregate: metadata + grid + tick
├── simulation/
│   ├── clock.py         time control: pause, speed ladder, accumulator
│   └── engine.py        a single step (systems = extension point)
├── persistence/
│   ├── database.py      SQLite connection + migrations (PRAGMA user_version)
│   ├── serialization.py grid ↔ BLOB (zlib)
│   └── worlds.py        WorldRepository: list / create / load / save / delete
└── ui/                  pygame layer only
    ├── camera.py        2D camera (position in tiles + zoom)
    ├── renderer.py      one base surface, scaled visible slice
    ├── widgets.py       button, text input
    └── screens/         state machine: menu → create / browser → world
```

Dependencies flow one way: `ui → simulation → world` and
`persistence → world`. The `world` and `simulation` layers do not know about
pygame, so the simulation can be run and tested without a window.

## Database

A single `worlds.db` file in the user data directory
(`%LOCALAPPDATA%\ai-world` on Windows; overridable via `AI_WORLD_DATA_DIR`).
The schema is versioned through `PRAGMA user_version` — new changes are
appended as the next entry in `_MIGRATIONS`.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

Coverage: the clock (pause, TPS accuracy, MAX mode, no "spiral of death"),
the generator (determinism, non-square dimensions), grid round-trip through a
BLOB, the repository (create/load/save/delete, list ordering, dimension clamp).

## Adding entities (later)

1. A model in `world/` (e.g. `entities.py`), stored on `World`.
2. A logic step as a `(world) -> None` function added to `Engine.systems`.
3. Extend serialization + a SQLite migration.
4. An entity drawing pass in `ui/renderer.py` on top of the tiles.
