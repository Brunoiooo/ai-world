# ai-world

A 2D tile-world sandbox with an **artificial-life ecosystem**: organisms whose
body, senses and brain all evolve through continuous, sexual neuroevolution —
no fitness function, no training loop. You set the rules of the world; foraging,
navigation, communication, predation and speciation are left to emerge.

## Running

```bash
python -m venv .venv
.venv\Scripts\activate                 # Windows
pip install -e .                         # pulls pygame-ce, numpy, torch (CPU, ~200 MB)
python -m ai_world                       # or: python run.py
```

## Controls (inside a world)

| Key / mouse                | Action                                    |
|----------------------------|-------------------------------------------|
| `Space`                    | pause / resume time                       |
| `+` / `-` / `0`            | faster / slower / reset speed             |
| `W` `A` `S` `D` / arrows   | pan the camera                            |
| mouse wheel                | zoom toward the cursor                    |
| middle / right drag        | pan the camera                            |
| `F`                        | fit the view to the whole map             |
| left click                 | inspect the organism under the cursor     |
| `H`                        | show / hide organisms                     |
| `E` `T` `1`–`6`            | overlay the enzyme / temperature / signal-channel field |
| `Tab`                      | species panel (count + share bar)         |
| `F5` / `Ctrl+S`            | save · `Esc` pause menu                    |

Time runs on a speed ladder up to an uncapped `MAX`; the `TPS` counter shows the
ticks-per-second actually achieved. The ecosystem runs only for maps up to
512×512 (a new world defaults to 144×144); larger maps are terrain-only.

## The ecosystem

**Fields.** Three continuous layers sit over the tiles:

* *enzymes* — the food resource. Regrows on fertile terrain (grass, forest),
  is eaten locally and decays, so it forms shifting patches. Corpses deposit
  enzymes back onto the tile.
* *temperature* — latitude + altitude + terrain, modulated by a seasonal cycle,
  drifting weather fronts, and rare climate events (cold snaps, heat waves,
  blooms, droughts).
* *spectrum* — six decaying, diffusing signal channels. The enzyme
  concentration and temperature occupy two of them; organisms radiate a body
  signature into the rest and can actively emit on them.

**Organisms.** Each has `energy` and `hp` in `[0, 1]` (`hp` falls when `energy`
hits zero, recovers while well fed), plus a genome of:

* *physiology* — `max_speed`, `size`, metabolic efficiency, a temperature
  comfort band, attack / armor, regeneration, mutation rate, senescence. Every
  trait both does something and **costs energy in proportion to its magnitude** —
  as does every brain node, every connection and every sensory port.
* *ports* — genetically encoded senses/emitters `{signature, IN/OUT, angle,
  arc, reach, gain}`. An IN port reports how strongly the world's local signal
  matches its signature along its wedge; an OUT port radiates
  `signature × output` back into the spectrum.
* *brain* — an arbitrary-topology recurrent network (NEAT-style: nodes and
  connections with historical markings). Fixed outputs are turn, thrust (chosen
  per tick, capped by `max_speed`), eat, attack and mate; extra outputs drive
  the OUT ports. The whole population's brains run as one batched `torch`
  matrix multiply each tick.

**Evolution.** When two well-fed, species-compatible organisms of a compatible
mating type meet, they produce one offspring by NEAT crossover, then mutation.
There is no selection step — organisms that forage badly, freeze, or never find
a mate simply leave fewer descendants. Offspring are placed into a species by
compatibility distance; the threshold self-tunes toward a target species count,
and the count over time is recorded for the species panel.

## Architecture

```
src/ai_world/
├── config.py            map limits, database path (AI_WORLD_DATA_DIR)
├── world/               domain model — no pygame, no SQLite
│   ├── tiles.py         tile types + palette
│   ├── grid.py          numpy uint8 array (cells[y, x])
│   ├── generator.py     deterministic terrain from value noise
│   ├── fields.py        EnzymeField / TemperatureField / SpectrumField
│   ├── weather.py       season phase, drifting fronts, climate events
│   ├── genome.py        physiology + ports + NEAT graph; mutate / crossover / compat
│   ├── brain.py         genome → padded weight matrix; BrainStore (batched torch)
│   ├── entity.py        Entity — a detached snapshot record
│   ├── population.py    column-wise organism store + spatial hash
│   ├── species.py       SpeciesRegistry (assignment, pruning, census)
│   ├── params.py        EcoParams — every tunable constant
│   └── world.py         aggregate + attach_ecosystem / rebuild_ecosystem
├── simulation/
│   ├── clock.py         time control
│   ├── engine.py        one step = run the registered systems
│   └── ecosystem.py     weather / fields / sense / think / act / vitals / speciation
├── persistence/
│   ├── database.py      SQLite + PRAGMA user_version migrations
│   ├── serialization.py grid / array / genome / eco-state blobs
│   ├── entities.py      entity rows
│   ├── species.py       species + census rows
│   └── worlds.py        WorldRepository: list / create / load / save / delete
└── ui/                  pygame layer only
    ├── camera.py · renderer.py · field_overlay.py · entity_renderer.py
    └── screens/         menu → create / browser → simulation
```

Dependencies flow one way: `ui → simulation → world` and `persistence → world`.
The `world` and `simulation` layers never import pygame, so the simulation runs
headless (and is tested that way).

## Database

A single `worlds.db` in the user data directory (`%LOCALAPPDATA%\ai-world` on
Windows; override with `AI_WORLD_DATA_DIR`). The schema is versioned through
`PRAGMA user_version`; a save stores the full population (genomes and all),
every field, the RNG state, the innovation registry and the species registry,
so a reloaded world continues bit-for-bit on the same machine.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

Covers the fields (bounded regen, diffusion, temperature gradient), the genome
(mutation invariants over many generations, deterministic mutation, crossover
gene provenance, compatibility distance), the brains (compile shapes, recurrent
state, batched-step determinism), the ecosystem (energy/hp invariants,
starvation, trait-scaled upkeep, sexual reproduction, corpse recycling,
determinism) and the repository (round-trips including the ecosystem, migration
from a terrain-only save).
