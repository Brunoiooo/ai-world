"""Population-recovery sweep + per-run census.

Runs the ecosystem at the shipped defaults across a range of seeds and reports
whether each lineage recovers from the founder die-off. Optionally dumps a
per-tick census CSV (headcount, births/deaths, mean energy/age, species count,
food total, mean fecundity genes) so a balance change can be judged on numbers
instead of vibes.

    python scripts/sweep.py                    # 8 seeds, 6000 ticks
    python scripts/sweep.py --seeds 12 --ticks 10000
    python scripts/sweep.py --seeds 4 --census out/   # + write out/census_seed*.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402

from ai_world.simulation.ecosystem import default_systems  # noqa: E402
from ai_world.world.generator import generate_grid  # noqa: E402
from ai_world.world.params import DEFAULT_ECOSYSTEM_DIM, EcoParams  # noqa: E402
from ai_world.world.population import TRAIT_IX  # noqa: E402
from ai_world.world.world import World, attach_ecosystem  # noqa: E402

_LITTER = TRAIT_IX["litter_size"]
_CD_MULT = TRAIT_IX["repro_cooldown_mult"]
_SAMPLE_EVERY = 200


def _census_row(world: World) -> dict:
    pop = world.population
    n = len(pop)
    return {
        "tick": world.tick,
        "pop": n,
        "births": pop.births,
        "deaths": pop.deaths,
        "mean_energy": round(float(np.mean(pop.energy)), 4) if n else 0.0,
        "mean_age": round(float(np.mean(pop.age)), 1) if n else 0.0,
        "max_gen": int(pop.generation.max()) if n else 0,
        "n_species": len(set(pop.species_id.tolist())) if n else 0,
        "food_total": round(float(world.food.total().sum()), 1),
        "mean_litter_gene": round(float(np.mean(pop.traits[:, _LITTER])), 3) if n else 0.0,
        "mean_cooldown_gene": round(float(np.mean(pop.traits[:, _CD_MULT])), 3) if n else 0.0,
    }


def run_seed(seed: int, ticks: int, census_dir: Path | None) -> dict:
    params = EcoParams()
    grid = generate_grid(DEFAULT_ECOSYSTEM_DIM, DEFAULT_ECOSYSTEM_DIM, seed)
    world = World(name=f"sweep-{seed}", grid=grid, seed=seed)
    attach_ecosystem(world, params)
    systems = default_systems()

    rows: list[dict] = []
    trough = len(world.population)
    settled_min = 10**9
    for t in range(ticks):
        for system in systems:
            system(world)
        world.tick += 1
        n = len(world.population)
        trough = min(trough, n)
        if t > 400:
            settled_min = min(settled_min, n)
        if census_dir is not None and t % _SAMPLE_EVERY == 0:
            rows.append(_census_row(world))

    if census_dir is not None:
        census_dir.mkdir(parents=True, exist_ok=True)
        path = census_dir / f"census_seed{seed}.csv"
        with path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    pop = world.population
    final = len(pop)
    trough_nz = max(trough, 1)
    if final == 0:
        verdict = "extinct"
    elif final <= 5:
        verdict = "critical"
    elif final < 0.6 * trough_nz:
        verdict = "declining"
    elif final >= 1.8 * trough_nz:
        verdict = "RECOVERED"
    else:
        verdict = "stable"
    return {
        "seed": seed,
        "trough": trough,
        "settled_min": 0 if settled_min == 10**9 else settled_min,
        "final": final,
        "births": pop.births,
        "deaths": pop.deaths,
        "n_species": len(set(pop.species_id.tolist())) if final else 0,
        "verdict": verdict,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=8, help="number of seeds, 0..N-1")
    ap.add_argument("--ticks", type=int, default=6000)
    ap.add_argument("--census", type=Path, default=None, metavar="DIR",
                    help="also write a per-run census CSV into DIR")
    args = ap.parse_args()

    print(f"defaults | dim {DEFAULT_ECOSYSTEM_DIM} | {args.ticks} ticks | seeds 0..{args.seeds - 1}\n")
    header = f"{'seed':>4}  {'trough':>6}  {'min>400':>7}  {'final':>5}  {'births':>6}  {'deaths':>6}  {'spp':>3}  verdict"
    print(header)
    print("-" * len(header))
    results = []
    for seed in range(args.seeds):
        r = run_seed(seed, args.ticks, args.census)
        results.append(r)
        print(f"{r['seed']:>4}  {r['trough']:>6}  {r['settled_min']:>7}  {r['final']:>5}  "
              f"{r['births']:>6}  {r['deaths']:>6}  {r['n_species']:>3}  {r['verdict']}")

    n = len(results)
    rec = sum(r["verdict"] == "RECOVERED" for r in results)
    extinct = sum(r["final"] == 0 for r in results)
    print("-" * len(header))
    print(f"recovered {rec}/{n}   extinct {extinct}/{n}   "
          f"median final {int(np.median([r['final'] for r in results]))}")


if __name__ == "__main__":
    main()
