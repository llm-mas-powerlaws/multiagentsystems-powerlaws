"""
benchmark_wrappers/task_curator.py
------------------------------------
Selects and balances the final task portfolio for the paper:

  MARBLE:     10 tasks  (3 easy,  4 medium,  3 hard)
  GAIA:       10 tasks  (3 easy,  3 medium,  4 hard)
  REALM:      10 tasks  (2 easy,  4 medium,  4 hard)
  SWE-bench:  10 tasks  (2 easy,  3 medium,  5 hard)
  ─────────────────────────────────────────────────
  Total:      40 tasks

The curator:
  1. Loads raw tasks from each benchmark wrapper
  2. Balances difficulty tiers to match targets above
  3. Saves a canonical manifest to data/task_manifest.json
  4. Returns List[BenchmarkTask] ready for the sweep runner

Usage
-----
    from benchmark_wrappers.task_curator import curate_tasks
    tasks = curate_tasks(data_root=Path("data"))

Or from the command line:
    python scripts/build_manifest.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from execution.graph_runner import BenchmarkTask


# ── Target portfolio counts ───────────────────────────────────────

PORTFOLIO: Dict[str, Dict[str, int]] = {
    "MARBLE":    {"easy": 1,  "medium": 2,  "hard": 2},
    "GAIA":      {"easy": 1,  "medium": 2,  "hard": 2},
    "REALM":     {"easy": 1,  "medium": 2,  "hard": 2},
    "SWE-bench": {"easy": 1,  "medium": 2,  "hard": 2},
}


# ── Per-benchmark loader registry ─────────────────────────────────

def _load_all_raw(seed: int = 42) -> Dict[str, List[BenchmarkTask]]:
    """Load all raw tasks from every benchmark (may load more than needed)."""
    from benchmark_wrappers.marble     import load_marble_tasks
    from benchmark_wrappers.realm_bench import load_realm_tasks
    from benchmark_wrappers.gaia        import load_gaia_tasks
    from benchmark_wrappers.swebench    import load_swebench_tasks

    loaders = {
        "MARBLE":    lambda: load_marble_tasks(max_tasks=60),
        "GAIA":      lambda: load_gaia_tasks(max_tasks=60),
        "REALM":     lambda: load_realm_tasks(max_tasks=60),
        "SWE-bench": lambda: load_swebench_tasks(max_tasks=30),
    }

    raw: Dict[str, List[BenchmarkTask]] = {}
    for name, fn in loaders.items():
        try:
            raw[name] = fn()
            print(f"  {name}: loaded {len(raw[name])} tasks")
        except Exception as e:
            print(f"  {name}: FAILED to load — {e}")
            raw[name] = []
    return raw


def _select(
    tasks: List[BenchmarkTask],
    targets: Dict[str, int],
    seed: int,
) -> List[BenchmarkTask]:
    """
    From a pool of tasks, select exactly the target number per difficulty tier.
    If a tier is under-represented, fill with tasks from any tier.
    """
    rng    = random.Random(seed)
    by_diff: Dict[str, List[BenchmarkTask]] = {"easy": [], "medium": [], "hard": []}
    for t in tasks:
        if t.difficulty in by_diff:
            by_diff[t.difficulty].append(t)

    selected: List[BenchmarkTask] = []
    for diff, count in targets.items():
        pool = by_diff.get(diff, [])
        rng.shuffle(pool)
        selected.extend(pool[:count])
        shortfall = count - len(pool)
        if shortfall > 0:
            # Borrow from any other tier
            others = [t for t in tasks if t not in selected]
            rng.shuffle(others)
            selected.extend(others[:shortfall])

    return selected


# ── Public API ─────────────────────────────────────────────────────

def curate_tasks(
    data_root: Path = Path("data"),
    seed: int       = 42,
    save_manifest:  bool = True,
) -> List[BenchmarkTask]:
    """
    Load, balance, and return the full 70-task portfolio.
    Optionally saves data/task_manifest.json.
    """
    print("Loading benchmark tasks...")
    raw = _load_all_raw(seed=seed)

    portfolio: List[BenchmarkTask] = []
    for bench, targets in PORTFOLIO.items():
        pool     = raw.get(bench, [])
        selected = _select(pool, targets, seed=seed)
        portfolio.extend(selected)
        total    = sum(targets.values())
        print(f"  {bench}: selected {len(selected)}/{total} target tasks")

    if save_manifest:
        data_root.mkdir(parents=True, exist_ok=True)
        manifest_path = data_root / "task_manifest.json"
        manifest = [
            {
                "task_id":     t.task_id,
                "benchmark":   t.benchmark,
                "task_family": t.task_family,
                "difficulty":  t.difficulty,
                "prompt_preview": t.prompt[:120] + "..." if len(t.prompt) > 120 else t.prompt,
                "has_gold":    t.gold_answer is not None,
                "requires_tools": t.requires_tools,
            }
            for t in portfolio
        ]
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"\nManifest saved → {manifest_path}  ({len(portfolio)} tasks)")

    return portfolio


def load_from_manifest(
    manifest_path: Path,
    benchmarks:    Optional[List[str]] = None,
    difficulties:  Optional[List[str]] = None,
    families:      Optional[List[str]] = None,
) -> List[BenchmarkTask]:
    """
    Reload full BenchmarkTask objects from a saved manifest.
    Supports filtering by benchmark, difficulty, or task_family.
    """
    raw_manifest = json.loads(manifest_path.read_text())

    # Apply filters
    if benchmarks:
        raw_manifest = [r for r in raw_manifest if r["benchmark"] in benchmarks]
    if difficulties:
        raw_manifest = [r for r in raw_manifest if r["difficulty"] in difficulties]
    if families:
        raw_manifest = [r for r in raw_manifest if r["task_family"] in families]

    # Re-hydrate from individual benchmark loaders
    from benchmark_wrappers.marble      import load_marble_tasks
    from benchmark_wrappers.realm_bench import load_realm_tasks
    from benchmark_wrappers.gaia         import load_gaia_tasks
    from benchmark_wrappers.swebench     import load_swebench_tasks

    all_tasks: Dict[str, BenchmarkTask] = {}
    for fn in [
        lambda: load_marble_tasks(60),
        lambda: load_realm_tasks(60),
        lambda: load_gaia_tasks(60),
        lambda: load_swebench_tasks(30),
    ]:
        try:
            for t in fn():
                all_tasks[t.task_id] = t
        except Exception:
            pass

    target_ids = {r["task_id"] for r in raw_manifest}
    return [t for tid, t in all_tasks.items() if tid in target_ids]


def portfolio_summary(tasks: List[BenchmarkTask]) -> str:
    """Pretty-print a summary of the task portfolio."""
    from collections import Counter
    bench_diff: Dict[str, Counter] = {}
    for t in tasks:
        if t.benchmark not in bench_diff:
            bench_diff[t.benchmark] = Counter()
        bench_diff[t.benchmark][t.difficulty] += 1

    lines = [f"Task portfolio: {len(tasks)} total tasks", ""]
    for bench, counts in sorted(bench_diff.items()):
        e = counts.get("easy", 0)
        m = counts.get("medium", 0)
        h = counts.get("hard", 0)
        lines.append(
            f"  {bench:<12}  {e+m+h:3d} tasks  "
            f"({e} easy / {m} medium / {h} hard)"
        )
    return "\n".join(lines)