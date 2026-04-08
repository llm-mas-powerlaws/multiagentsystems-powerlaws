#!/usr/bin/env python3
"""
scripts/extract_events.py
--------------------------
Extract all coordination events and compute metrics across topologies,
benchmarks, and tasks for H1, H2, and H3.

Produces:
  {out}/observables.json          — raw observable lists per (topology, benchmark, N)
  {out}/observables_pooled.json   — pooled across all dimensions for global fits
  {out}/run_metrics.csv           — one row per run: all H2 metrics from run_metadata
  {out}/event_metrics.csv         — per-event metrics for regression analysis
  {out}/topology_summary.csv      — H1/H2/H3 metrics aggregated per topology
  {out}/scaling_table.csv         — H3: metrics vs N (for xmax scaling figures)

Usage
-----
  # After sweep on multiple GPUs:
  python scripts/extract_events.py \\
      --data-roots /scratch/.../MARBLE /scratch/.../GAIA \\
                   /scratch/.../REALM  /scratch/.../SWE-bench \\
      --out data/extracted

  # Filter to one topology/benchmark:
  python scripts/extract_events.py \\
      --data-roots data/runs \\
      --topology chain --benchmark MARBLE \\
      --out data/extracted_chain_marble
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from event_extraction.coordination import (
    _load_all_events, filter_events,
    extract_delegation_cascades, extract_revision_waves,
    extract_contradiction_bursts, extract_merge_fanin,
    extract_influence_per_agent,
    extract_tce_per_run_from_events,
    PRIMARY_OBSERVABLES, SECONDARY_OBSERVABLES, EXPLORATORY_OBSERVABLES,
)
from event_extraction.tce import (
    tce_per_run, tce_per_event, tce_per_agent_per_run,
    tce_per_cascade, tce_per_revision_wave,
)
from event_extraction.graph_builder import (
    extract_graph_rows, graph_observables_from_tables,
)


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract events and compute metrics for H1/H2/H3"
    )
    p.add_argument("--data-roots", nargs="+", required=True, metavar="PATH")
    p.add_argument("--out", default="data/extracted", metavar="PATH")
    p.add_argument("--topology",  default=None, help="Filter to one topology")
    p.add_argument("--benchmark", default=None, help="Filter to one benchmark")
    p.add_argument("--n",         type=int, default=None, help="Filter to one agent count")
    p.add_argument("--verbose",   action="store_true")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _safe(v, default=0.0):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return default
    return v


def _gini(arr: np.ndarray) -> float:
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0 or arr.sum() == 0:
        return 0.0
    arr = np.sort(arr)
    n   = len(arr)
    idx = np.arange(1, n + 1)
    return float((2 * (idx * arr).sum() / (n * arr.sum()) - (n + 1) / n))


def _xmax(sizes: List[float]) -> float:
    return float(max(sizes)) if sizes else 0.0


def _describe(sizes: List[float]) -> Dict[str, float]:
    if not sizes:
        return {"n": 0, "mean": 0.0, "median": 0.0, "std": 0.0,
                "xmax": 0.0, "gini": 0.0, "p90": 0.0, "p99": 0.0}
    arr = np.array(sizes, dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if len(arr) == 0:
        return {"n": 0, "mean": 0.0, "median": 0.0, "std": 0.0,
                "xmax": 0.0, "gini": 0.0, "p90": 0.0, "p99": 0.0}
    return {
        "n":      len(arr),
        "mean":   round(float(arr.mean()), 4),
        "median": round(float(np.median(arr)), 4),
        "std":    round(float(arr.std()), 4),
        "xmax":   round(float(arr.max()), 2),
        "gini":   round(_gini(arr), 4),
        "p90":    round(float(np.percentile(arr, 90)), 4),
        "p99":    round(float(np.percentile(arr, 99)), 4),
    }


# ─────────────────────────────────────────────────────────────────
# H1 observables per (topology, benchmark, N)
# ─────────────────────────────────────────────────────────────────

H1_EXTRACTORS = {
    "delegation_cascade":  extract_delegation_cascades,
    "revision_wave":       extract_revision_waves,
    "contradiction_burst": extract_contradiction_bursts,
    "merge_fanin":         extract_merge_fanin,
    "influence_per_agent": extract_influence_per_agent,
}


def extract_h1_observables(
    roots: List[Path],
    topology_filter: Optional[str],
    benchmark_filter: Optional[str],
    n_filter: Optional[int],
    verbose: bool,
) -> Dict[str, Any]:
    """
    Extract H1 observables at three granularities:
      - pooled (all runs merged)
      - per topology
      - per (topology, benchmark, N)
    """
    print("\n── H1: Extracting coordination observables ──────────────────")

    # Collect events grouped by (topology, benchmark, num_agents)
    groups: Dict[tuple, List[Dict]] = defaultdict(list)
    for root in roots:
        if not root.exists():
            continue
        all_evs = _load_all_events(root)
        filtered = filter_events(
            all_evs,
            topology=topology_filter,
            benchmark=benchmark_filter,
            num_agents=n_filter,
        )
        for ev in filtered:
            key = (
                ev.get("topology", "unknown"),
                ev.get("benchmark", "unknown"),
                ev.get("num_agents", 0),
            )
            groups[key].append(ev)

    # TCE per run (file-level, not event-level)
    tce_by_group: Dict[tuple, List[float]] = defaultdict(list)
    for root in roots:
        if not root.exists():
            continue
        for meta_path in root.rglob("run_metadata.json"):
            try:
                meta  = json.loads(meta_path.read_text())
                topo  = meta.get("run_id", "").split("__")[1] if "__" in meta.get("run_id","") else "unknown"
                bench = meta.get("run_id", "").split("__")[0] if "__" in meta.get("run_id","") else "unknown"
                n     = meta.get("run_id", "").split("__")[2].replace("n","") if "__" in meta.get("run_id","") else "0"
                key   = (topo, bench, int(n) if n.isdigit() else 0)
                tokens = meta.get("tokens_total", 0)
                if tokens > 0:
                    tce_by_group[key].append(float(tokens))
            except Exception:
                pass

    result: Dict[str, Any] = {
        "pooled":   {},
        "by_topo":  {},
        "by_group": {},
    }

    all_events_flat: List[Dict] = []
    for evs in groups.values():
        all_events_flat.extend(evs)

    # Pooled
    for obs_name, extractor in H1_EXTRACTORS.items():
        sizes = extractor(all_events_flat)
        result["pooled"][obs_name] = sizes
        if verbose:
            print(f"  pooled/{obs_name:<24} n={len(sizes):6d}  "
                  f"xmax={_xmax(sizes):.1f}")

    # TCE pooled
    all_tce = [v for vals in tce_by_group.values() for v in vals]
    result["pooled"]["tce_per_run"] = all_tce
    if verbose:
        print(f"  pooled/tce_per_run             n={len(all_tce):6d}  "
              f"xmax={_xmax(all_tce):.1f}")

    # Per topology
    topo_events: Dict[str, List[Dict]] = defaultdict(list)
    for (topo, bench, n), evs in groups.items():
        topo_events[topo].extend(evs)

    for topo, evs in topo_events.items():
        result["by_topo"][topo] = {}
        for obs_name, extractor in H1_EXTRACTORS.items():
            sizes = extractor(evs)
            result["by_topo"][topo][obs_name] = sizes
        tce_all = [v for (t, b, n), vals in tce_by_group.items()
                   if t == topo for v in vals]
        result["by_topo"][topo]["tce_per_run"] = tce_all

    # Per group
    for (topo, bench, n), evs in groups.items():
        key_str = f"{topo}__{bench}__n{n}"
        result["by_group"][key_str] = {}
        for obs_name, extractor in H1_EXTRACTORS.items():
            result["by_group"][key_str][obs_name] = extractor(evs)
        result["by_group"][key_str]["tce_per_run"] = tce_by_group.get((topo, bench, n), [])

    n_groups = len(result["by_group"])
    print(f"  Extracted {n_groups} (topology, benchmark, N) groups")
    return result


# ─────────────────────────────────────────────────────────────────
# H2 run-level metrics from run_metadata.json
# ─────────────────────────────────────────────────────────────────

H2_META_FIELDS = [
    # Core outcome
    "task_score", "task_success",
    # Completeness
    "completion_ratio", "coherence_score", "integration_score",
    "num_subtasks_total", "num_subtasks_completed", "num_subtasks_open_final",
    "num_claims_total", "num_claims_merged", "num_claims_unresolved_final",
    # Event counts
    "num_revisions_total", "num_contradictions_total",
    "num_merges_total", "num_endorsements_total",
    # Efficiency
    "tokens_total", "wall_time_seconds", "messages_total",
    "success_per_token", "completion_per_token", "quality_adjusted_efficiency",
    # Activation (H3)
    "num_unique_agents_activated", "activation_rate", "mean_active_per_step",
    # Benchmark-specific
    "gaia_exact_match", "gaia_rubric_score",
    "swe_patch_applied", "swe_tests_passed", "swe_tests_total",
    "marble_subgoals_completed", "marble_team_objective_met",
    "realm_plan_valid", "realm_recovered_from_disruption",
]

H2_EXTRA_FIELDS = [
    "claim_participation_rate", "resolution_rate",
    "revisions_per_claim", "merges_per_claim",
    "contradictions_per_claim", "endorsements_per_claim",
    "tokens_per_event", "events_per_agent", "architecture",
]


def extract_h2_run_metrics(
    roots: List[Path],
    topology_filter: Optional[str],
    benchmark_filter: Optional[str],
    n_filter: Optional[int],
) -> List[Dict]:
    """
    Load one row per run from run_metadata.json.
    Returns list of dicts ready for CSV export.
    """
    print("\n── H2: Loading run-level metrics ────────────────────────────")
    rows = []

    for root in roots:
        if not root.exists():
            continue
        for meta_path in sorted(root.rglob("run_metadata.json")):
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:
                continue

            run_id = meta.get("run_id", "")
            parts  = run_id.split("__")

            # Parse run_id: benchmark__topology__nN__sS__task_id
            bench  = parts[0] if len(parts) > 0 else "unknown"
            topo   = parts[1] if len(parts) > 1 else "unknown"
            n_str  = parts[2].replace("n", "") if len(parts) > 2 else "0"
            seed   = parts[3].replace("s", "") if len(parts) > 3 else "0"
            task_id = parts[4] if len(parts) > 4 else "unknown"

            try:
                num_agents = int(n_str)
            except ValueError:
                num_agents = 0

            # Apply filters
            if topology_filter  and topo  != topology_filter:   continue
            if benchmark_filter and bench != benchmark_filter:   continue
            if n_filter         and num_agents != n_filter:      continue

            row = {
                "run_id":     run_id,
                "benchmark":  bench,
                "topology":   topo,
                "num_agents": num_agents,
                "seed":       seed,
                "task_id":    task_id,
            }

            # Standard meta fields
            for field in H2_META_FIELDS:
                row[field] = _safe(meta.get(field))

            # Extra fields from meta["extra"]
            extra = meta.get("extra", {})
            for field in H2_EXTRA_FIELDS:
                row[field] = _safe(extra.get(field))

            rows.append(row)

    print(f"  Loaded {len(rows)} run records")
    return rows


# ─────────────────────────────────────────────────────────────────
# H2 topology summary (aggregated)
# ─────────────────────────────────────────────────────────────────

def build_topology_summary(run_rows: List[Dict]) -> List[Dict]:
    """
    Aggregate H2 metrics per topology for comparison table.
    """
    print("\n── H2: Building topology summary ────────────────────────────")

    by_topo: Dict[str, List[Dict]] = defaultdict(list)
    for row in run_rows:
        by_topo[row["topology"]].append(row)

    summary = []
    numeric_fields = [
        "task_score", "completion_ratio", "coherence_score", "integration_score",
        "num_revisions_total", "num_contradictions_total", "num_merges_total",
        "num_endorsements_total", "tokens_total", "wall_time_seconds",
        "success_per_token", "quality_adjusted_efficiency",
        "claim_participation_rate", "resolution_rate",
        "revisions_per_claim", "merges_per_claim",
        "num_unique_agents_activated", "activation_rate",
    ]

    for topo, rows in sorted(by_topo.items()):
        row_out = {"topology": topo, "n_runs": len(rows)}
        for field in numeric_fields:
            vals = [r[field] for r in rows
                    if r.get(field) is not None and r[field] != 0.0
                    or field in ("task_score",)]  # include zeros for task_score
            vals = [v for v in vals if isinstance(v, (int, float)) and np.isfinite(v)]
            if vals:
                row_out[f"{field}_mean"] = round(float(np.mean(vals)), 4)
                row_out[f"{field}_std"]  = round(float(np.std(vals)), 4)
                row_out[f"{field}_p50"]  = round(float(np.median(vals)), 4)
            else:
                row_out[f"{field}_mean"] = None
                row_out[f"{field}_std"]  = None
                row_out[f"{field}_p50"]  = None
        summary.append(row_out)

    return summary


# ─────────────────────────────────────────────────────────────────
# H3 scaling table (xmax vs N)
# ─────────────────────────────────────────────────────────────────

def build_scaling_table(h1_obs: Dict[str, Any]) -> List[Dict]:
    """
    H3: for each (topology, observable), compute descriptive stats
    as a function of N for xmax scaling analysis.
    """
    print("\n── H3: Building scaling table ───────────────────────────────")
    rows = []

    for group_key, obs_dict in h1_obs["by_group"].items():
        parts = group_key.split("__")
        if len(parts) < 3:
            continue
        topo  = parts[0]
        bench = parts[1]
        n_str = parts[2].replace("n", "")
        try:
            n = int(n_str)
        except ValueError:
            n = 0

        for obs_name, sizes in obs_dict.items():
            stats = _describe(sizes)
            rows.append({
                "topology":    topo,
                "benchmark":   bench,
                "num_agents":  n,
                "observable":  obs_name,
                **{f"obs_{k}": v for k, v in stats.items()},
            })

    print(f"  Built {len(rows)} scaling rows")
    return rows


# ─────────────────────────────────────────────────────────────────
# Per-event metrics for regression / visualization
# ─────────────────────────────────────────────────────────────────

EVENT_FIELDS = [
    "run_id", "benchmark", "topology", "num_agents", "seed",
    "step_id", "agent_id", "agent_role", "event_type",
    "tokens_total_event", "latency_ms", "claim_depth", "subtask_depth",
    "merge_num_inputs", "claim_type", "subtask_type",
    "revision_chain_id", "contradiction_group_id",
    "agent_influence_score_so_far", "agent_degree_so_far",
    "visible_neighbors",  # will be converted to degree
    "action_success",
]


def extract_event_level(
    roots: List[Path],
    topology_filter: Optional[str],
    benchmark_filter: Optional[str],
    n_filter: Optional[int],
    max_events: int = 500_000,
) -> List[Dict]:
    """
    Extract per-event rows for regression / fine-grained analysis.
    Capped at max_events to keep memory sane.
    """
    print("\n── Per-event extraction ─────────────────────────────────────")
    rows = []

    for root in roots:
        if not root.exists():
            continue
        for ep in sorted(root.rglob("events.jsonl")):
            for ev in _iter_events_file(ep):
                if topology_filter  and ev.get("topology")   != topology_filter:  continue
                if benchmark_filter and ev.get("benchmark")  != benchmark_filter: continue
                if n_filter         and ev.get("num_agents") != n_filter:         continue

                row = {f: ev.get(f) for f in EVENT_FIELDS}
                # Flatten visible_neighbors to degree
                vn = ev.get("visible_neighbors") or []
                row["visible_degree"] = len(vn) if isinstance(vn, list) else 0
                row.pop("visible_neighbors", None)
                rows.append(row)

                if len(rows) >= max_events:
                    print(f"  Capped at {max_events} events")
                    return rows

    print(f"  Extracted {len(rows)} events")
    return rows


def _iter_events_file(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except Exception:
                    pass


# ─────────────────────────────────────────────────────────────────
# CSV writer (no pandas dependency)
# ─────────────────────────────────────────────────────────────────

def _write_csv(rows: List[Dict], path: Path) -> None:
    if not rows:
        print(f"  (empty — skipped {path.name})")
        return
    try:
        import pandas as pd  # type: ignore
        pd.DataFrame(rows).to_csv(path, index=False)
    except ImportError:
        import csv
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    print(f"  → {path}  ({len(rows)} rows)")


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    args  = parse_args()
    out   = Path(args.out)
    roots = [Path(r) for r in args.data_roots]
    out.mkdir(parents=True, exist_ok=True)

    print(f"\nExtracting events from {len(roots)} data root(s)")
    for r in roots:
        n_files = len(list(r.rglob("events.jsonl"))) if r.exists() else 0
        print(f"  {r}  ({n_files} event files)")

    # ── H1: observables ──────────────────────────────────────────
    h1_obs = extract_h1_observables(
        roots,
        topology_filter=args.topology,
        benchmark_filter=args.benchmark,
        n_filter=args.n,
        verbose=args.verbose,
    )

    # Save pooled observables (for power-law fitting)
    pooled_path = out / "observables_pooled.json"
    pooled_path.write_text(json.dumps(h1_obs["pooled"], indent=2))
    print(f"\n  Pooled observables → {pooled_path}")

    # Save per-group observables (for topology/N comparisons)
    by_group_path = out / "observables_by_group.json"
    by_group_path.write_text(json.dumps(h1_obs["by_group"], indent=2))
    print(f"  Per-group observables → {by_group_path}")

    # Observable summary stats per topology
    topo_obs_summary = []
    for topo, obs_dict in h1_obs["by_topo"].items():
        for obs_name, sizes in obs_dict.items():
            stats = _describe(sizes)
            topo_obs_summary.append({
                "topology": topo,
                "observable": obs_name,
                **{f"obs_{k}": v for k, v in stats.items()},
            })
    _write_csv(topo_obs_summary, out / "h1_observable_summary.csv")

    # ── H2: run metrics ──────────────────────────────────────────
    run_rows = extract_h2_run_metrics(
        roots,
        topology_filter=args.topology,
        benchmark_filter=args.benchmark,
        n_filter=args.n,
    )
    _write_csv(run_rows, out / "run_metrics.csv")

    # Topology summary
    topo_summary = build_topology_summary(run_rows)
    _write_csv(topo_summary, out / "topology_summary.csv")

    # ── H3: scaling table ────────────────────────────────────────
    scaling_rows = build_scaling_table(h1_obs)
    _write_csv(scaling_rows, out / "scaling_table.csv")

    # ── Per-event (for viz/regression) ──────────────────────────
    event_rows = extract_event_level(
        roots,
        topology_filter=args.topology,
        benchmark_filter=args.benchmark,
        n_filter=args.n,
    )
    _write_csv(event_rows, out / "event_metrics.csv")

    # ── Graph reconstruction ──────────────────────────────────────
    # Reconstruct claim propagation DAG and agent influence graph.
    # These tables enable full network-science analysis and richer
    # H1 observables (cascade_size, claim_out_degree, merge_fan_in).
    claim_nodes, claim_edges, agent_edges, run_graph_summary = extract_graph_rows(
        roots,
        topology_filter=args.topology,
        benchmark_filter=args.benchmark,
        n_filter=args.n,
        verbose=args.verbose,
    )
    _write_csv(claim_nodes,       out / "claim_nodes.csv")
    _write_csv(claim_edges,       out / "claim_edges.csv")
    _write_csv(agent_edges,       out / "agent_edges.csv")
    _write_csv(run_graph_summary, out / "run_graph_summary.csv")

    # Merge graph-derived observables into pooled pool for power-law fitting.
    # These complement event-level observables with full DAG structure:
    #   cascade_size      = reachable descendants per root claim (BFS)
    #   claim_out_degree  = fan-out per claim (claim reuse)
    #   merge_fan_in      = in-degree for merge_claims events
    #   agent_out_degree  = total outgoing influence per agent
    graph_obs = graph_observables_from_tables(claim_nodes, claim_edges)
    print("\n  Graph-derived H1 observables:")
    for obs_name, sizes in sorted(graph_obs.items()):
        arr = [s for s in sizes if s > 0]
        xmax = float(max(arr)) if arr else 0.0
        print(f"    {obs_name:<28} n={len(arr):6d}  xmax={xmax:.1f}")
        if arr:
            h1_obs["pooled"].setdefault(obs_name, []).extend(arr)

    # Re-save pooled observables with graph-derived additions
    pooled_path.write_text(json.dumps(h1_obs["pooled"], indent=2))

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Extraction complete → {out}/")
    print(f"  observables_pooled.json   (H1 event + graph observables for fitting)")
    print(f"  observables_by_group.json (per topology/benchmark/N)")
    print(f"  h1_observable_summary.csv (H1 descriptive stats)")
    print(f"  run_metrics.csv           (H2 per-run metrics)")
    print(f"  topology_summary.csv      (H2 aggregated per topology)")
    print(f"  scaling_table.csv         (H3 xmax vs N)")
    print(f"  event_metrics.csv         (per-event for viz/regression)")
    print(f"  claim_nodes.csv           (graph: one row per claim)")
    print(f"  claim_edges.csv           (graph: parent→child propagation)")
    print(f"  agent_edges.csv           (graph: agent influence edges)")
    print(f"  run_graph_summary.csv     (graph: per-run stats)")
    print(f"\nNext: python scripts/run_analysis.py --data-roots {args.out}/observables_pooled.json")


if __name__ == "__main__":
    main()
