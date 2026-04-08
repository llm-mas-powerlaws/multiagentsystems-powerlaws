"""
event_extraction/coordination.py
----------------------------------
Reads events.jsonl files and extracts coordination event size
distributions for power-law fitting.

Observable families
-------------------
PRIMARY (graph-derived — use for paper claims):
  cascade_size, claim_in_degree, claim_out_degree, agent_out_degree
  → these come from graph_builder.py, not this file

SECONDARY (event-native — structural, bounded by topology design):
  merge_fanin          MERGE_CLAIMS.merge_num_inputs  ← cleanest extractor here
  tce_per_run          total tokens per run            ← filter-aware
  revision_wave        agents per revision_chain_id    ← run-scoped
  contradiction_burst  agents per contradiction_group  ← run-scoped
  delegation_cascade   agents per subtask tree         ← run-scoped, BFS

EXPLORATORY (heuristic — do not drive paper conclusions):
  influence_per_agent  target-count proxy              ← downgraded label

Fix summary vs prior version
-----------------------------
1. All grouping keys are now run-scoped: (run_id, group_id)
   Prevents cross-run ID collisions merging unrelated cascades.
2. TCE extraction is now filter-aware (event-based, not file-based).
   extract_tce_per_run(data_root) replaced with
   extract_tce_per_run_from_events(events).
3. Delegation cascade uses subtask DAG reconstruction via
   parent_subtask_id → subtask_id edges, not just root grouping.
4. influence_per_agent renamed/docstring downgraded to proxy status.
5. extract_all_observables wires TCE correctly through filtered events.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


# ── Event type constants ──────────────────────────────────────────────

DELEGATE   = "delegate_subtask"
COMPLETE   = "complete_subtask"
PROPOSE    = "propose_claim"
REVISE     = "revise_claim"
CONTRADICT = "contradict_claim"
MERGE      = "merge_claims"
ENDORSE    = "endorse_claim"
FINALIZE   = "finalize_answer"


# ────────────────────────────────────────────────────────────────
# Internal helpers
# ────────────────────────────────────────────────────────────────

def _iter_events(events_path: Path):
    with open(events_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def _load_all_events(data_root: Path) -> List[Dict]:
    """Recursively load all events from all runs under data_root."""
    events = []
    for path in sorted(data_root.rglob("events.jsonl")):
        events.extend(_iter_events(path))
    return events


def _group_by_run(events: List[Dict]) -> Dict[str, List[Dict]]:
    """Bucket events by run_id. Skips events without run_id."""
    by_run: Dict[str, List[Dict]] = defaultdict(list)
    for ev in events:
        rid = ev.get("run_id")
        if rid:
            by_run[rid].append(ev)
    return dict(by_run)


# ────────────────────────────────────────────────────────────────
# Observable extractors
# ────────────────────────────────────────────────────────────────

def extract_delegation_cascades(events: List[Dict]) -> List[float]:
    """
    Delegation cascade size = number of unique agents in one subtask tree.

    Reconstruction: build a subtask DAG per run using parent_subtask_id → subtask_id
    edges. For each root (in-degree 0), BFS to count all agents touched.

    Key improvement over prior version: run-scoped grouping prevents
    cross-run ID collisions; DAG reconstruction captures deeper trees.

    Size ≥ 2 (at least delegator + one worker).
    """
    sizes = []

    for run_id, run_events in _group_by_run(events).items():
        # Build subtask graph: parent_subtask_id → {child_subtask_id}
        children: Dict[str, Set[str]] = defaultdict(set)
        subtask_agents: Dict[str, Set[str]] = defaultdict(set)
        all_subtask_ids: Set[str] = set()
        has_parent: Set[str] = set()

        for ev in run_events:
            if ev.get("event_type") not in (DELEGATE, COMPLETE):
                continue
            sid = ev.get("subtask_id")
            if not sid:
                continue
            all_subtask_ids.add(sid)
            aid = ev.get("agent_id")
            if aid:
                subtask_agents[sid].add(aid)
            assigned_to = ev.get("subtask_assigned_to")
            if assigned_to:
                subtask_agents[sid].add(assigned_to)

            parent = ev.get("parent_subtask_id")
            root   = ev.get("root_subtask_id")
            if parent and parent != sid:
                children[parent].add(sid)
                has_parent.add(sid)
            elif root and root != sid:
                children[root].add(sid)
                has_parent.add(sid)

        # BFS from each root (no parent)
        roots = all_subtask_ids - has_parent
        if not roots and all_subtask_ids:
            roots = all_subtask_ids  # fallback: treat all as roots

        for root_id in roots:
            visited: Set[str] = set()
            queue = [root_id]
            agents_in_tree: Set[str] = set()
            while queue:
                cur = queue.pop()
                if cur in visited:
                    continue
                visited.add(cur)
                agents_in_tree.update(subtask_agents.get(cur, set()))
                queue.extend(children.get(cur, set()))
            if len(agents_in_tree) >= 2:
                sizes.append(float(len(agents_in_tree)))

    return sizes


def extract_revision_waves(events: List[Dict]) -> List[float]:
    """
    Revision wave size = number of distinct agents in one revision chain,
    identified by (run_id, revision_chain_id).

    Run-scoped to prevent cross-run ID collisions.
    """
    # key: (run_id, chain_id) → set of agent_ids
    chain_agents: Dict[Tuple, Set[str]] = defaultdict(set)

    for ev in events:
        if ev.get("event_type") != REVISE:
            continue
        chain = ev.get("revision_chain_id")
        if not chain:
            continue
        run_id = ev.get("run_id", "__unknown__")
        key = (run_id, chain)
        aid = ev.get("agent_id")
        if aid:
            chain_agents[key].add(aid)
        tgt = ev.get("target_agent_id")
        if tgt:
            chain_agents[key].add(tgt)

    return [float(len(a)) for a in chain_agents.values() if len(a) >= 1]


def extract_contradiction_bursts(events: List[Dict]) -> List[float]:
    """
    Contradiction burst size = distinct agents in one contradiction group,
    identified by (run_id, contradiction_group_id).

    Run-scoped to prevent cross-run ID collisions.
    """
    group_agents: Dict[Tuple, Set[str]] = defaultdict(set)

    for ev in events:
        if ev.get("event_type") != CONTRADICT and not ev.get("contradiction_group_id"):
            continue
        grp = ev.get("contradiction_group_id")
        if not grp:
            continue
        run_id = ev.get("run_id", "__unknown__")
        key = (run_id, grp)
        aid = ev.get("agent_id")
        if aid:
            group_agents[key].add(aid)
        tgt = ev.get("target_agent_id")
        if tgt:
            group_agents[key].add(tgt)

    return [float(len(a)) for a in group_agents.values() if len(a) >= 1]


def extract_merge_fanin(events: List[Dict]) -> List[float]:
    """
    Merge fan-in = merge_num_inputs for each MERGE_CLAIMS event.
    This is the cleanest event-native extractor — directly logged.
    """
    sizes = []
    for ev in events:
        if ev.get("event_type") == MERGE:
            n = ev.get("merge_num_inputs")
            if n and n >= 2:
                sizes.append(float(n))
    return sizes


def extract_tce_per_run_from_events(events: List[Dict]) -> List[float]:
    """
    Total Cognitive Effort (TCE) per run from a pre-filtered event list.
    Replaces the old extract_tce_per_run(data_root) which ignored filters.

    Sums tokens_total_event per run_id within the provided (filtered) events.
    """
    by_run: Dict[str, float] = defaultdict(float)
    for ev in events:
        run_id = ev.get("run_id")
        if run_id:
            by_run[run_id] += float(ev.get("tokens_total_event") or 0)
    return [v for v in by_run.values() if v > 0]


# Keep old name as alias for backward compatibility with callers
def extract_tce_per_run(data_root: Path) -> List[float]:
    """
    DEPRECATED: ignores any active filter.
    Use extract_tce_per_run_from_events(filtered_events) instead.
    Left here so existing callers don't break.
    """
    sizes = []
    for events_path in sorted(data_root.rglob("events.jsonl")):
        total = sum(
            float(ev.get("tokens_total_event") or 0)
            for ev in _iter_events(events_path)
        )
        if total > 0:
            sizes.append(total)
    return sizes


def extract_influence_per_agent(events: List[Dict]) -> List[float]:
    """
    EXPLORATORY PROXY — do not use as a primary paper observable.

    Heuristic target-count score: how often each agent was cited as
    target_agent_id, merge_synthesizer_agent_id, or endorsed_agent_id.

    This is topology-specific and not comparable across topologies.
    For paper claims, use graph-derived agent_out_degree or cascade_size
    from graph_builder.py instead.
    """
    influence: Dict[str, float] = defaultdict(float)
    for ev in events:
        tgt = ev.get("target_agent_id")
        syn = ev.get("merge_synthesizer_agent_id")
        end = ev.get("endorsed_agent_id")
        if tgt:
            influence[tgt] += 1.0
        if syn:
            influence[syn] += 2.0
        if end:
            influence[end] += 1.0
    return [v for v in influence.values() if v > 0]


# ────────────────────────────────────────────────────────────────
# Observable classification
# ────────────────────────────────────────────────────────────────

# Global project-level observable taxonomy — not all emitted by this file.
# Graph-derived observables (cascade_size, claim_descendant_count, etc.)
# come from graph_builder.py. This file emits only coordination
# observables: merge_fanin, tce_per_run, revision_wave, etc.
PRIMARY_OBSERVABLES = [
    # Graph-derived (from graph_builder.py) — strongest for paper
    "cascade_size",
    "claim_descendant_count",
    "agent_descendant_influence",
    "agent_unique_descendant_reach",  # strongest "few players" metric
    "claim_in_degree",
    "claim_out_degree",
    "agent_out_degree",
    "agent_out_strength",
    "merge_fan_in",
]

SECONDARY_OBSERVABLES = [
    # Event-native, structurally meaningful but often bounded
    "merge_fanin",
    "tce_per_run",
    "revision_wave",
    "contradiction_burst",
    "delegation_cascade",
]

EXPLORATORY_OBSERVABLES = [
    # Heuristic proxies — useful for debugging, not for claims
    "influence_per_agent",
]

# Backward-compatibility alias — some __init__.py and older callers import this
OBSERVABLE_NAMES = SECONDARY_OBSERVABLES


# ────────────────────────────────────────────────────────────────
# Slicing helpers
# ────────────────────────────────────────────────────────────────

def filter_events(
    events:      List[Dict],
    topology:    Optional[str] = None,
    benchmark:   Optional[str] = None,
    num_agents:  Optional[int] = None,
    task_family: Optional[str] = None,
    difficulty:  Optional[str] = None,
) -> List[Dict]:
    """Filter events by any combination of run dimensions."""
    out = events
    if topology:    out = [e for e in out if e.get("topology")    == topology]
    if benchmark:   out = [e for e in out if e.get("benchmark")   == benchmark]
    if num_agents:  out = [e for e in out if e.get("num_agents")  == num_agents]
    if task_family: out = [e for e in out if e.get("task_family") == task_family]
    if difficulty:  out = [e for e in out if e.get("difficulty")  == difficulty]
    return out


# ────────────────────────────────────────────────────────────────
# Main extraction entry point
# ────────────────────────────────────────────────────────────────

ObservableDict = Dict[str, List[float]]


def extract_all_observables(
    data_root:   Path,
    topology:    Optional[str] = None,
    benchmark:   Optional[str] = None,
    num_agents:  Optional[int] = None,
    task_family: Optional[str] = None,
    difficulty:  Optional[str] = None,
) -> ObservableDict:
    """
    Extract all coordination observable distributions from data_root.
    Optionally filter by any run dimension.

    TCE is now filter-aware (uses filtered events, not raw file scan).
    All grouping keys are run-scoped to prevent cross-run collisions.

    Returns dict mapping observable name → List[float] of sizes.
    """
    print(f"Loading events from {data_root} ...")
    all_events = _load_all_events(data_root)
    print(f"  Total events loaded: {len(all_events)}")

    filtered = filter_events(
        all_events,
        topology=topology,
        benchmark=benchmark,
        num_agents=num_agents,
        task_family=task_family,
        difficulty=difficulty,
    )
    print(f"  After filtering: {len(filtered)} events")

    obs: ObservableDict = {
        "delegation_cascade":  extract_delegation_cascades(filtered),
        "revision_wave":       extract_revision_waves(filtered),
        "contradiction_burst": extract_contradiction_bursts(filtered),
        "merge_fanin":         extract_merge_fanin(filtered),
        "tce_per_run":         extract_tce_per_run_from_events(filtered),  # filter-aware
        "influence_per_agent": extract_influence_per_agent(filtered),
    }

    for name, sizes in obs.items():
        arr = np.array(sizes)
        family = ("PRIMARY" if name in PRIMARY_OBSERVABLES
                  else "SECONDARY" if name in SECONDARY_OBSERVABLES
                  else "EXPLORATORY")
        if len(arr) > 0:
            print(f"  [{family:<11}] {name:<22}  n={len(arr):5d}  "
                  f"mean={arr.mean():.1f}  max={arr.max():.0f}")
        else:
            print(f"  [{family:<11}] {name:<22}  n=0  (no data)")

    return obs