"""
event_extraction/graph_builder.py
-----------------------------------
Reconstruct claim propagation and agent influence graphs from events.jsonl logs.

Produces four CSV tables per extraction run:
  claim_nodes.csv     — one row per unique claim_id
  claim_edges.csv     — directed parent → child propagation edges
  agent_edges.csv     — derived agent-to-agent influence edges (aggregated)
  run_graph_summary.csv — per-run graph statistics

Graph model
-----------
  Claim propagation graph:
    node = claim_id
    edge = parent_claim_id → claim_id   (directed)

  Agent influence graph (derived):
    node = agent_id
    edge = agent_of_parent_claim → agent_of_child_claim
    weight = number of propagation edges between the pair

Power-law observables computed directly from these graphs:
  out-degree    → reuse / fan-out per claim
  descendant count → cascade size
  in-degree     → merge fan-in
  agent out-degree → influence concentration (Lorenz / Gini input)

Design notes
------------
- Per-run separation is maintained: run_id on every node and edge.
  Never merge across runs before computing per-run observables.
- Dangling parent references become placeholder nodes (is_placeholder=True).
  If the real event arrives later, metadata is backfilled.
- Topology-specific sanity checks are printed but non-fatal.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────

def _iter_events_file(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def _first(events: List[Dict], key: str, default=None):
    for ev in events:
        v = ev.get(key)
        if v is not None:
            return v
    return default


# ─────────────────────────────────────────────────────────────────
# Per-run graph construction
# ─────────────────────────────────────────────────────────────────

def _make_placeholder_node(
    pid: str,
    run_id: str,
    benchmark: str,
    topology: str,
    n_agents: int,
) -> Dict:
    """Centralised placeholder schema — prevents drift if node schema changes."""
    return {
        "claim_id": pid, "run_id": run_id, "benchmark": benchmark,
        "topology": topology, "num_agents": n_agents,
        "agent_id": None, "agent_role": None, "step_id": None,
        "event_type": None, "claim_type": None, "claim_status": None,
        "root_claim_id": None, "claim_depth": None, "subtask_id": None,
        "subtask_depth": None, "revision_chain_id": None,
        "contradiction_group_id": None, "text_hash": None,
        "tokens_total_event": None,
        "is_placeholder": True,
        "placeholder_reason": "unseen_parent_reference",
    }


def _build_graph_for_run(events: List[Dict]) -> Tuple[list, list, list, dict]:
    """
    Build claim node dict, claim edge list, agent edge list, and
    run summary from one run's event list.

    Returns (claim_nodes_list, claim_edges_list, agent_edges_list, summary_dict)
    """
    claim_nodes: Dict[str, Dict] = {}
    claim_edges: List[Dict]       = []

    run_id    = _first(events, "run_id",    "unknown")
    benchmark = _first(events, "benchmark", "unknown")
    topology  = _first(events, "topology",  "unknown")
    n_agents  = _first(events, "num_agents", 0)

    # ── Build nodes and edges ──────────────────────────────────────
    # Single seen set for the whole run — deduplicates both parent and
    # merge_parent edges. Prevents degree/strength inflation from the
    # same (parent, child) pair appearing in multiple events.
    # Semantics: for power-law analysis, one propagation DAG edge is enough;
    # edge_type of the first occurrence wins (parent takes priority).
    seen_claim_edges: set = set()  # {(parent_claim_id, child_claim_id)}

    for ev in events:
        cid = ev.get("claim_id")
        if not cid:
            continue

        ev_type  = ev.get("event_type", "")
        # Skip endorsement events — they are support edges, not new DAG nodes
        if ev_type == "endorse_claim":
            continue

        # ── Upsert claim node ──────────────────────────────────────
        if cid not in claim_nodes:
            claim_nodes[cid] = {
                "claim_id":              cid,
                "run_id":                run_id,
                "benchmark":             benchmark,
                "topology":              topology,
                "num_agents":            n_agents,
                "agent_id":              ev.get("agent_id"),
                "agent_role":            ev.get("agent_role"),
                "step_id":               ev.get("step_id"),
                "event_type":            ev_type,
                "claim_type":            ev.get("claim_type"),
                "claim_status":          ev.get("claim_status"),
                "root_claim_id":         ev.get("root_claim_id"),
                "claim_depth":           ev.get("claim_depth"),
                "subtask_id":            ev.get("subtask_id"),
                "subtask_depth":         ev.get("subtask_depth"),
                "revision_chain_id":     ev.get("revision_chain_id"),
                "contradiction_group_id":ev.get("contradiction_group_id"),
                "text_hash":             ev.get("claim_text_hash"),
                "tokens_total_event":    ev.get("tokens_total_event"),
                "is_placeholder":        False,
            }
        else:
            # Backfill missing fields only (first complete event wins)
            node = claim_nodes[cid]
            for k, v in {
                "agent_id":          ev.get("agent_id"),
                "agent_role":        ev.get("agent_role"),
                "step_id":           ev.get("step_id"),
                "event_type":        ev_type,
                "claim_type":        ev.get("claim_type"),
                "claim_status":      ev.get("claim_status"),
                "root_claim_id":     ev.get("root_claim_id"),
                "claim_depth":       ev.get("claim_depth"),
                "text_hash":         ev.get("claim_text_hash"),
                "tokens_total_event":ev.get("tokens_total_event"),
            }.items():
                if node.get(k) in (None, "", []):
                    node[k] = v

        # ── Build parent edges ─────────────────────────────────────
        parents = ev.get("parent_claim_ids") or []
        for pid in parents:
            if not pid or pid == cid:
                continue
            if pid not in claim_nodes:
                claim_nodes[pid] = _make_placeholder_node(
                    pid, run_id, benchmark, topology, n_agents
                )
            if (pid, cid) not in seen_claim_edges:
                seen_claim_edges.add((pid, cid))
                claim_edges.append({
                    "run_id":          run_id,
                    "benchmark":       benchmark,
                    "topology":        topology,
                    "num_agents":      n_agents,
                    "parent_claim_id": pid,
                    "child_claim_id":  cid,
                    "parent_agent_id": claim_nodes[pid].get("agent_id"),
                    "child_agent_id":  ev.get("agent_id"),
                    "step_id":         ev.get("step_id"),
                    "child_event_type":ev_type,
                    "root_claim_id":   ev.get("root_claim_id"),
                    "edge_type":       "parent",
                })

        # Also handle merge_parent_claim_ids — reuse run-level seen_claim_edges
        # merge_parent edge_type is preserved only when not already a parent edge
        for pid in (ev.get("merge_parent_claim_ids") or []):
            if not pid or pid == cid:
                continue
            if pid not in claim_nodes:
                claim_nodes[pid] = _make_placeholder_node(
                    pid, run_id, benchmark, topology, n_agents
                )
            if (pid, cid) not in seen_claim_edges:
                seen_claim_edges.add((pid, cid))
                claim_edges.append({
                    "run_id":          run_id,
                    "benchmark":       benchmark,
                    "topology":        topology,
                    "num_agents":      n_agents,
                    "parent_claim_id": pid,
                    "child_claim_id":  cid,
                    "parent_agent_id": claim_nodes[pid].get("agent_id"),
                    "child_agent_id":  ev.get("agent_id"),
                    "step_id":         ev.get("step_id"),
                    "child_event_type":ev_type,
                    "root_claim_id":   ev.get("root_claim_id"),
                    "edge_type":       "merge_parent",
                })

    # ── Compute degree info and derived node properties ────────────
    indeg:  Dict[str, int] = defaultdict(int)
    outdeg: Dict[str, int] = defaultdict(int)

    for e in claim_edges:
        outdeg[e["parent_claim_id"]] += 1
        indeg[e["child_claim_id"]]   += 1

    for cid, node in claim_nodes.items():
        node["in_degree"]     = indeg.get(cid, 0)
        node["out_degree"]    = outdeg.get(cid, 0)
        node["is_root_claim"] = (indeg.get(cid, 0) == 0 and not node["is_placeholder"])
        node["is_leaf_claim"] = (outdeg.get(cid, 0) == 0)
        node["is_final_claim"]= (node.get("claim_type") == "final_claim")

    # Backfill parent_agent_id in edges now that degrees are computed
    cid_to_agent = {cid: n.get("agent_id") for cid, n in claim_nodes.items()}
    for e in claim_edges:
        if e["parent_agent_id"] is None:
            e["parent_agent_id"] = cid_to_agent.get(e["parent_claim_id"])

    # ── Derive agent-agent influence edges ────────────────────────
    agent_weights: Dict[Tuple[str, str], int] = defaultdict(int)
    for e in claim_edges:
        src = e.get("parent_agent_id")
        dst = e.get("child_agent_id")
        if src and dst:
            agent_weights[(src, dst)] += 1

    agent_edges = [
        {
            "run_id":            run_id,
            "benchmark":         benchmark,
            "topology":          topology,
            "num_agents":        n_agents,
            "source_agent_id":   src,
            "target_agent_id":   dst,
            "weight":            w,
            "interaction_type":  "claim_propagation",
        }
        for (src, dst), w in agent_weights.items()
    ]

    # ── Run summary ───────────────────────────────────────────────
    nodes_list = list(claim_nodes.values())
    depths     = [n["claim_depth"] for n in nodes_list if n.get("claim_depth") is not None]
    summary = {
        "run_id":                  run_id,
        "benchmark":               benchmark,
        "topology":                topology,
        "num_agents":              n_agents,
        "num_events":              len(events),
        "num_claim_nodes":         len(nodes_list),
        "num_claim_edges":         len(claim_edges),
        "num_agent_edges":         len(agent_edges),
        "num_root_claims":         sum(1 for n in nodes_list if n["is_root_claim"]),
        "num_final_claims":        sum(1 for n in nodes_list if n["is_final_claim"]),
        "num_placeholder_claims":  sum(1 for n in nodes_list if n["is_placeholder"]),
        "max_out_degree":          max((n["out_degree"] for n in nodes_list), default=0),
        "max_in_degree":           max((n["in_degree"]  for n in nodes_list), default=0),
        "max_claim_depth":         max(depths, default=0),
        "mean_claim_depth":        round(sum(depths) / len(depths), 3) if depths else 0,
    }

    return nodes_list, claim_edges, agent_edges, summary


# ─────────────────────────────────────────────────────────────────
# Sanity checks
# ─────────────────────────────────────────────────────────────────

def _sanity_check(
    claim_nodes: list,
    claim_edges: list,
    summary: dict,
    verbose: bool = False,
) -> List[str]:
    """
    Run topology-specific invariant checks.
    Returns list of warning strings (non-fatal).
    """
    warnings = []
    topo = summary.get("topology", "")
    n_nodes = len(claim_nodes)
    n_edges = len(claim_edges)

    node_ids = {n["claim_id"] for n in claim_nodes}
    for e in claim_edges:
        if e["parent_claim_id"] not in node_ids:
            warnings.append(f"  WARN: edge parent {e['parent_claim_id'][:12]} not in nodes")
        if e["child_claim_id"] not in node_ids:
            warnings.append(f"  WARN: edge child {e['child_claim_id'][:12]} not in nodes")

    n_roots  = summary.get("num_root_claims", 0)
    n_finals = summary.get("num_final_claims", 0)

    if n_finals > 0 and n_roots == 0:
        warnings.append("  WARN: final claims exist but no root claims detected")
    if n_nodes > 0 and n_edges == 0:
        warnings.append("  WARN: claim nodes exist but no edges — parent_claim_ids may be empty")

    # Topology-specific
    if "chain" in topo:
        over_fan = [n for n in claim_nodes if n.get("in_degree", 0) > 1 and not n["is_placeholder"]]
        if len(over_fan) > n_nodes * 0.3:
            warnings.append(f"  WARN: chain topology has {len(over_fan)} nodes with in_degree>1 (>30%)")

    if "tree" in topo or "star" in topo:
        merge_nodes = [n for n in claim_nodes if n.get("in_degree", 0) >= 2]
        if not merge_nodes:
            warnings.append(f"  WARN: {topo} topology has no merge fan-in (all in_degree < 2)")

    return warnings


# ─────────────────────────────────────────────────────────────────
# Main extraction function
# ─────────────────────────────────────────────────────────────────

def extract_graph_rows(
    roots:            List[Path],
    topology_filter:  Optional[str] = None,
    benchmark_filter: Optional[str] = None,
    n_filter:         Optional[int] = None,
    verbose:          bool = False,
) -> Tuple[list, list, list, list]:
    """
    Walk all events.jsonl files, reconstruct per-run claim graphs,
    and return flat tables ready for CSV export.

    Returns
    -------
    (claim_nodes, claim_edges, agent_edges, run_graph_summary)
    Each is a list of dicts.
    """
    print("\n── Graph reconstruction ─────────────────────────────────────")

    events_by_run: Dict[str, List[Dict]] = defaultdict(list)
    file_count = 0

    for root in roots:
        if not root.exists():
            print(f"  SKIP (not found): {root}")
            continue
        for ep in sorted(root.rglob("events.jsonl")):
            file_count += 1
            for ev in _iter_events_file(ep):
                # Apply filters
                if topology_filter  and ev.get("topology")   != topology_filter:  continue
                if benchmark_filter and ev.get("benchmark")  != benchmark_filter: continue
                if n_filter         and ev.get("num_agents") != n_filter:         continue
                run_id = ev.get("run_id")
                if run_id:
                    events_by_run[run_id].append(ev)

    print(f"  Files scanned:  {file_count}")
    print(f"  Runs found:     {len(events_by_run)}")

    all_claim_nodes:       list = []
    all_claim_edges:       list = []
    all_agent_edges:       list = []
    all_run_graph_summary: list = []

    warn_count = 0
    for run_id, events in sorted(events_by_run.items()):
        nodes, edges, agent_edges, summary = _build_graph_for_run(events)
        warnings = _sanity_check(nodes, edges, summary, verbose=verbose)
        warn_count += len(warnings)
        if warnings and verbose:
            print(f"  {run_id}:")
            for w in warnings:
                print(w)
        all_claim_nodes.extend(nodes)
        all_claim_edges.extend(edges)
        all_agent_edges.extend(agent_edges)
        all_run_graph_summary.append(summary)

    # Print aggregate stats
    n_nodes    = len(all_claim_nodes)
    n_edges    = len(all_claim_edges)
    n_runs     = len(all_run_graph_summary)
    n_roots    = sum(s.get("num_root_claims", 0)    for s in all_run_graph_summary)
    n_finals   = sum(s.get("num_final_claims", 0)   for s in all_run_graph_summary)
    n_holders  = sum(s.get("num_placeholder_claims", 0) for s in all_run_graph_summary)
    max_indeg  = max((s.get("max_in_degree", 0)     for s in all_run_graph_summary), default=0)
    max_outdeg = max((s.get("max_out_degree", 0)    for s in all_run_graph_summary), default=0)

    print(f"\n  Claim nodes:      {n_nodes:8,d}")
    print(f"  Claim edges:      {n_edges:8,d}")
    print(f"  Agent edges:      {len(all_agent_edges):8,d}")
    print(f"  Root claims:      {n_roots:8,d}")
    print(f"  Final claims:     {n_finals:8,d}")
    print(f"  Placeholder nodes:{n_holders:8,d}")
    print(f"  Max in-degree:    {max_indeg:8,d}  (merge fan-in)")
    print(f"  Max out-degree:   {max_outdeg:8,d}  (claim reuse)")
    if warn_count:
        print(f"  Sanity warnings:  {warn_count}  (run with --verbose to see)")

    return all_claim_nodes, all_claim_edges, all_agent_edges, all_run_graph_summary


# ─────────────────────────────────────────────────────────────────
# Graph-derived H1 observables
# (these complement the event-level observables in coordination.py)
# ─────────────────────────────────────────────────────────────────

def graph_observables_from_tables(
    claim_nodes: list,
    claim_edges: list,
) -> Dict[str, List[float]]:
    """
    Compute H1 power-law observables from the reconstructed claim DAG.

    BFS traversals are performed PER RUN to avoid cross-run contamination.
    Even if claim IDs are globally unique, per-run grouping is more robust.

    Observable definitions (strongest → weakest for paper claims)
    ─────────────────────────────────────────────────────────────
    PRIMARY (strongest — descendant-based):
      cascade_size            reachable descendants per root claim (BFS per run)
      claim_descendant_count  reachable descendants per non-placeholder claim
      agent_descendant_influence  cumulative sum of descendant counts per agent
                                  (may double-count overlapping downstream regions)
      agent_unique_descendant_reach  union of all descendant sets per agent
                                     (each downstream claim counted exactly once)
                                     — stronger metric for "few players steer many"

    SECONDARY (degree-based):
      claim_out_degree    immediate fan-out per claim (direct reuse)
      claim_in_degree     immediate fan-in per claim (direct parents)
      agent_out_strength  total outgoing propagation edges per agent (weighted)
      agent_out_degree    distinct downstream agents reached per agent (unique)

    SUPPORTING:
      merge_fan_in        in-degree for merge_claims events only
      claim_depth         depth in DAG
    """
    from collections import deque

    obs: Dict[str, List[float]] = {
        # Primary
        "cascade_size":                    [],
        "claim_descendant_count":          [],
        "agent_descendant_influence":      [],  # cumulative (may double-count)
        "agent_unique_descendant_reach":   [],  # deduplicated (union of desc sets)
        # Secondary
        "claim_out_degree":           [],
        "claim_in_degree":            [],
        "agent_out_strength":         [],
        "agent_out_degree":           [],
        # Supporting
        "merge_fan_in":               [],
        "claim_depth":                [],
    }

    # ── Group nodes and edges by run_id ───────────────────────────────────────
    runs_nodes: Dict[str, list] = defaultdict(list)
    for n in claim_nodes:
        rid = n.get("run_id", "__unknown__")
        runs_nodes[rid].append(n)

    runs_edges: Dict[str, list] = defaultdict(list)
    for e in claim_edges:
        rid = e.get("run_id", "__unknown__")
        runs_edges[rid].append(e)

    # ── Per-run graph traversal ───────────────────────────────────────────────
    for run_id in runs_nodes:
        r_nodes = runs_nodes[run_id]
        r_edges = runs_edges.get(run_id, [])

        # Degree observables from node attributes (already computed in build phase)
        for n in r_nodes:
            if n.get("is_placeholder"):
                continue
            od  = n.get("out_degree", 0)
            id_ = n.get("in_degree", 0)
            if od > 0:
                obs["claim_out_degree"].append(float(od))
            if id_ > 0:
                obs["claim_in_degree"].append(float(id_))
            if n.get("claim_depth") is not None and n["claim_depth"] > 0:
                obs["claim_depth"].append(float(n["claim_depth"]))
            if n.get("event_type") == "merge_claims" and id_ >= 2:
                obs["merge_fan_in"].append(float(id_))

        # agent_out_strength: total outgoing propagation edges (weighted).
        #   Counts every edge from this agent's claims, including to self.
        #   Best for Lorenz/Gini concentration analysis.
        # agent_out_degree: distinct *other* agents reached downstream.
        #   Self-propagation (agent cites their own prior claim) excluded.
        #   Best for social influence / information routing analysis.
        agent_out_strength: Dict[str, int] = defaultdict(int)
        agent_downstream_agents: Dict[str, set] = defaultdict(set)
        for e in r_edges:
            src = e.get("parent_agent_id")
            dst_agent = e.get("child_agent_id")
            if src:
                agent_out_strength[src] += 1
                # Exclude self-links: self-propagation is not social influence
                if dst_agent and dst_agent != src:
                    agent_downstream_agents[src].add(dst_agent)
        for agent_id, strength in agent_out_strength.items():
            if strength > 0:
                obs["agent_out_strength"].append(float(strength))
        for agent_id, targets in agent_downstream_agents.items():
            if targets:
                obs["agent_out_degree"].append(float(len(targets)))

        # Build adjacency for BFS: parent → set(children)
        # Use set to prevent duplicate traversal from repeated edges
        children_of: Dict[str, set] = defaultdict(set)
        for e in r_edges:
            children_of[e["parent_claim_id"]].add(e["child_claim_id"])

        # BFS to compute descendant count for every non-placeholder node
        # Returns (visited_set, count) so callers can reuse the set for
        # unique-reach calculations without a second traversal.
        def _bfs_descendants(start_id: str):
            visited: set = set()
            queue = deque([start_id])
            while queue:
                cur = queue.popleft()
                if cur in visited:
                    continue
                visited.add(cur)
                for child in children_of.get(cur, set()):
                    if child not in visited:
                        queue.append(child)
            visited.discard(start_id)  # exclude start node itself
            return visited

        # ── BFS from every non-placeholder claim ────────────────────────────
        # claim_desc_sets: claim_id → frozenset of all reachable descendants
        # Used for both count-based and unique-reach observables.
        claim_desc_sets: Dict[str, frozenset] = {}
        for n in r_nodes:
            if n.get("is_placeholder"):
                continue
            cid  = n["claim_id"]
            desc_set = _bfs_descendants(cid)
            claim_desc_sets[cid] = frozenset(desc_set)
            desc_count = len(desc_set)
            if desc_count > 0:
                obs["claim_descendant_count"].append(float(desc_count))

        # cascade_size: descendant count for root claims only
        for n in r_nodes:
            if n.get("is_root_claim") and not n.get("is_placeholder"):
                cascade = len(claim_desc_sets.get(n["claim_id"], frozenset()))
                if cascade > 0:
                    obs["cascade_size"].append(float(cascade))

        # agent_descendant_influence:
        #   CUMULATIVE — sum of descendant counts across all claims by this agent.
        #   If the agent authored multiple ancestor claims in overlapping regions,
        #   the same downstream claims may be counted multiple times.
        #   Interpretation: cumulative structural influence mass, not unique reach.
        #   Use agent_unique_descendant_reach for deduplicated influence.
        agent_claim_desc_sets: Dict[str, List[frozenset]] = defaultdict(list)
        for n in r_nodes:
            if n.get("is_placeholder"):
                continue
            agent_id = n.get("agent_id")
            if agent_id and n["claim_id"] in claim_desc_sets:
                agent_claim_desc_sets[agent_id].append(claim_desc_sets[n["claim_id"]])

        for agent_id, desc_sets in agent_claim_desc_sets.items():
            # Cumulative influence (sum — may double-count overlapping regions)
            cumulative = sum(len(s) for s in desc_sets)
            if cumulative > 0:
                obs["agent_descendant_influence"].append(float(cumulative))
            # Unique reach (union — counts each downstream claim exactly once)
            unique_reach = len(frozenset().union(*desc_sets)) if desc_sets else 0
            if unique_reach > 0:
                obs["agent_unique_descendant_reach"].append(float(unique_reach))

    return obs