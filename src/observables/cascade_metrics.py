"""
src/observables/cascade_metrics.py

Extracts all 5 power-law observables + agent influence metrics from
annotated trace rows and cascades produced by dag_builder.

Lives in src/observables/ alongside dag_builder.py.

Design decisions (explicit):

  delegation_subtree_sizes:
    One sample per delegate_subtask event, sized as the subtask subtree
    rooted at that event's subtask_id. If multiple delegations occur along
    the same branch, subtrees overlap. This is intentional: each delegation
    event is a distinct observable sample for the power-law distribution,
    consistent with the paper treating delegation cascade size as an
    event-level quantity. If you want non-overlapping cascades, deduplicate
    by taking only the topmost delegation per branch (not done here).

  revision_wave_sizes:
    Wave size = number of revise_claim events in the chain, NOT including
    the original proposed claim. Rationale: we are measuring coordination
    effort, and the root propose is a baseline, not a coordination event.
    This matches the TCE definition (which also excludes the root).
    Comment this explicitly so figures/tables can state it clearly.

  cascade weighting in agent influence:
    Weighted by cascade.tce (downstream coordination events), not cascade.size
    (raw claim count). Rationale: an agent in a large but low-activity claim
    structure should not receive the same weight as one whose claims trigger
    genuine coordination cascades. TCE better reflects coordination effort.

  extract_all_observables return structure:
    Separated into "event_observables" (scalar sample lists, for power-law
    fitting) and "agent_metrics" (per-agent dicts, for elite/concentration
    analysis). Keeps downstream analysis code semantically clean.

  root cascade initiation:
    Detected by claim_id == root_claim_id on each row, after dag_builder has
    written root_claim_id back into rows. Cleaner than set membership check.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

from observables.dag_builder import Cascade, SubtaskNode
from metrics.inequality import gini, top_k_share, effective_n


# ── Main entry point ──────────────────────────────────────────

def extract_all_observables(
    rows: List[dict],
    subtask_tree: Dict[str, SubtaskNode],
    cascades: List[Cascade],
) -> dict:
    """
    Compute all observables for one run.

    Returns:
      {
        "event_observables": {
            "delegation_sizes":     List[int],
            "revision_waves":       List[int],
            "contradiction_bursts": List[int],
            "merge_fan_in":         List[int],
            "tce":                  List[int],
        },
        "agent_metrics": {
            agent_id: {
                "raw_events":                int,
                "cascade_tce_weighted":      float,
                "root_cascades_initiated":   int,
            },
            ...
        }
      }
    """
    return {
        "event_observables": {
            "delegation_sizes":     delegation_subtree_sizes(subtask_tree, rows),
            "revision_waves":       revision_wave_sizes(rows),
            "contradiction_bursts": contradiction_burst_sizes(rows),
            "merge_fan_in":         merge_fan_in_sizes(rows),
            "tce":                  cascade_tce_sizes(cascades),
        },
        "agent_metrics": compute_agent_influence(rows, cascades),
    }


# ── 1. Delegation cascade size ────────────────────────────────

def delegation_subtree_sizes(
    subtask_tree: Dict[str, SubtaskNode],
    rows: List[dict],
) -> List[int]:
    """
    One sample per delegate_subtask event.
    Size = number of nodes in the subtask subtree rooted at that event's subtask_id.

    Note: if multiple delegations occur along the same branch, subtrees overlap.
    Each event is still a valid independent sample for the size distribution.
    """
    delegation_roots = {
        row["subtask_id"]
        for row in rows
        if row.get("event_type") == "delegate_subtask" and row.get("subtask_id")
    }
    return [
        s
        for s in (_subtask_subtree_size(sid, subtask_tree) for sid in delegation_roots)
        if s > 0
    ]


def _subtask_subtree_size(root_sid: str, tree: Dict[str, SubtaskNode]) -> int:
    if root_sid not in tree:
        return 0
    count, visited, queue = 0, set(), [root_sid]
    while queue:
        sid = queue.pop()
        if sid in visited:
            continue
        visited.add(sid)
        count += 1
        node = tree.get(sid)
        if node:
            queue.extend(node.children)
    return count


# ── 2. Revision wave size ─────────────────────────────────────

def revision_wave_sizes(rows: List[dict]) -> List[int]:
    """
    Wave size = number of revise_claim events per revision_chain_id.
    The original proposed claim is NOT included (consistent with TCE definition).
    Paper figures/tables should state: "revision wave size excludes root claim."
    """
    chain_counts: Dict[str, int] = defaultdict(int)
    for row in rows:
        cid = row.get("revision_chain_id")
        if cid and row.get("event_type") == "revise_claim":
            chain_counts[cid] += 1
    return [s for s in chain_counts.values() if s > 0]


# ── 3. Contradiction burst size ───────────────────────────────

def contradiction_burst_sizes(rows: List[dict]) -> List[int]:
    """
    Burst size = number of distinct agents contradicting the same parent claim
    within the temporal window defined in event_extractor (window_steps=3).
    """
    group_agents: Dict[str, set] = defaultdict(set)
    for row in rows:
        gid = row.get("contradiction_group_id")
        if gid and row.get("event_type") == "contradict_claim":
            aid = row.get("agent_id", "")
            if aid:
                group_agents[gid].add(aid)
    return [len(agents) for agents in group_agents.values() if agents]


# ── 4. Merge fan-in ───────────────────────────────────────────

def merge_fan_in_sizes(rows: List[dict]) -> List[int]:
    """Fan-in = number of parent claims per merge event."""
    return [
        len(row.get("parent_claim_ids", []))
        for row in rows
        if row.get("event_type") == "merge_claims"
        and len(row.get("parent_claim_ids", [])) >= 2
    ]


# ── 5. Total Cognitive Effort (TCE) ──────────────────────────

def cascade_tce_sizes(cascades: List[Cascade]) -> List[int]:
    """TCE per root-centered cascade (excludes root node, see dag_builder)."""
    return [c.tce for c in cascades if c.tce > 0]


# ── 6. Agent influence ────────────────────────────────────────

def compute_agent_influence(
    rows: List[dict],
    cascades: List[Cascade],
) -> Dict[str, dict]:
    """
    Per-agent influence statistics for one run.

    cascade_tce_weighted: each event weighted by the TCE of its cascade,
      not by raw cascade size. Rationale: TCE measures coordination effort;
      an agent whose claims trigger high-TCE cascades has higher real influence
      than one in a large but quiet claim structure.

    root_cascades_initiated: detected by claim_id == root_claim_id on each
      row, which is valid after dag_builder has written root_claim_id back.

    These are ingredients. Downstream callers use metrics.inequality for
    Gini, top-k share, and effective N over cascade_tce_weighted values.
    """
    # Raw event count
    raw_counts: Dict[str, int] = defaultdict(int)
    for row in rows:
        aid = row.get("agent_id", "")
        if aid:
            raw_counts[aid] += 1

    # TCE-weighted influence: weight = cascade.tce, not cascade.size
    claim_to_tce_weight: Dict[str, int] = {
        cid: cascade.tce
        for cascade in cascades
        for cid in cascade.claim_ids
    }
    tce_weighted: Dict[str, float] = defaultdict(float)
    for row in rows:
        aid = row.get("agent_id", "")
        cid = row.get("claim_id", "")
        if aid and cid:
            tce_weighted[aid] += claim_to_tce_weight.get(cid, 0)

    # Root cascade initiations: clean check via written-back root_claim_id
    root_initiations: Dict[str, int] = defaultdict(int)
    for row in rows:
        if row.get("claim_id") and row.get("claim_id") == row.get("root_claim_id"):
            aid = row.get("agent_id", "")
            if aid:
                root_initiations[aid] += 1

    all_agents = set(raw_counts) | set(tce_weighted) | set(root_initiations)
    return {
        agent: {
            "raw_events":              raw_counts.get(agent, 0),
            "cascade_tce_weighted":    round(tce_weighted.get(agent, 0.0), 4),
            "root_cascades_initiated": root_initiations.get(agent, 0),
        }
        for agent in all_agents
    }


# ── Concentration metrics (convenience wrappers) ──────────────

def influence_concentration(
    agent_metrics: Dict[str, dict],
    metric: str = "cascade_tce_weighted",
    top_k: int = 3,
) -> dict:
    """
    Compute concentration statistics over agent influence.

    Args:
        agent_metrics: output of compute_agent_influence()
        metric: which field to use — "cascade_tce_weighted" or "raw_events"
        top_k: k for top-k share

    Returns:
        gini, top_k_share, effective_n, active_agent_fraction
    """
    values = [m[metric] for m in agent_metrics.values() if m[metric] > 0]
    n_total = len(agent_metrics)
    n_active = len(values)

    return {
        "gini":                   gini(values),
        f"top_{top_k}_share":     top_k_share(values, top_k),
        "effective_n":            effective_n(values),
        "active_agent_fraction":  n_active / n_total if n_total > 0 else 0.0,
        "n_active":               n_active,
        "n_total":                n_total,
    }