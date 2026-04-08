"""
src/event_extraction/event_extractor.py

Post-hoc event type inference from structured traces.
Lives in src/event_extraction/ alongside coordination.py, graph_builder.py, tce.py.

Classification rules (priority order):
  1. delegate_subtask   new subtask_id + valid parent + topology/role allows it
  2. merge_claims       STRUCTURAL — parent_claim_ids >= 2 (no semantic gate)
  3. contradict_claim   STRICT — explicit signal OR strong contradiction keywords only
  4. revise_claim       exactly 1 parent + correction signal or keywords
  5. endorse_claim      supports_claims non-empty, no correction
  6. propose_claim      default

Design philosophy:
  - Merge is structural, not semantic. Any claim with >=2 parents is a merge.
    Requiring synthesis language would undercount merges and distort tail distributions.
  - Contradiction is strict. "but", "however", "instead" are excluded from
    correction patterns — they appear in non-contradictory revisions and would
    inflate false contradictions, which inflates heavy tails incorrectly.
  - False negatives are acceptable. False positives break the DAG.
  - Delegation requires explicit assignment signal, not just a new subtask_id,
    to prevent misclassification of reused or delayed subtask IDs.

Revision chain propagation uses parent-based inheritance as primary method,
with forward frontier propagation as secondary pass for cross-row chains.

claim_status is derived deterministically from event_type:
  propose_claim   → proposed
  revise_claim    → revised
  contradict_claim→ contradicted
  merge_claims    → merged
  endorse_claim   → endorsed
  delegate_subtask→ None (work event, not a claim status)
"""

from __future__ import annotations

import re
import uuid
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from loggers.trace_schema import EventType, ClaimStatus


# ── Event type constants ──────────────────────────────────────

DELEGATE   : EventType = "delegate_subtask"
MERGE      : EventType = "merge_claims"
CONTRADICT : EventType = "contradict_claim"
REVISE     : EventType = "revise_claim"
ENDORSE    : EventType = "endorse_claim"
PROPOSE    : EventType = "propose_claim"

EVENT_TO_CLAIM_STATUS: Dict[str, Optional[ClaimStatus]] = {
    PROPOSE:    "proposed",
    REVISE:     "revised",
    CONTRADICT: "contradicted",
    MERGE:      "merged",
    ENDORSE:    "endorsed",
    DELEGATE:   None,
}

# ── Regex patterns ────────────────────────────────────────────
#
# Correction: excludes "but", "however", "instead" — these appear in
# non-contradictory revisions and would cause false contradictions if
# shared across both patterns.
#
# Contradiction: strong adversarial language only. Conservative by design.

_CORRECTION_RE = re.compile(
    r"\b(incorrect|wrong|error|mistake|fix|correct(?:ion)?|should be|actually|disagree)\b",
    re.IGNORECASE,
)

_CONTRADICTION_RE = re.compile(
    r"\b(contradict|conflict|incompatible|oppose|reject|"
    r"cannot be both|mutually exclusive|fundamentally disagree)\b",
    re.IGNORECASE,
)

DELEGATION_TOPOLOGIES = {"star", "tree", "hybrid_modular"}
DELEGATION_ROLES      = {"hub", "supervisor", "coordinator", "bridge"}


# ── Core classifier ───────────────────────────────────────────

def infer_event_type(
    row: dict,
    known_subtask_ids: Set[str],
    topology_allows_delegation: bool,
) -> EventType:
    # parent_claim_ids = topology-computed visible claim IDs (same as claims_visible in TraceRow)
    # claims_referenced = LLM-reported claim IDs (what agent chose to cite — may be subset)
    parent_claim_ids: List[str] = row.get("parent_claim_ids", []) or row.get("claims_visible", []) or []
    subtask_id: str             = row.get("subtask_id", "") or ""
    parent_subtask_id           = row.get("parent_subtask_id")
    signals: dict               = row.get("coordination_signals") or {}
    reasoning: str              = row.get("reasoning_text", "") or ""

    # 1. delegate_subtask
    # Requires: new subtask_id, valid parent_subtask_id, topology/role permits it,
    # AND an explicit assignment signal to prevent misclassification of reused IDs.
    if (
        subtask_id
        and subtask_id not in known_subtask_ids
        and parent_subtask_id
        and topology_allows_delegation
        and (
            signals.get("requested_subtask_creation")
            or signals.get("proposed_assignee") is not None
            or row.get("assigned_agent") is not None
        )
    ):
        return DELEGATE

    # 2. merge_claims — STRUCTURAL, no semantic gate
    # Any claim referencing >=2 parents is a merge by definition.
    # Synthesis language is not required: requiring it would undercount merges
    # and distort the merge fan-in tail distribution.
    if len(parent_claim_ids) >= 2:
        return MERGE

    # 3. contradict_claim — STRICT
    # Only explicit signal or strong adversarial language.
    # "but", "however", "instead" deliberately excluded — they appear in revisions.
    if signals.get("explicit_disagreement_with"):
        return CONTRADICT
    if len(parent_claim_ids) >= 1 and _CONTRADICTION_RE.search(reasoning):
        return CONTRADICT

    # 4. revise_claim
    if len(parent_claim_ids) == 1:
        if signals.get("explicit_correction_of") or _CORRECTION_RE.search(reasoning):
            return REVISE

    # 5. endorse_claim
    if signals.get("supports_claims") and len(parent_claim_ids) >= 1:
        return ENDORSE

    # 6. propose_claim (default)
    return PROPOSE


# ── Batch annotator ───────────────────────────────────────────

def annotate_event_types(rows: List[dict]) -> List[dict]:
    """
    Annotate TraceRow dicts (sorted by step_id, timestamp) with:
      event_type, claim_status, revision_chain_id,
      contradiction_group_id, merge_id.
    Mutates rows in place. Returns the sorted list.
    """
    rows = sorted(rows, key=lambda r: (r.get("step_id", 0), r.get("timestamp", 0)))
    known_subtask_ids: Set[str] = set()

    for row in rows:
        topology = row.get("topology", "")
        role     = row.get("role", "")
        allows   = topology in DELEGATION_TOPOLOGIES or role in DELEGATION_ROLES

        event_type          = infer_event_type(row, known_subtask_ids, allows)
        row["event_type"]   = event_type
        row["claim_status"] = EVENT_TO_CLAIM_STATUS[event_type]

        if row.get("subtask_id"):
            known_subtask_ids.add(row["subtask_id"])

    _assign_revision_chains(rows)
    _assign_contradiction_groups(rows, window_steps=3)
    _assign_merge_ids(rows)
    return rows


# ── Grouping ID assignment ────────────────────────────────────

def _assign_revision_chains(rows: List[dict]) -> None:
    """
    Assign revision_chain_id to all revise_claim rows.

    Primary method: parent-based inheritance.
      If a revision's parent already has a revision_chain_id, inherit it.
      This correctly handles branching chains and non-contiguous rows.

    Secondary method: forward frontier propagation.
      For chains that start mid-run, propagate the new chain_id forward
      to any unassigned revision that references a claim already in the chain.
    """
    claim_to_idx: Dict[str, int] = {
        row["claim_id"]: i
        for i, row in enumerate(rows)
        if row.get("claim_id")
    }

    for i, row in enumerate(rows):
        if row.get("event_type") != REVISE or row.get("revision_chain_id"):
            continue
        parent_ids = row.get("parent_claim_ids", [])
        if not parent_ids:
            continue

        # Primary: inherit from parent
        parent_idx = claim_to_idx.get(parent_ids[0])
        if parent_idx is not None and rows[parent_idx].get("revision_chain_id"):
            row["revision_chain_id"] = rows[parent_idx]["revision_chain_id"]
            continue

        # Secondary: start a new chain and propagate forward
        chain_id = f"rev_{uuid.uuid4().hex[:8]}"
        row["revision_chain_id"] = chain_id
        frontier: Set[str] = {row["claim_id"]}
        for j in range(i + 1, len(rows)):
            r = rows[j]
            if r.get("event_type") != REVISE or r.get("revision_chain_id"):
                continue
            if set(r.get("parent_claim_ids", [])) & frontier:
                r["revision_chain_id"] = chain_id
                frontier.add(r["claim_id"])


def _assign_contradiction_groups(rows: List[dict], window_steps: int = 3) -> None:
    """
    Group contradictory claims targeting the same parent within a temporal window.
    Burst size (distinct agents per group) is computed downstream in cascade_metrics,
    where agent_id deduplication happens over each group's rows.
    """
    contra_by_parent: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    for i, row in enumerate(rows):
        if row.get("event_type") == CONTRADICT:
            for pid in row.get("parent_claim_ids", []):
                contra_by_parent[pid].append((row.get("step_id", 0), i))

    for _pid, entries in contra_by_parent.items():
        entries.sort(key=lambda x: x[0])
        groups: List[List[int]] = []
        current: List[int] = []
        last_step: Optional[int] = None
        for step, idx in entries:
            if last_step is None or (step - last_step) <= window_steps:
                current.append(idx)
            else:
                if current:
                    groups.append(current)
                current = [idx]
            last_step = step
        if current:
            groups.append(current)
        for group in groups:
            gid = f"con_{uuid.uuid4().hex[:8]}"
            for idx in group:
                rows[idx]["contradiction_group_id"] = gid


def _assign_merge_ids(rows: List[dict]) -> None:
    for row in rows:
        if row.get("event_type") == MERGE:
            row["merge_id"] = f"merge_{uuid.uuid4().hex[:8]}"