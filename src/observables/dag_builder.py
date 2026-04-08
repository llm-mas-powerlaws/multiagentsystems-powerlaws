"""
src/observables/dag_builder.py

Builds subtask tree, claim DAG, and cascades from annotated trace rows.
Lives in src/observables/ alongside cascade_metrics.py.

Input:  annotated TraceRow dicts (event_type + claim_status already set
        by event_extraction/event_extractor.py)
Output: subtask_tree, claim_dag, cascades

Also writes back into each row dict in place:
  root_claim_id, secondary_root_claim_ids, claim_depth, subtask_depth

Key design decisions:
  - Root/depth assignment uses topological order (sorted by step_id), not BFS.
    This guarantees all parents are processed before their children, which is
    required for correctness in the presence of merge edges and cross-dependencies.
    BFS from roots can assign root/depth before all parents are resolved.
  - Stub nodes (created for parent references not yet seen) are marked with
    is_stub=True and excluded from cascade extraction. They exist only to
    keep the DAG structurally complete without crashing on ordering issues.
  - TCE excludes the root node explicitly by claim_id comparison, not by
    event_type. This is safer: it does not rely on the root always being
    propose_claim (which holds in practice but is an assumption).
  - Cascades are root-centered connected components, not temporal windows.
    Paper wording: "root-centered cascade", not "temporal cascade".

Downstream:
  - observables/cascade_metrics.py  power-law observables
  - event_extraction/coordination.py  reads event_type
  - event_extraction/tce.py           reads cascade TCE
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from loggers.trace_schema import EventType


@dataclass
class SubtaskNode:
    subtask_id: str
    parent_subtask_id: Optional[str]
    assigned_agent: Optional[str]
    children: List[str] = field(default_factory=list)
    depth: int = 0


@dataclass
class ClaimNode:
    claim_id: str
    agent_id: str
    step_id: int
    event_type: EventType
    parent_claim_ids: List[str]
    is_stub: bool = False                              # True = referenced but never emitted
    root_claim_id: Optional[str] = None               # assigned post hoc
    secondary_root_claim_ids: List[str] = field(default_factory=list)
    depth: int = 0
    children: List[str] = field(default_factory=list)
    revision_chain_id: Optional[str] = None
    contradiction_group_id: Optional[str] = None
    merge_id: Optional[str] = None


@dataclass
class Cascade:
    """
    Root-centered coordination cascade.
    All non-stub claims reachable from root_claim_id.
    The unit on which TCE and power-law observables are computed.
    """
    root_claim_id: str
    claim_ids: Set[str] = field(default_factory=set)
    event_types: List[EventType] = field(default_factory=list)
    size: int = 0
    tce: int = 0    # coordination events excluding the root node itself


# ── Main entry point ──────────────────────────────────────────

def build_all(rows: List[dict]) -> Tuple[
    Dict[str, SubtaskNode],
    Dict[str, ClaimNode],
    List[Cascade],
]:
    """
    Full pipeline: annotated trace rows → subtask tree, claim DAG, cascades.
    Writes root_claim_id, secondary_root_claim_ids, claim_depth, subtask_depth
    back into each row dict in place.
    """
    rows = sorted(rows, key=lambda r: (r.get("step_id", 0), r.get("timestamp", 0.0)))

    subtask_tree = _build_subtask_tree(rows)
    _assign_subtask_depths(subtask_tree)

    claim_dag = _build_claim_dag(rows)
    _assign_root_claim_ids_topological(claim_dag)

    # Write post-hoc fields back into rows
    for row in rows:
        cid = row.get("claim_id")
        if cid and cid in claim_dag:
            node = claim_dag[cid]
            row["root_claim_id"]            = node.root_claim_id
            row["secondary_root_claim_ids"] = node.secondary_root_claim_ids
            row["claim_depth"]              = node.depth
        sid = row.get("subtask_id")
        if sid and sid in subtask_tree:
            row["subtask_depth"] = subtask_tree[sid].depth

    cascades = _extract_cascades(claim_dag)
    return subtask_tree, claim_dag, cascades


# ── Subtask tree ──────────────────────────────────────────────

def _build_subtask_tree(rows: List[dict]) -> Dict[str, SubtaskNode]:
    tree: Dict[str, SubtaskNode] = {}
    for row in rows:
        sid = row.get("subtask_id")
        if not sid:
            continue
        if sid not in tree:
            tree[sid] = SubtaskNode(
                subtask_id=sid,
                parent_subtask_id=row.get("parent_subtask_id"),
                assigned_agent=row.get("assigned_agent") or row.get("agent_id"),
            )
        parent_sid = row.get("parent_subtask_id")
        if parent_sid and parent_sid != sid:
            if parent_sid not in tree:
                tree[parent_sid] = SubtaskNode(
                    subtask_id=parent_sid, parent_subtask_id=None, assigned_agent=None
                )
            if sid not in tree[parent_sid].children:
                tree[parent_sid].children.append(sid)
    return tree


def _assign_subtask_depths(tree: Dict[str, SubtaskNode]) -> None:
    from collections import deque
    roots = [
        n for n in tree.values()
        if not n.parent_subtask_id or n.parent_subtask_id not in tree
    ]
    queue: deque = deque()
    for root in roots:
        root.depth = 0
        queue.append(root.subtask_id)
    while queue:
        sid = queue.popleft()
        node = tree[sid]
        for child_id in node.children:
            if child_id in tree:
                tree[child_id].depth = node.depth + 1
                queue.append(child_id)


# ── Claim DAG ─────────────────────────────────────────────────

def _build_claim_dag(rows: List[dict]) -> Dict[str, ClaimNode]:
    dag: Dict[str, ClaimNode] = {}
    for row in rows:
        cid = row.get("claim_id")
        if not cid:
            continue
        parent_ids: List[str] = row.get("parent_claim_ids", []) or []

        # Self-loop guard
        if cid in parent_ids:
            parent_ids = [p for p in parent_ids if p != cid]

        node = ClaimNode(
            claim_id=cid,
            agent_id=row.get("agent_id", ""),
            step_id=row.get("step_id", 0),
            event_type=row.get("event_type", "propose_claim"),
            parent_claim_ids=parent_ids,
            is_stub=False,
            revision_chain_id=row.get("revision_chain_id"),
            contradiction_group_id=row.get("contradiction_group_id"),
            merge_id=row.get("merge_id"),
        )
        dag[cid] = node

        for pid in parent_ids:
            if pid not in dag:
                # Stub: parent referenced before its row appeared (ordering issue)
                dag[pid] = ClaimNode(
                    claim_id=pid, agent_id="", step_id=0,
                    event_type="propose_claim", parent_claim_ids=[],
                    is_stub=True,
                )
            if cid not in dag[pid].children:
                dag[pid].children.append(cid)

    return dag


# ── Root claim assignment (topological order) ─────────────────

def _assign_root_claim_ids_topological(dag: Dict[str, ClaimNode]) -> None:
    """
    Assign root_claim_id and depth to every node using topological ordering
    (sorted by step_id ascending). This guarantees all parents are resolved
    before each child is processed, which is required for correctness when
    merge edges introduce cross-dependencies.

    BFS from roots is NOT used here because it can process a child before
    all its parents have been assigned a root, producing incorrect depths
    and inconsistent root assignments on merge nodes.
    """
    nodes_sorted = sorted(dag.values(), key=lambda n: n.step_id)

    for node in nodes_sorted:
        parent_nodes = [dag[p] for p in node.parent_claim_ids if p in dag]
        parent_nodes = [p for p in parent_nodes if p.claim_id != node.claim_id]  # no self

        if not parent_nodes:
            # Root node (or stub with no resolvable parents)
            node.root_claim_id = node.claim_id
            node.depth = 0

        elif len(parent_nodes) == 1:
            parent = parent_nodes[0]
            node.root_claim_id = parent.root_claim_id or parent.claim_id
            node.depth = parent.depth + 1

        else:
            # Merge node: primary root = earliest-step parent
            primary = min(parent_nodes, key=lambda n: n.step_id)
            node.root_claim_id = primary.root_claim_id or primary.claim_id
            node.secondary_root_claim_ids = list({
                p.root_claim_id or p.claim_id
                for p in parent_nodes
                if (p.root_claim_id or p.claim_id) != node.root_claim_id
            })
            # Depth = max parent depth + 1 (all parents resolved before this node)
            node.depth = max(p.depth for p in parent_nodes) + 1


# ── Cascade extraction ────────────────────────────────────────

def _extract_cascades(dag: Dict[str, ClaimNode]) -> List[Cascade]:
    """
    Group non-stub nodes by root_claim_id to form cascades.

    Stubs are excluded: they have no real row, carry no reasoning content,
    and would corrupt cascade size and TCE counts.

    TCE excludes the root node itself by claim_id comparison (not by
    event_type), so it remains correct even if the root is not propose_claim.
    """
    root_to_nodes: Dict[str, List[ClaimNode]] = defaultdict(list)
    for node in dag.values():
        if node.is_stub:
            continue
        rid = node.root_claim_id or node.claim_id
        root_to_nodes[rid].append(node)

    cascades: List[Cascade] = []
    for root_id, nodes in root_to_nodes.items():
        claim_ids    = {n.claim_id for n in nodes}
        event_types  = [n.event_type for n in nodes]

        # TCE: coordination events, excluding the root node itself
        tce = sum(1 for n in nodes if n.claim_id != root_id)

        cascades.append(Cascade(
            root_claim_id=root_id,
            claim_ids=claim_ids,
            event_types=event_types,
            size=len(nodes),
            tce=tce,
        ))

    return sorted(cascades, key=lambda c: -c.size)