"""
tree.py  —  Hierarchical / Tree / Supervisor-Worker topology

Structure (example, 8 agents, B=2):
              supervisor
             /           \
        agent_000      agent_001      ← coordinators
         /    \           /    \
    agent_002 agent_003  agent_004 agent_005  ← workers

Execution order: post-order (leaves first, supervisor last).

Design (paper-faithful)
-----------------------
Topology owns:
  - claim_id, parent_claim_ids, subtask_id (structural lineage)
  - AgentContextSpec construction
  - _call_llm / _log_event calls
  - minimal structural state["claims"] / state["subtasks"]

Topology does NOT own:
  - event_type (except PROPOSE_CLAIM at supervisor root) → event_extractor.py
  - subtask_type, claim_type, claim_status enums → removed
  - merge_* / contradiction_* fields → dag_builder.py
  - "status" in state["claims"] → removed

Causal structure:
  - Leaf workers depend on their parent coordinator's claim.
    parent_claim_ids = [parent_cid] — NOT [] — so cascades propagate upward.
  - Coordinators (synthesis) merge child claim IDs.
  - Supervisor (root) merges coordinator claim IDs, emits PROPOSE_CLAIM hint.
  - Coordinator (decomposition) passes event_type=None; delegation logged
    separately via _log_event (zero-token, correct) not via _call_llm.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, END

from context_builder import build_context, AgentContextSpec
from .base import BaseTopology, MASState, make_agent_id, new_id, text_hash
from loggers.schemas import EventType, TopologyName

SUPERVISOR_ID    = "supervisor"
BRANCHING_FACTOR = 2


# ── Tree construction ──────────────────────────────────────────────────

def _build_tree(num_agents: int, branching_factor: int) -> Dict[str, List[str]]:
    if num_agents <= 1:
        return {SUPERVISOR_ID: []}
    children_map: Dict[str, List[str]] = {}
    all_ids  = [SUPERVISOR_ID] + [make_agent_id("agent", i) for i in range(num_agents - 1)]
    queue    = [SUPERVISOR_ID]
    assigned = 1
    while queue and assigned < len(all_ids):
        parent = queue.pop(0)
        children_map[parent] = []
        for _ in range(branching_factor):
            if assigned >= len(all_ids):
                break
            child = all_ids[assigned]
            children_map[parent].append(child)
            queue.append(child)
            assigned += 1
    for aid in all_ids:
        if aid not in children_map:
            children_map[aid] = []
    return children_map


# ── State helpers ──────────────────────────────────────────────────────

def _get_latest_claim(state: MASState, agent_id: str) -> Optional[str]:
    ptr = state.get("metadata", {}).get("latest_claim_by_agent", {}).get(agent_id)
    if ptr:
        return ptr
    for cid, cdata in reversed(list(state["claims"].items())):
        if isinstance(cdata, dict) and cdata.get("agent_id") == agent_id:
            return cdata.get("claim_id", cid)
    return None


def _get_latest_subtask(state: MASState, agent_id: str) -> Optional[str]:
    ptr = state.get("metadata", {}).get("latest_subtask_by_agent", {}).get(agent_id)
    if ptr:
        return ptr
    for sid, sdata in reversed(list(state["subtasks"].items())):
        if isinstance(sdata, dict) and sdata.get("agent_id") == agent_id:
            return sdata.get("subtask_id", sid)
    return None


def _get_incoming_subtask(state: MASState, agent_id: str) -> Optional[str]:
    return state.get("metadata", {}).get("incoming_subtask", {}).get(agent_id)


def _child_claim_ids_ordered(state: MASState, child_agent_ids: List[str]) -> List[str]:
    latest_ptr = state.get("metadata", {}).get("latest_claim_by_agent", {})
    result = []
    for child in child_agent_ids:
        cid = latest_ptr.get(child)
        if not cid:
            for k, v in reversed(list(state["claims"].items())):
                if isinstance(v, dict) and v.get("agent_id") == child:
                    cid = v.get("claim_id", k)
                    break
        if cid:
            result.append(cid)
    return result


def _root_claim_from_parents(state: MASState, parent_claim_ids: List[str], fallback: str) -> str:
    if not parent_claim_ids:
        return fallback
    first = parent_claim_ids[0]
    cdata = state["claims"].get(first)
    if isinstance(cdata, dict) and cdata.get("root_claim_id"):
        return cdata["root_claim_id"]
    return first


def _global_root_subtask(state: MASState) -> Optional[str]:
    ptr = state.get("metadata", {}).get("global_root_subtask_id")
    if ptr:
        return ptr
    for sid, sdata in state["subtasks"].items():
        if isinstance(sdata, dict) and sdata.get("depth") == 0:
            return sdata.get("subtask_id", sid)
    return None


def _update_metadata_pointers(
    state: MASState, agent_id: str, claim_id: Optional[str],
    subtask_id: Optional[str], incoming_subtask_map: Optional[Dict[str, str]] = None,
    global_root_subtask: Optional[str] = None,
) -> Dict:
    meta = dict(state.get("metadata") or {})
    lcba = dict(meta.get("latest_claim_by_agent") or {})
    if claim_id:
        lcba[agent_id] = claim_id
    meta["latest_claim_by_agent"] = lcba
    lsba = dict(meta.get("latest_subtask_by_agent") or {})
    if subtask_id:
        lsba[agent_id] = subtask_id
    meta["latest_subtask_by_agent"] = lsba
    if incoming_subtask_map:
        inc = dict(meta.get("incoming_subtask") or {})
        inc.update(incoming_subtask_map)
        meta["incoming_subtask"] = inc
    if global_root_subtask:
        meta["global_root_subtask_id"] = global_root_subtask
    return meta


# ── Topology ───────────────────────────────────────────────────────────

class TreeTopology(BaseTopology):

    def __init__(self, *, branching_factor: int = BRANCHING_FACTOR, **kwargs):
        super().__init__(**kwargs)
        self.branching_factor = branching_factor
        self._tree      = _build_tree(self.num_agents, branching_factor)
        self._depth_map = self._compute_depths()
        self._role_map  = self._compute_roles()

    def _compute_depths(self) -> Dict[str, int]:
        depths = {SUPERVISOR_ID: 0}
        queue  = [SUPERVISOR_ID]
        while queue:
            p = queue.pop(0)
            for c in self._tree.get(p, []):
                depths[c] = depths[p] + 1
                queue.append(c)
        return depths

    def _compute_roles(self) -> Dict[str, str]:
        roles: Dict[str, str] = {SUPERVISOR_ID: "supervisor"}
        for aid, children in self._tree.items():
            if children and aid != SUPERVISOR_ID:
                roles[aid] = "coordinator"
            elif not children:
                roles[aid] = "worker"
        return roles

    def name(self) -> TopologyName:
        return TopologyName.TREE

    def agent_ids(self) -> List[str]:
        return list(self._tree.keys())

    def edge_list(self) -> List[Tuple[str, str]]:
        edges = []
        for parent, children in self._tree.items():
            for child in children:
                edges.append((parent, child))
                edges.append((child, parent))
        return edges

    def build_graph(self) -> Any:
        builder = StateGraph(MASState)
        tree    = self._tree

        def post_order(node: str) -> List[str]:
            result = []
            for child in tree.get(node, []):
                result.extend(post_order(child))
            result.append(node)
            return result

        execution_order = post_order(SUPERVISOR_ID)

        def make_node(agent_id: str):
            role     = self._role_map.get(agent_id, "worker")
            depth    = self._depth_map.get(agent_id, 0)
            children = tree.get(agent_id, [])
            is_root  = (agent_id == SUPERVISOR_ID)
            is_leaf  = (len(children) == 0)

            parent_id: Optional[str] = None
            for pid, cs in tree.items():
                if agent_id in cs:
                    parent_id = pid
                    break

            def node_fn(state: MASState) -> Dict:
                self._step += 1
                self._maybe_snapshot()

                claim_id   = new_id("claim")
                subtask_id = new_id("sub")

                if is_leaf:
                    parent_subtask_id = (
                        _get_incoming_subtask(state, agent_id)
                        or (_get_latest_subtask(state, parent_id) if parent_id else None)
                    )
                else:
                    parent_subtask_id = _get_latest_subtask(state, parent_id) if parent_id else None

                root_subtask_id = _global_root_subtask(state) or subtask_id

                new_claims    = dict(state["claims"])
                new_subtasks  = dict(state["subtasks"])
                new_outputs   = dict(state["agent_outputs"])
                new_influence = dict(state["influence"])
                new_meta      = None

                # ── LEAF: worker node ──────────────────────────────────
                if is_leaf:
                    # FIX 1: leaf depends on parent claim — NOT a new cascade root.
                    # parent_claim_ids = [parent_cid] so cascades propagate upward.
                    parent_cid    = _get_latest_claim(state, parent_id or SUPERVISOR_ID)
                    parent_output = state["agent_outputs"].get(parent_id or SUPERVISOR_ID, "")

                    parent_claim_ids = [parent_cid] if parent_cid else []
                    root_claim_id    = _root_claim_from_parents(
                        state, parent_claim_ids, fallback=claim_id
                    )

                    prior_outputs_list = (
                        [(parent_id or SUPERVISOR_ID,
                          f"msg_tree_parent_{parent_id}",
                          parent_cid or "",
                          parent_output)]
                        if parent_output else []
                    )

                    spec = AgentContextSpec(
                        task=state["task"],
                        agent_id=agent_id,
                        agent_role="worker",
                        topology="tree",
                        task_family=self.task_family,
                        local_goal="Solve your assigned subtask.",
                        subtask_id=subtask_id,
                        parent_subtask_id=parent_subtask_id,
                        step=self._step,
                        neighbor_ids=([parent_id] if parent_id else []),
                        prior_outputs=prior_outputs_list,
                        available_tools=self._tool_names,
                    )
                    user_content = build_context(spec)

                    # FIX 2: event_type=None — leaf is NOT a root, NOT PROPOSE_CLAIM.
                    # extractor infers from parent_claim_ids + reasoning content.
                    output = self._call_llm(
                        agent_id=agent_id, agent_role=role,
                        system_prompt=f"You are a worker (depth {depth}) in a hierarchical MAS tree.",
                        user_content=user_content,
                        event_type=None,
                        claim_id=claim_id,
                        claim_depth=depth,
                        parent_claim_ids=parent_claim_ids,
                        root_claim_id=root_claim_id,
                        subtask_id=subtask_id,
                        parent_subtask_id=parent_subtask_id,
                        root_subtask_id=root_subtask_id,
                        subtask_depth=depth,
                        subtask_status="complete",
                        subtask_assigned_by=parent_id,
                        subtask_assigned_to=agent_id,
                        visible_neighbors=[parent_id] if parent_id else [],
                    )
                    new_meta = _update_metadata_pointers(state, agent_id, claim_id, subtask_id)

                # ── ROOT: supervisor node ──────────────────────────────
                elif is_root:
                    py_parent_claim_ids = _child_claim_ids_ordered(state, children)
                    root_claim_id       = _root_claim_from_parents(
                        state, py_parent_claim_ids, fallback=claim_id
                    )

                    active_children = [c for c in children if state["agent_outputs"].get(c)]
                    active_cids     = _child_claim_ids_ordered(state, active_children)
                    prior_outputs_list = [
                        (c, f"msg_tree_{c}", cid, state["agent_outputs"][c])
                        for c, cid in zip(active_children, active_cids)
                    ]

                    spec = AgentContextSpec(
                        task=state["task"],
                        agent_id=agent_id,
                        agent_role="supervisor",
                        topology="tree",
                        task_family=self.task_family,
                        local_goal="Synthesize all team reports into one final answer.",
                        subtask_id=subtask_id,
                        parent_subtask_id=parent_subtask_id,
                        step=self._step,
                        neighbor_ids=list(children),
                        prior_outputs=prior_outputs_list,
                        available_tools=self._tool_names,
                    )
                    user_content = build_context(spec)

                    # FIX 5: supervisor root gets PROPOSE_CLAIM — it is the true cascade root.
                    # Extractor also detects merge from parent_claim_ids >= 2, which is fine —
                    # both signals are consistent. PROPOSE_CLAIM marks it as the top root.
                    output = self._call_llm(
                        agent_id=agent_id, agent_role=role,
                        system_prompt="You are the top-level supervisor. Synthesize all team reports.",
                        user_content=user_content,
                        event_type=EventType.PROPOSE_CLAIM,
                        claim_id=claim_id,
                        claim_depth=depth,
                        parent_claim_ids=py_parent_claim_ids,
                        root_claim_id=root_claim_id,
                        subtask_id=subtask_id,
                        parent_subtask_id=parent_subtask_id,
                        root_subtask_id=root_subtask_id,
                        subtask_depth=depth,
                        subtask_status="active",
                        subtask_assigned_by=None,
                        subtask_assigned_to=agent_id,
                        visible_neighbors=list(children),
                    )
                    new_meta = _update_metadata_pointers(state, agent_id, claim_id, subtask_id)

                # ── COORDINATOR: team lead node ────────────────────────
                else:
                    parent_output = state["agent_outputs"].get(
                        parent_id or SUPERVISOR_ID, state["task"]
                    )
                    children_done = all(state["agent_outputs"].get(c) for c in children)

                    if children_done:
                        # Synthesis phase: merge child claims
                        active_children     = [c for c in children if state["agent_outputs"].get(c)]
                        py_parent_claim_ids = _child_claim_ids_ordered(state, active_children)
                        root_claim_id       = _root_claim_from_parents(
                            state, py_parent_claim_ids, fallback=claim_id
                        )
                        active_cids = _child_claim_ids_ordered(state, active_children)
                        prior_outputs_list = [
                            (c, f"msg_tree_{c}", cid, state["agent_outputs"][c])
                            for c, cid in zip(active_children, active_cids)
                        ]

                        spec = AgentContextSpec(
                            task=state["task"],
                            agent_id=agent_id,
                            agent_role="coordinator",
                            topology="tree",
                            task_family=self.task_family,
                            local_goal="Synthesize worker claims. Cite their claim IDs.",
                            subtask_id=subtask_id,
                            parent_subtask_id=parent_subtask_id,
                            step=self._step,
                            neighbor_ids=children + ([parent_id] if parent_id else []),
                            prior_outputs=prior_outputs_list,
                            available_tools=self._tool_names,
                        )
                        user_content = build_context(spec)

                        # event_type=None — extractor detects merge from parent_claim_ids >= 2
                        output = self._call_llm(
                            agent_id=agent_id, agent_role=role,
                            system_prompt=f"You are a coordinator (depth {depth}). Synthesize worker outputs.",
                            user_content=user_content,
                            event_type=None,
                            claim_id=claim_id,
                            claim_depth=depth,
                            parent_claim_ids=py_parent_claim_ids,
                            root_claim_id=root_claim_id,
                            subtask_id=subtask_id,
                            parent_subtask_id=parent_subtask_id,
                            root_subtask_id=root_subtask_id,
                            subtask_depth=depth,
                            subtask_status="active",
                            subtask_assigned_by=parent_id,
                            subtask_assigned_to=agent_id,
                            visible_neighbors=children + ([parent_id] if parent_id else []),
                        )
                        new_meta = _update_metadata_pointers(state, agent_id, claim_id, subtask_id)

                    else:
                        # Decomposition phase: delegate to children
                        # FIX 3: event_type=None in _call_llm.
                        # Delegation is logged separately via _log_event (zero-token, correct).
                        # FIX 7: use AgentContextSpec + build_context instead of raw f-string.
                        py_parent_claim_ids = []
                        root_claim_id       = claim_id
                        incoming_map: Dict[str, str] = {}

                        for child in children:
                            child_subtask_id = new_id("sub")
                            self._step += 1
                            self._log_event(
                                agent_id=agent_id, agent_role=role,
                                event_type=EventType.DELEGATE_SUBTASK,
                                message_id=new_id("msg"),
                                message_length_tokens=0, message_length_chars=0,
                                tokens_input=0, tokens_output=0, tokens_total_event=0,
                                latency_ms=0.0, action_success=True,
                                confidence_score=None, claim_text_hash=None,
                                claim_id=None, claim_type=None,
                                claim_depth=None, claim_status=None,
                                parent_claim_ids=[], root_claim_id=None,
                                subtask_id=child_subtask_id,
                                parent_subtask_id=subtask_id,
                                root_subtask_id=root_subtask_id,
                                subtask_depth=depth + 1,
                                subtask_assigned_by=agent_id,
                                subtask_assigned_to=child,
                                subtask_status="active",
                                target_agent_id=child,
                                visible_neighbors=[child],
                            )
                            new_subtasks[child_subtask_id] = {
                                "agent_id":   child,
                                "subtask_id": child_subtask_id,
                                "depth":      depth + 1,
                                "assigned_by": agent_id,
                            }
                            incoming_map[child] = child_subtask_id

                        # FIX 7: AgentContextSpec for decomposition prompt
                        prior_cid = _get_latest_claim(state, parent_id or SUPERVISOR_ID)
                        prior_out = state["agent_outputs"].get(parent_id or SUPERVISOR_ID, "")
                        prior_outputs_list = (
                            [(parent_id or SUPERVISOR_ID,
                              f"msg_tree_parent_{parent_id}",
                              prior_cid or "",
                              prior_out)]
                            if prior_out else []
                        )
                        spec = AgentContextSpec(
                            task=state["task"],
                            agent_id=agent_id,
                            agent_role="coordinator",
                            topology="tree",
                            task_family=self.task_family,
                            local_goal=(
                                f"Decompose into {len(children)} subtasks, "
                                "one per worker. Be explicit."
                            ),
                            subtask_id=subtask_id,
                            parent_subtask_id=parent_subtask_id,
                            step=self._step,
                            neighbor_ids=children + ([parent_id] if parent_id else []),
                            prior_outputs=prior_outputs_list,
                            available_tools=self._tool_names,
                        )
                        user_content = build_context(spec)

                        # FIX 3: event_type=None (not DELEGATE_SUBTASK in _call_llm)
                        output = self._call_llm(
                            agent_id=agent_id, agent_role=role,
                            system_prompt=f"You are a coordinator (depth {depth}). Decompose the task.",
                            user_content=user_content,
                            event_type=None,
                            claim_id=claim_id,
                            claim_depth=depth,
                            parent_claim_ids=py_parent_claim_ids,
                            root_claim_id=root_claim_id,
                            subtask_id=subtask_id,
                            parent_subtask_id=parent_subtask_id,
                            root_subtask_id=root_subtask_id,
                            subtask_depth=depth,
                            subtask_status="active",
                            subtask_assigned_by=parent_id,
                            subtask_assigned_to=agent_id,
                            visible_neighbors=children + ([parent_id] if parent_id else []),
                        )
                        new_meta = _update_metadata_pointers(
                            state, agent_id, claim_id, subtask_id,
                            incoming_subtask_map=incoming_map,
                            global_root_subtask=subtask_id if depth == 0 else None,
                        )

                # ── Minimal structural claim state ─────────────────────
                # No event semantics, no status, no action labels.
                new_claims[claim_id] = {
                    "agent_id":        agent_id,
                    "claim_id":        claim_id,
                    "parent_claim_ids": (
                        parent_claim_ids if is_leaf else py_parent_claim_ids
                    ),
                    "root_claim_id":   root_claim_id,
                    "depth":           depth,
                    "text_hash":       text_hash(output),
                }
                new_subtasks[subtask_id] = {
                    "agent_id":   agent_id,
                    "subtask_id": subtask_id,
                    "depth":      depth,
                    "status":     "complete" if is_leaf else "active",
                }
                new_outputs[agent_id]   = output
                new_influence[agent_id] = new_influence.get(agent_id, 0) + max(1, len(children))
                self._record_activation(self._step, [agent_id])

                updates: Dict = {
                    "messages":      [AIMessage(content=output, name=agent_id)],
                    "agent_outputs": new_outputs,
                    "influence":     new_influence,
                    "claims":        new_claims,
                    "subtasks":      new_subtasks,
                    "step":          self._step,
                }
                if new_meta is not None:
                    updates["metadata"] = new_meta
                if is_root:
                    updates["final_answer"] = output
                return updates

            return node_fn

        for aid in execution_order:
            builder.add_node(aid, make_node(aid))

        builder.set_entry_point(execution_order[0])
        for i in range(len(execution_order) - 1):
            builder.add_edge(execution_order[i], execution_order[i + 1])
        builder.add_edge(execution_order[-1], END)

        return builder.compile()