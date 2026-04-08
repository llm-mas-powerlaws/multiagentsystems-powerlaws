"""
star.py  —  Star / Hub-and-Spoke topology
Patch: build_context + AgentContextSpec replacing star_worker_context/star_hub_context.
topology="star", task_family=self.task_family, available_tools=self._tool_names.
prior_outputs as (agent_id, message_id, claim_id, text). All other logic unchanged.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, END

from .base import (
    BaseTopology, MASState,
    make_agent_id, new_id, text_hash,
)
from context_builder import build_context, AgentContextSpec   # PATCHED
from loggers.schemas import EventType, TopologyName

HUB_ID = "hub"


def _get_latest_claim(state: MASState, agent_id: str) -> Optional[str]:
    ptr = state.get("metadata", {}).get("latest_claim_by_agent", {}).get(agent_id)
    if ptr:
        return ptr
    for cid, cdata in reversed(list(state["claims"].items())):
        if isinstance(cdata, dict) and cdata.get("agent_id") == agent_id:
            return cdata.get("claim_id", cid)
    return None


def _ordered_claim_ids(
    state: MASState,
    agent_ids: List[str],
    new_claims: Optional[Dict] = None,
) -> List[str]:
    ptr_map = dict(state.get("metadata", {}).get("latest_claim_by_agent") or {})
    if new_claims:
        for cid, cdata in new_claims.items():
            if isinstance(cdata, dict) and cdata.get("agent_id"):
                ptr_map[cdata["agent_id"]] = cdata.get("claim_id", cid)
    result = []
    for aid in agent_ids:
        cid = ptr_map.get(aid)
        if not cid:
            combined = dict(state["claims"])
            if new_claims:
                combined.update(new_claims)
            for k, v in reversed(list(combined.items())):
                if isinstance(v, dict) and v.get("agent_id") == aid:
                    cid = v.get("claim_id", k)
                    break
        if cid:
            result.append(cid)
    return result


def _root_claim_from_parents(state, parent_claim_ids, fallback, new_claims=None):
    if not parent_claim_ids:
        return fallback
    combined = dict(state["claims"])
    if new_claims:
        combined.update(new_claims)
    cdata = combined.get(parent_claim_ids[0])
    if isinstance(cdata, dict) and cdata.get("root_claim_id"):
        return cdata["root_claim_id"]
    return parent_claim_ids[0]


def _update_meta(state, claim_updates=None, incoming_subtask=None,
                 hub_root_claim_id=None, extra=None):
    meta = dict(state.get("metadata") or {})
    if claim_updates:
        lcba = dict(meta.get("latest_claim_by_agent") or {})
        lcba.update(claim_updates)
        meta["latest_claim_by_agent"] = lcba
    if incoming_subtask:
        inc = dict(meta.get("incoming_subtask") or {})
        inc.update(incoming_subtask)
        meta["incoming_subtask"] = inc
    if hub_root_claim_id:
        meta["hub_root_claim_id"] = hub_root_claim_id
    if extra:
        meta.update(extra)
    return meta


class StarTopology(BaseTopology):

    def name(self) -> TopologyName:
        return TopologyName.STAR

    def _worker_ids(self) -> List[str]:
        return [make_agent_id("worker", i) for i in range(self.num_agents - 1)]

    def agent_ids(self) -> List[str]:
        return [HUB_ID] + self._worker_ids()

    def edge_list(self) -> List[Tuple[str, str]]:
        edges = []
        for wid in self._worker_ids():
            edges.append((HUB_ID, wid))
            edges.append((wid, HUB_ID))
        return edges

    def build_graph(self) -> Any:
        worker_ids = self._worker_ids()
        builder    = StateGraph(MASState)

        def hub_dispatch(state: MASState) -> Dict:
            self._step += 1
            self._maybe_snapshot()

            hub_claim_id = new_id("claim")
            root_subtask = new_id("sub")

            # PATCHED: build_context
            spec = AgentContextSpec(
                task=state["task"],
                agent_id=HUB_ID,
                agent_role="hub",
                topology="star",
                task_family=self.task_family,
                local_goal=(
                    f"Decompose the task into exactly {len(worker_ids)} subtasks, "
                    "one per worker. Format: WORKER_0: <subtask>\\nWORKER_1: <subtask>\\n..."
                ),
                subtask_id=root_subtask,
                step=self._step,
                neighbor_ids=worker_ids,
                prior_outputs=[],
                available_tools=self._tool_names,
            )
            user_content = build_context(spec)

            output = self._call_llm(
                agent_id=HUB_ID,
                agent_role="hub",
                system_prompt=(
                    "You are the central hub in a star-topology multi-agent system. "
                    "Decompose the task into subtasks, one per worker."
                ),
                user_content=user_content,
                event_type=EventType.PROPOSE_CLAIM,  # hub root claim — structural root, no parents
                claim_id=hub_claim_id,
                claim_depth=0,
                parent_claim_ids=[],
                root_claim_id=hub_claim_id,
                subtask_id=root_subtask,
                subtask_type=SubtaskType.PLANNING,
                subtask_depth=0,
                subtask_status="active",
                subtask_assigned_by=None,
                subtask_assigned_to=HUB_ID,
                visible_neighbors=worker_ids,
            )

            subtasks: Dict[str, str] = {}
            for i, wid in enumerate(worker_ids):
                marker = f"WORKER_{i}:"
                if marker in output:
                    start = output.index(marker) + len(marker)
                    end = output.index(f"WORKER_{i+1}:", start) if f"WORKER_{i+1}:" in output[start:] else len(output)
                    subtasks[wid] = output[start:end].strip()
                else:
                    subtasks[wid] = state["task"]

            new_subtasks = dict(state["subtasks"])
            incoming_map: Dict[str, str] = {}
            for wid, st in subtasks.items():
                child_sid = new_id("sub")
                self._step += 1
                self._log_event(
                    agent_id=HUB_ID, agent_role="hub",
                    event_type=EventType.DELEGATE_SUBTASK,
                    message_id=new_id("msg"), message_length_tokens=0,
                    message_length_chars=0, tokens_input=0, tokens_output=0,
                    tokens_total_event=0, latency_ms=0.0, action_success=True,
                    confidence_score=None, claim_text_hash=None,
                    claim_id=None, claim_type=None, claim_depth=None, claim_status=None,
                    parent_claim_ids=[], root_claim_id=hub_claim_id,
                    subtask_id=child_sid, parent_subtask_id=root_subtask,
                    root_subtask_id=root_subtask, subtask_depth=1,
                    subtask_assigned_by=HUB_ID,
                    subtask_assigned_to=wid, subtask_status="active",
                    target_agent_id=wid, visible_neighbors=[wid],
                )
                new_subtasks[child_sid] = {
                    "agent_id": wid, "subtask_id": child_sid,
                    "assigned_by": HUB_ID, "parent_subtask_id": root_subtask,
                    "root_subtask_id": root_subtask, "depth": 1,
                    "status": "assigned", "text": st,
                }
                incoming_map[wid] = child_sid

            new_subtasks[root_subtask] = {
                "agent_id": HUB_ID, "subtask_id": root_subtask,
                "depth": 0, "status": "in_progress",
            }

            new_claims = dict(state["claims"])
            new_claims[hub_claim_id] = {
                "agent_id": HUB_ID, "claim_id": hub_claim_id,
                "root_claim_id": hub_claim_id,
                "text_hash": text_hash(output), "depth": 0,
            }

            new_outputs = dict(state["agent_outputs"])
            new_outputs[HUB_ID] = output
            new_influence = dict(state["influence"])
            new_influence[HUB_ID] = new_influence.get(HUB_ID, 0) + len(worker_ids)
            self._record_activation(self._step, [HUB_ID])

            new_meta = _update_meta(
                state, claim_updates={HUB_ID: hub_claim_id},
                incoming_subtask=incoming_map, hub_root_claim_id=hub_claim_id,
                extra={"worker_subtasks": subtasks, "root_subtask_id": root_subtask},
            )

            return {
                "messages":      [AIMessage(content=output, name=HUB_ID)],
                "agent_outputs": new_outputs, "influence": new_influence,
                "claims": new_claims, "subtasks": new_subtasks,
                "metadata": new_meta, "step": self._step,
            }

        builder.add_node("hub_dispatch", hub_dispatch)

        def make_worker(worker_id: str, worker_index: int):
            def worker_fn(state: MASState) -> Dict:
                self._step += 1
                self._maybe_snapshot()

                subtask_map  = state["metadata"].get("worker_subtasks", {})
                subtask_text = subtask_map.get(worker_id, state["task"])
                assigned_subtask_id = state["metadata"].get("incoming_subtask", {}).get(worker_id)
                root_subtask_id     = state["metadata"].get("root_subtask_id")
                hub_claim_id        = state["metadata"].get("hub_root_claim_id")
                root_claim_id       = hub_claim_id
                claim_id            = new_id("claim")

                # PATCHED: prior_outputs includes hub's dispatch
                hub_cid  = hub_claim_id or ""
                hub_text = state["agent_outputs"].get(HUB_ID, "")
                prior_outputs_list = [(HUB_ID, f"msg_hub_dispatch", hub_cid, hub_text)] if hub_text else []

                spec = AgentContextSpec(
                    task=state["task"],
                    agent_id=worker_id,
                    agent_role="worker",
                    topology="star",
                    task_family=self.task_family,
                    local_goal=f"Complete your assigned subtask: {subtask_text}",
                    subtask_id=assigned_subtask_id or new_id("sub"),
                    parent_subtask_id=root_subtask_id,
                    step=self._step,
                    neighbor_ids=[HUB_ID],
                    prior_outputs=prior_outputs_list,
                    available_tools=self._tool_names,
                )
                user_content = build_context(spec)

                output = self._call_llm(
                    agent_id=worker_id, agent_role="worker",
                    system_prompt=f"You are worker {worker_index} in a star-topology MAS. Complete your subtask.",
                    user_content=user_content,
                    event_type=None,  # post-hoc: extractor infers from parent_claim_ids + signals
                    claim_id=claim_id,
                    claim_depth=1,
                    parent_claim_ids=[hub_claim_id] if hub_claim_id else [],
                    root_claim_id=root_claim_id or claim_id,
                    subtask_id=assigned_subtask_id or new_id("sub"),
                    parent_subtask_id=root_subtask_id, root_subtask_id=root_subtask_id,
                    subtask_depth=1,
                    subtask_status="complete",
                    subtask_assigned_by=HUB_ID, subtask_assigned_to=worker_id,
                    target_agent_id=HUB_ID, visible_neighbors=[HUB_ID],
                )

                new_claims = dict(state["claims"])
                new_claims[claim_id] = {
                    "agent_id": worker_id, "claim_id": claim_id,
                    "root_claim_id": root_claim_id or claim_id,
                    "text_hash": text_hash(output), "depth": 1,
                }
                new_subtasks = dict(state["subtasks"])
                if assigned_subtask_id and assigned_subtask_id in new_subtasks:
                    updated = dict(new_subtasks[assigned_subtask_id])
                    updated["status"] = "completed"
                    new_subtasks[assigned_subtask_id] = updated

                new_outputs = dict(state["agent_outputs"])
                new_outputs[worker_id] = output
                new_influence = dict(state["influence"])
                new_influence[worker_id] = new_influence.get(worker_id, 0) + 1
                self._record_activation(self._step, [worker_id])
                new_meta = _update_meta(state, claim_updates={worker_id: claim_id})

                return {
                    "messages": [AIMessage(content=output, name=worker_id)],
                    "agent_outputs": new_outputs, "influence": new_influence,
                    "claims": new_claims, "subtasks": new_subtasks,
                    "metadata": new_meta, "step": self._step,
                }
            return worker_fn

        for i, wid in enumerate(worker_ids):
            builder.add_node(wid, make_worker(wid, i))

        def hub_synthesize(state: MASState) -> Dict:
            self._step += 1
            self._maybe_snapshot()

            worker_claim_ids = _ordered_claim_ids(state, worker_ids)
            final_claim_id   = new_id("claim")
            root_claim_id    = _root_claim_from_parents(
                state, worker_claim_ids,
                fallback=state["metadata"].get("hub_root_claim_id") or final_claim_id,
            )


            # PATCHED: prior_outputs with claim IDs
            ptr_map = state.get("metadata", {}).get("latest_claim_by_agent", {})
            prior_outputs_list = [
                (wid, f"msg_worker_{i}", ptr_map.get(wid, ""), state["agent_outputs"].get(wid, ""))
                for i, wid in enumerate(worker_ids)
                if state["agent_outputs"].get(wid)
            ]

            spec = AgentContextSpec(
                task=state["task"],
                agent_id=HUB_ID,
                agent_role="synthesizer",
                topology="star",
                task_family=self.task_family,
                local_goal="Synthesize all worker outputs into one final answer. Resolve contradictions.",
                subtask_id=new_id("sub"),
                step=self._step,
                neighbor_ids=worker_ids,
                prior_outputs=prior_outputs_list,
                available_tools=self._tool_names,
            )
            user_content = build_context(spec)

            output = self._call_llm(
                agent_id=HUB_ID, agent_role="hub",
                system_prompt="You are the central hub. Synthesize all worker outputs into one final answer.",
                user_content=user_content,
                event_type=None,  # post-hoc: extractor detects merge from parent_claim_ids >= 2
                claim_id=final_claim_id,
                claim_depth=2,
                parent_claim_ids=worker_claim_ids,
                root_claim_id=root_claim_id,
                visible_neighbors=worker_ids, num_agents_involved=len(worker_ids),
            )

            new_claims = dict(state["claims"])
            new_claims[final_claim_id] = {
                "agent_id": HUB_ID, "claim_id": final_claim_id,
                "root_claim_id": root_claim_id, "text_hash": text_hash(output),
                "depth": 2,
            }
            new_outputs = dict(state["agent_outputs"])
            new_outputs[HUB_ID + "_final"] = output
            new_influence = dict(state["influence"])
            new_influence[HUB_ID] = new_influence.get(HUB_ID, 0) + len(worker_ids)
            self._record_activation(self._step, [HUB_ID])
            new_meta = _update_meta(state, claim_updates={HUB_ID: final_claim_id})

            return {
                "messages": [AIMessage(content=output, name=HUB_ID)],
                "agent_outputs": new_outputs, "influence": new_influence,
                "claims": new_claims, "metadata": new_meta,
                "final_answer": output, "step": self._step,
            }

        builder.add_node("hub_synthesize", hub_synthesize)

        builder.set_entry_point("hub_dispatch")
        prev = "hub_dispatch"
        for wid in worker_ids:
            builder.add_edge(prev, wid)
            prev = wid
        builder.add_edge(prev, "hub_synthesize")
        builder.add_edge("hub_synthesize", END)

        return builder.compile()