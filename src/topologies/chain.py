"""
src/topologies/chain.py  —  Sequential / Chain / Pipeline topology

Structure
---------
  agent_000 → agent_001 → ... → agent_{N-1}

Each agent receives all prior outputs as structured context and produces
its own output. The last agent is the synthesizer.

Design (post-refactor)
----------------------
Topology is responsible for:
  - assigning claim_id, parent_claim_ids, subtask_id (structural lineage)
  - building AgentContextSpec (what the agent can see)
  - calling _call_llm and logging raw trace rows
  - maintaining minimal claim state for lineage tracking

Topology is NOT responsible for:
  - deciding event types (revise/contradict/endorse/merge) → event_extractor.py
  - parsing action from output → removed
  - computing novelty/critique/similarity → removed
  - assigning revision_chain_id or contradiction_group_id → dag_builder.py
  - storing event semantics in state["claims"] → removed

state["claims"] contains only structural lineage:
  claim_id, agent_id, parent_claim_ids, depth, text_hash
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, END

from .base import BaseTopology, MASState, make_agent_id, new_id, text_hash
from context_builder import build_context, AgentContextSpec
from loggers.schemas import EventType, TopologyName


class ChainTopology(BaseTopology):
    """Sequential pipeline: agent_0 → agent_1 → ... → agent_{N-1}."""

    def name(self) -> TopologyName:
        return TopologyName.CHAIN

    def agent_ids(self) -> List[str]:
        return [make_agent_id("agent", i) for i in range(self.num_agents)]

    def edge_list(self) -> List[Tuple[str, str]]:
        ids = self.agent_ids()
        return [(ids[i], ids[i + 1]) for i in range(len(ids) - 1)]

    # ── Lineage summary for extra_context ─────────────────────────────
    # Shows agents the structural history — claim IDs, depth, prior text.
    # Does not include action/event labels (those are post-hoc).

    def _build_lineage_summary(self, state: MASState, upto_index: int) -> str:
        if upto_index <= 0:
            return "No prior lineage."
        lines: List[str] = []
        claims       = state.get("claims", {})
        agent_outputs = state.get("agent_outputs", {})
        for j in range(upto_index):
            aid    = make_agent_id("agent", j)
            output = agent_outputs.get(aid, "")
            # Find this agent's claim_id from structural state
            cid = next(
                (c for c, meta in claims.items() if meta.get("agent_id") == aid),
                "unknown"
            )
            short = " ".join(output.split())[:160]
            if len(" ".join(output.split())) > 160:
                short += "..."
            lines.append(f"- Agent {j} | claim_id={cid} | output={short}")
        return "\n".join(lines)

    # ── Graph construction ─────────────────────────────────────────────

    def build_graph(self) -> Any:
        ids     = self.agent_ids()
        builder = StateGraph(MASState)

        for i, aid in enumerate(ids):
            is_last    = (i == len(ids) - 1)
            agent_role = "synthesizer" if is_last else "worker"

            def make_node(agent_id=aid, agent_index=i, last=is_last, role=agent_role):
                def node_fn(state: MASState) -> Dict:
                    self._step += 1
                    self._maybe_snapshot()
                    self._record_activation(agent_index, [agent_id])

                    # ── Structural lineage from state ──────────────────
                    prev_claim_id   = state.get("metadata", {}).get("last_claim_id")
                    prev_subtask_id = state.get("metadata", {}).get("last_subtask_id")
                    root_claim_id   = state.get("metadata", {}).get("root_claim_id")
                    root_subtask_id = state.get("metadata", {}).get("root_subtask_id_sub")

                    claim_id   = new_id("claim")
                    subtask_id = new_id("sub")

                    # parent_claim_ids: topology-structural (what this agent depends on)
                    # For chain: always the immediately preceding agent's claim.
                    parent_claim_ids = [prev_claim_id] if prev_claim_id else []

                    # ── Build prior_outputs in (aid, mid, cid, text) format ──
                    claims        = state.get("claims", {})
                    agent_outputs = state.get("agent_outputs", {})
                    prior_outputs_list: List[Tuple[str, str, str, str]] = []
                    for j in range(agent_index):
                        prior_aid  = make_agent_id("agent", j)
                        prior_text = agent_outputs.get(prior_aid, "")
                        if not prior_text:
                            continue
                        prior_cid = next(
                            (cid for cid, meta in claims.items()
                             if meta.get("agent_id") == prior_aid),
                            f"claim_unknown_{j}"
                        )
                        prior_outputs_list.append(
                            (prior_aid, f"msg_chain_{j}", prior_cid, prior_text)
                        )

                    # ── Build context ──────────────────────────────────
                    lineage_summary = self._build_lineage_summary(state, agent_index)

                    spec = AgentContextSpec(
                        task=state["task"],
                        agent_id=agent_id,
                        agent_role=role,
                        topology="chain",
                        task_family=self.task_family,
                        local_goal=(
                            "Synthesize all prior outputs into one final answer."
                            if last else
                            "Improve, refine, or correct the prior reasoning."
                        ),
                        subtask_id=subtask_id,
                        parent_subtask_id=prev_subtask_id,
                        step=agent_index,
                        max_steps=self.num_agents,
                        neighbor_ids=(
                            [make_agent_id("agent", agent_index + 1)] if not last else []
                        ),
                        prior_outputs=prior_outputs_list,
                        available_tools=self._tool_names,
                        extra_context=(
                            f"Chain position: {agent_index} of {self.num_agents - 1}\n"
                            f"Previous claim id: {prev_claim_id or 'none'}\n"
                            f"Lineage:\n{lineage_summary}"
                        ),
                    )
                    user_content  = build_context(spec)
                    system_prompt = (
                        f"You are agent {agent_index} of {self.num_agents} "
                        f"in a sequential reasoning pipeline. Role: {role}. "
                        f"Return valid JSON only."
                    )

                    # ── LLM call — raw trace row logged by _call_llm ───
                    # event_type: PROPOSE_CLAIM only for true root (no parents).
                    # All other turns pass None — event_extractor.py decides.
                    answer = self._call_llm(
                        agent_id=agent_id,
                        agent_role=role,
                        system_prompt=system_prompt,
                        user_content=user_content,
                        event_type=EventType.PROPOSE_CLAIM if agent_index == 0 else None,
                        claim_id=claim_id,
                        parent_claim_ids=parent_claim_ids,
                        root_claim_id=root_claim_id or claim_id,
                        claim_depth=agent_index,
                        subtask_id=subtask_id,
                        parent_subtask_id=prev_subtask_id,
                        root_subtask_id=root_subtask_id or subtask_id,
                        subtask_depth=agent_index,
                        subtask_status="complete" if last else "active",
                        subtask_assigned_by=(
                            make_agent_id("agent", agent_index - 1)
                            if agent_index > 0 else None
                        ),
                        subtask_assigned_to=agent_id,
                        visible_neighbors=(
                            [make_agent_id("agent", agent_index + 1)] if not last else []
                        ),
                    )

                    # ── Minimal structural claim state ─────────────────
                    # No event semantics, no action, no novelty, no critique.
                    # event_extractor.py infers everything from the raw trace row.
                    new_claims = dict(state.get("claims", {}))
                    new_claims[claim_id] = {
                        "agent_id":        agent_id,
                        "claim_id":        claim_id,
                        "parent_claim_ids": parent_claim_ids,
                        "depth":           agent_index,
                        "text_hash":       text_hash(answer),
                    }

                    new_outputs = dict(agent_outputs)
                    new_outputs[agent_id] = answer

                    new_influence = dict(state.get("influence", {}))
                    new_influence[agent_id] = new_influence.get(agent_id, 0) + 1

                    new_meta = dict(state.get("metadata", {}))
                    new_meta["last_claim_id"]   = claim_id
                    new_meta["last_subtask_id"] = subtask_id
                    if "root_claim_id" not in new_meta:
                        new_meta["root_claim_id"] = claim_id
                    if "root_subtask_id_sub" not in new_meta:
                        new_meta["root_subtask_id_sub"] = subtask_id

                    updates: Dict[str, Any] = {
                        "messages":      [AIMessage(content=answer, name=agent_id)],
                        "agent_outputs": new_outputs,
                        "influence":     new_influence,
                        "claims":        new_claims,
                        "metadata":      new_meta,
                        "step":          self._step,
                    }
                    if last:
                        updates["final_answer"] = answer

                    return updates

                return node_fn

            builder.add_node(aid, make_node())

        builder.set_entry_point(ids[0])
        for i in range(len(ids) - 1):
            builder.add_edge(ids[i], ids[i + 1])
        builder.add_edge(ids[-1], END)

        return builder.compile()