"""
dynamic_reputation.py  —  Dynamic Reputation-Routed Graph
----------------------------------------------------------
Structure
---------
  Starts as a flat peer graph. After each step, agents accumulate
  reputation scores based on how often other agents consult them.
  Routing is preferential-attachment: p(route to j) ∝ reputation[j] + epsilon.

  Over time, high-reputation agents attract more traffic → emergent
  elite agents → rich-get-richer → power-law influence distribution.

Design (paper-faithful)
-----------------------
Topology owns: structural lineage (claim_id, parent_claim_ids),
  routing logic (_softmax_sample), edge tracking, AgentContextSpec.
Topology does NOT own: event_type for non-root turns → event_extractor.py,
  merge_* kwargs → dag_builder.py, reputation injected into agent prompts.

Reputation routing is a topology-level mechanism (who sees whom).
Agents reason independently — they do not see reputation scores directly.
This ensures emergent coordination is not confounded by reputation awareness.

Power-law observables
---------------------
- Influence distribution: should be most heavy-tailed of all topologies
- Emergence of elite agents: top-k% commanding X% of interactions
- Preferential attachment: does influence ∝ prior influence?
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, END

from .base import BaseTopology, MASState, make_agent_id, new_id, text_hash
from context_builder import build_context, AgentContextSpec
from loggers.schemas import EventType, TopologyName

EPSILON   = 0.1
MAX_STEPS = 5
CONSULT_K = 2


def _softmax_sample(
    ids: List[str],
    reputation: Dict[str, float],
    exclude: str,
    k: int,
    rng: random.Random,
) -> List[str]:
    """Sample k agents without replacement using reputation-weighted probabilities."""
    candidates = [a for a in ids if a != exclude]
    weights    = [reputation.get(a, EPSILON) + EPSILON for a in candidates]
    total      = sum(weights)
    probs      = [w / total for w in weights]
    k          = min(k, len(candidates))
    chosen: List[str] = []
    remaining = list(zip(candidates, probs))
    for _ in range(k):
        if not remaining:
            break
        r_ids, r_probs = zip(*remaining)
        cum  = 0.0
        roll = rng.random()
        for cid, cp in zip(r_ids, r_probs):
            cum += cp
            if roll <= cum:
                chosen.append(cid)
                remaining = [(x, p) for x, p in remaining if x != cid]
                break
    return chosen


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
    result   = []
    combined = dict(state["claims"])
    if new_claims:
        combined.update(new_claims)
    for aid in agent_ids:
        cid = ptr_map.get(aid)
        if not cid:
            for k, v in reversed(list(combined.items())):
                if isinstance(v, dict) and v.get("agent_id") == aid:
                    cid = v.get("claim_id", k)
                    break
        if cid:
            result.append(cid)
    return result


def _root_claim_from_parents(
    state: MASState,
    parent_claim_ids: List[str],
    fallback: str,
    new_claims: Optional[Dict] = None,
) -> str:
    if not parent_claim_ids:
        return fallback
    combined = dict(state["claims"])
    if new_claims:
        combined.update(new_claims)
    cdata = combined.get(parent_claim_ids[0])
    if isinstance(cdata, dict) and cdata.get("root_claim_id"):
        return cdata["root_claim_id"]
    return parent_claim_ids[0]


def _update_meta(
    state: MASState,
    claim_updates: Optional[Dict[str, str]] = None,
    extra: Optional[Dict] = None,
) -> Dict:
    meta = dict(state.get("metadata") or {})
    if claim_updates:
        lcba = dict(meta.get("latest_claim_by_agent") or {})
        lcba.update(claim_updates)
        meta["latest_claim_by_agent"] = lcba
    if extra:
        meta.update(extra)
    return meta


class DynamicReputationTopology(BaseTopology):

    def __init__(self, *, consult_k: int = CONSULT_K, max_steps: int = MAX_STEPS, **kwargs):
        super().__init__(**kwargs)
        self.consult_k  = consult_k
        self.max_steps  = max_steps
        self._dynamic_edges: List[Tuple[str, str]] = []
        self._edge_weight_counter: Dict[str, float] = {}

    def name(self) -> TopologyName:
        return TopologyName.DYNAMIC_REPUTATION

    def agent_ids(self) -> List[str]:
        return [make_agent_id("agent", i) for i in range(self.num_agents)]

    def edge_list(self) -> List[Tuple[str, str]]:
        return list(self._dynamic_edges)

    def edge_weights(self) -> Dict[str, float]:
        return dict(self._edge_weight_counter)

    def _update_edges(self, src: str, consulted: List[str]) -> None:
        for dst in consulted:
            edge = (src, dst)
            if edge not in self._dynamic_edges:
                self._dynamic_edges.append(edge)
            key = f"{src}->{dst}"
            self._edge_weight_counter[key] = self._edge_weight_counter.get(key, 0) + 1

    def build_graph(self) -> Any:
        ids     = self.agent_ids()
        builder = StateGraph(MASState)

        # ── Step node ─────────────────────────────────────────────────

        def make_step_node(step_idx: int):
            def step_fn(state: MASState) -> Dict:
                rng        = random.Random(self.seed + step_idx * 1000)
                reputation = dict(state["influence"])
                new_outputs   = dict(state["agent_outputs"])
                new_influence = dict(reputation)
                new_claims    = dict(state["claims"])
                consultation_hits: Dict[str, float] = {}
                claim_ptr_updates: Dict[str, str]   = {}

                shuffled = list(ids)
                rng.shuffle(shuffled)

                from sparse_activation import select_active_agents
                _active = select_active_agents(
                    all_agents=shuffled, step_idx=step_idx,
                    agent_outputs=new_outputs, reputation=reputation,
                    rng=rng, strategy="reputation",
                )
                self._record_activation(self._step, _active)

                for agent_id in _active:
                    self._step += 1
                    self._maybe_snapshot()

                    # Reputation-weighted routing — topology mechanism, not agent-visible
                    consulted = _softmax_sample(
                        ids, reputation, agent_id, self.consult_k, rng
                    )
                    self._update_edges(agent_id, consulted)
                    for dst in consulted:
                        consultation_hits[dst] = consultation_hits.get(dst, 0.0) + 1.0

                    consulted_with_output = [p for p in consulted if new_outputs.get(p)]

                    # PROPOSE_CLAIM only at step 0 (no parents → structural root)
                    if step_idx == 0:
                        ev_type             = EventType.PROPOSE_CLAIM
                        py_parent_claim_ids = []
                    else:
                        ev_type             = None  # extractor infers from parent_claim_ids + signals
                        py_parent_claim_ids = _ordered_claim_ids(
                            state, consulted_with_output, new_claims
                        )

                    claim_id      = new_id("claim")
                    root_claim_id = _root_claim_from_parents(
                        state, py_parent_claim_ids, claim_id, new_claims
                    )

                    ptr_map = dict((state.get("metadata") or {}).get("latest_claim_by_agent") or {})
                    prior_outputs_list = [
                        (p, f"msg_rep_s{step_idx}_{p}", ptr_map.get(p, ""), new_outputs[p])
                        for p in consulted_with_output
                    ]

                    # FIX 2: no reputation scores in extra_context — agents reason independently.
                    # Routing is topology-level; agents do not see who was routed to them.
                    spec = AgentContextSpec(
                        task=state["task"],
                        agent_id=agent_id,
                        agent_role="worker",
                        topology="dynamic_reputation",
                        task_family=self.task_family,
                        local_goal=(
                            "Critically review consulted peer outputs. "
                            "Revise your reasoning where peers provide useful information. "
                            "Do not blindly agree with any peer."
                        ),
                        subtask_id=claim_id,
                        step=step_idx,
                        neighbor_ids=consulted,
                        prior_outputs=prior_outputs_list,
                        available_tools=self._tool_names,
                    )
                    user_content = build_context(spec)

                    # FIX 1: clean _call_llm — no stray tokens, correct commas
                    output = self._call_llm(
                        agent_id=agent_id,
                        agent_role="peer",
                        system_prompt=(
                            f"You are in a reputation-routed consultation network "
                            f"(step {step_idx + 1}/{self.max_steps}). "
                            "Evaluate peer reasoning critically and independently."
                        ),
                        user_content=user_content,
                        event_type=ev_type,
                        claim_id=claim_id,
                        claim_depth=step_idx,
                        parent_claim_ids=py_parent_claim_ids,
                        root_claim_id=root_claim_id,
                        visible_neighbors=consulted,
                    )

                    new_outputs[agent_id] = output
                    new_claims[claim_id]  = {
                        "agent_id":         agent_id,
                        "claim_id":         claim_id,
                        "parent_claim_ids": py_parent_claim_ids,
                        "root_claim_id":    root_claim_id,
                        "depth":            step_idx,
                        "text_hash":        text_hash(output),
                    }
                    claim_ptr_updates[agent_id] = claim_id

                # Reputation update: consultation frequency drives preferential attachment.
                # This is the core mechanism — logged in metadata for analysis.
                for consulted_id, count in consultation_hits.items():
                    new_influence[consulted_id] = new_influence.get(consulted_id, 0.0) + count
                for agent_id in _active:
                    new_influence[agent_id] = new_influence.get(agent_id, 0.0) + 0.1

                new_meta = _update_meta(
                    state,
                    claim_updates=claim_ptr_updates,
                    extra={
                        "step_consultations": (
                            list(state.get("metadata", {}).get("step_consultations", []))
                            + [{
                                "step":               step_idx,
                                "consultation_hits":  dict(consultation_hits),
                                "reputation_snapshot": dict(new_influence),
                            }]
                        )
                    },
                )

                return {
                    "messages":      [AIMessage(content=f"Reputation step {step_idx} done", name="system")],
                    "agent_outputs": new_outputs,
                    "influence":     new_influence,
                    "claims":        new_claims,
                    "step":          self._step,
                    "metadata":      new_meta,
                }
            return step_fn

        # ── Synthesis node ─────────────────────────────────────────────

        def synthesis(state: MASState) -> Dict:
            self._step += 1
            self._maybe_snapshot()

            reputation    = state["influence"]
            sorted_agents = sorted(ids, key=lambda a: reputation.get(a, 0.0), reverse=True)
            synthesizer   = sorted_agents[0]
            elite_agents  = sorted_agents[:max(1, len(sorted_agents) // 4)]
            total_rep     = sum(reputation.values()) or 1.0
            top_share     = sum(reputation.get(a, 0.0) for a in elite_agents) / total_rep

            elite_claim_ids = _ordered_claim_ids(state, elite_agents)
            final_claim_id  = new_id("claim")
            root_claim_id   = _root_claim_from_parents(
                state, elite_claim_ids, final_claim_id
            )

            ptr_map = dict((state.get("metadata") or {}).get("latest_claim_by_agent") or {})
            prior_outputs_list = [
                (a, f"msg_rep_synth_{a}", ptr_map.get(a, ""), state["agent_outputs"].get(a, ""))
                for a in elite_agents if state["agent_outputs"].get(a)
            ]

            # FIX 2: no reputation share in extra_context — synthesis is independent reasoning
            spec = AgentContextSpec(
                task=state["task"],
                agent_id=synthesizer,
                agent_role="synthesizer",
                topology="dynamic_reputation",
                task_family=self.task_family,
                local_goal=(
                    "Synthesize the best reasoning from the provided agent outputs "
                    "into one final answer."
                ),
                subtask_id=final_claim_id,
                step=self._step,
                neighbor_ids=elite_agents,
                prior_outputs=prior_outputs_list,
                available_tools=self._tool_names,
            )
            user_content = build_context(spec)

            # FIX 1: clean synthesis _call_llm — correct commas, no stray tokens
            output = self._call_llm(
                agent_id=synthesizer,
                agent_role="elite_synthesizer",
                system_prompt=(
                    "You are the synthesizer for this multi-agent system. "
                    "Produce the final answer from the provided agent reasoning."
                ),
                user_content=user_content,
                event_type=None,
                claim_id=final_claim_id,
                claim_depth=self.max_steps,
                parent_claim_ids=elite_claim_ids,
                root_claim_id=root_claim_id,
                visible_neighbors=elite_agents,
            )

            new_claims = dict(state["claims"])
            new_claims[final_claim_id] = {
                "agent_id":         synthesizer,
                "claim_id":         final_claim_id,
                "parent_claim_ids": elite_claim_ids,
                "root_claim_id":    root_claim_id,
                "depth":            self.max_steps,
                "text_hash":        text_hash(output),
            }
            new_outputs   = dict(state["agent_outputs"])
            new_outputs["final"] = output
            new_influence = dict(reputation)
            new_influence[synthesizer] = new_influence.get(synthesizer, 0.0) + self.num_agents
            self._record_activation(self._step, [synthesizer])

            new_meta = _update_meta(
                state,
                claim_updates={synthesizer: final_claim_id},
                extra={
                    "final_elite_share": round(top_share, 4),
                    "final_reputation":  dict(reputation),
                    "synthesizer":       synthesizer,
                    "elite_agents":      elite_agents,
                },
            )

            return {
                "messages":      [AIMessage(content=output, name=synthesizer)],
                "agent_outputs": new_outputs,
                "influence":     new_influence,
                "claims":        new_claims,
                "final_answer":  output,
                "step":          self._step,
                "metadata":      new_meta,
            }

        # ── Wire graph ─────────────────────────────────────────────────
        step_names = [f"step_{s}" for s in range(self.max_steps)]
        for s, sname in enumerate(step_names):
            builder.add_node(sname, make_step_node(s))
        builder.add_node("synthesis", synthesis)
        builder.set_entry_point(step_names[0])
        for i in range(len(step_names) - 1):
            builder.add_edge(step_names[i], step_names[i + 1])
        builder.add_edge(step_names[-1], "synthesis")
        builder.add_edge("synthesis", END)
        return builder.compile()