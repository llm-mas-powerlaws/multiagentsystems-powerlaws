"""
sparse_mesh.py  —  Sparse Mesh / Peer-to-Peer topology
-------------------------------------------------------
Structure
---------
  Each agent is connected to exactly K neighbors (K-regular random graph
  by default; falls back to Erdős–Rényi if K-regular not feasible).
  No central hub — communication is local-neighborhood only.

  agent_i ↔ {k random neighbors}

Execution model
---------------
  Each agent, in a shuffled order, reads its neighbors' outputs and
  updates its own. Multiple passes (rounds). The agent with the highest
  weighted degree at the end becomes the synthesizer.

Power-law observables
---------------------
- Emergent hub detection: some agents accumulate more influence via
  random walk — are their degree distributions heavy-tailed?
- Decentralized cascade size: how far does a single claim propagate
  through the sparse graph?

PATCH vs original:
  - Round node: replaced `peer_context as build_peer_ctx` + `user_content, _ = build_peer_ctx(...)`
    with `build_context(AgentContextSpec(...))`. prior_outputs in new (aid, mid, cid, text) format.
  - Synthesis node: replaced hand-built user_content string with build_context(AgentContextSpec(...)).
  - Added topology="sparse_mesh", task_family=self.task_family, available_tools=self._tool_names.
  - All other logic (graph construction, edge weights, activation, claim lineage,
    state updates, graph wiring) is byte-for-byte identical to the original.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, END

from .base import (
    BaseTopology, MASState,
    make_agent_id, new_id, text_hash,
)
from context_builder import build_context, AgentContextSpec
from loggers.schemas import EventType, TopologyName

DEFAULT_K   = 3    # each agent connects to K neighbors
MAX_ROUNDS  = 3


def _build_sparse_graph(
    agent_ids: List[str],
    k: int,
    seed: int,
) -> nx.Graph:
    """Build a K-regular random graph; fallback to Erdős–Rényi."""
    n = len(agent_ids)
    k = min(k, n - 1)
    try:
        if n * k % 2 == 0:
            g = nx.random_regular_graph(k, n, seed=seed)
        else:
            g = nx.random_regular_graph(k - 1, n, seed=seed)
    except Exception:
        g = nx.erdos_renyi_graph(n, (k / max(n - 1, 1)), seed=seed)

    # Relabel: int nodes → agent_ids
    mapping = {i: agent_ids[i] for i in range(n)}
    return nx.relabel_nodes(g, mapping)


# ─────────────────────────────────────────────────────────────────
# State helpers — same pattern as full_mesh/dynamic_reputation
# ─────────────────────────────────────────────────────────────────

def _ordered_claim_ids(
    state: "MASState",
    agent_ids: List[str],
    new_claims: Optional[Dict] = None,
) -> List[str]:
    """Return claim IDs in exactly the same order as agent_ids. Skips missing."""
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
    state: "MASState",
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
    state: "MASState",
    claim_updates: Dict[str, str],
) -> Dict:
    meta = dict(state.get("metadata") or {})
    lcba = dict(meta.get("latest_claim_by_agent") or {})
    lcba.update(claim_updates)
    meta["latest_claim_by_agent"] = lcba
    return meta


class SparseMeshTopology(BaseTopology):
    """K-regular sparse random graph: local neighborhood communication."""

    def __init__(self, *, k: int = DEFAULT_K, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self._graph: Optional[nx.Graph] = None  # built lazily in build_graph

    def name(self) -> TopologyName:
        return TopologyName.SPARSE_MESH

    def agent_ids(self) -> List[str]:
        return [make_agent_id("agent", i) for i in range(self.num_agents)]

    def edge_list(self) -> List[Tuple[str, str]]:
        if self._graph is None:
            return []
        return list(self._graph.edges())

    def edge_weights(self) -> Dict[str, float]:
        """Edge weight = number of times the edge was used for messaging."""
        if self._graph is None:
            return {}
        return {
            f"{u}->{v}": self._graph[u][v].get("weight", 1.0)
            for u, v in self._graph.edges()
        }

    def build_graph(self) -> Any:
        ids           = self.agent_ids()
        self._graph   = _build_sparse_graph(ids, self.k, self.seed)
        adj: Dict[str, List[str]] = {
            aid: list(self._graph.neighbors(aid)) for aid in ids
        }

        builder = StateGraph(MASState)

        # ── Per-round node: each agent updates based on its neighbors ─

        def make_round_node(round_idx: int):
            def round_fn(state: MASState) -> Dict:
                rng          = random.Random(self.seed + round_idx)
                shuffled_ids = list(ids)
                rng.shuffle(shuffled_ids)

                new_outputs   = dict(state["agent_outputs"])
                new_influence = dict(state["influence"])
                new_claims    = dict(state["claims"])

                claim_ptr_updates: Dict[str, str] = {}

                # Sparse activation for large N
                from sparse_activation import select_active_agents
                _active = select_active_agents(
                    all_agents=shuffled_ids, step_idx=round_idx,
                    agent_outputs=new_outputs, strategy="frontier",
                )
                self._record_activation(self._step, _active)

                for agent_id in _active:
                    self._step += 1
                    self._maybe_snapshot()

                    neighbors = adj.get(agent_id, [])
                    neighbors_with_output = [nb for nb in neighbors if new_outputs.get(nb)]

                    # Event type from graph structure — no fake endorsement heuristic
                    if round_idx == 0:
                        ev_type             = EventType.PROPOSE_CLAIM
                        py_parent_claim_ids = []
                    else:
                        ev_type             = None  # post-hoc: extractor infers revise/contradict/endorse
                        py_parent_claim_ids = _ordered_claim_ids(
                            state, neighbors_with_output, new_claims
                        )

                    claim_id      = new_id("claim")
                    root_claim_id = _root_claim_from_parents(
                        state, py_parent_claim_ids, claim_id, new_claims
                    )

                    # PATCHED: build prior_outputs in new (aid, mid, cid, text) format
                    ptr_map = dict((state.get("metadata") or {}).get("latest_claim_by_agent") or {})
                    prior_outputs_list = [
                        (nb,
                         f"msg_sparse_r{round_idx}_{nb}",
                         ptr_map.get(nb, ""),
                         new_outputs[nb])
                        for nb in neighbors_with_output
                    ]

                    # PATCHED: AgentContextSpec + build_context replaces peer_context call
                    spec = AgentContextSpec(
                        task=state["task"],
                        agent_id=agent_id,
                        agent_role="worker",
                        topology="sparse_mesh",
                        task_family=self.task_family,
                        local_goal="Update your answer using neighbor outputs.",
                        subtask_id=claim_id,
                        step=round_idx,
                        neighbor_ids=neighbors,
                        prior_outputs=prior_outputs_list,
                        available_tools=self._tool_names,
                    )
                    system_prompt = (
                        f"You are agent {agent_id} in a sparse P2P network "
                        f"(round {round_idx+1}/{MAX_ROUNDS}). "
                        f"Only communicate with {len(neighbors)} direct neighbors."
                    )
                    user_content = build_context(spec)

                    output = self._call_llm(
                        agent_id=agent_id,
                        agent_role="peer",
                        system_prompt=system_prompt,
                        user_content=user_content,
                        event_type=ev_type,
                        claim_id=claim_id,
                        claim_depth=round_idx,
                        claim_status=None,  # post-hoc: extractor assigns revised/contradicted/merged
                        parent_claim_ids=py_parent_claim_ids,
                        root_claim_id=root_claim_id,
                        visible_neighbors=neighbors,
                    )

                    # Update edge weights (communication happened)
                    for nb in neighbors:
                        if self._graph and self._graph.has_edge(agent_id, nb):
                            current_w = self._graph[agent_id][nb].get("weight", 0.0)
                            self._graph[agent_id][nb]["weight"] = current_w + 1.0

                    new_outputs[agent_id]   = output
                    new_influence[agent_id] = new_influence.get(agent_id, 0) + len(neighbors)
                    new_claims[claim_id]    = {
                        "agent_id":        agent_id,
                        "claim_id":        claim_id,
                        "parent_claim_ids": py_parent_claim_ids,
                        "root_claim_id":   root_claim_id,
                        "depth":           round_idx,
                        "text_hash":       text_hash(output),
                    }
                    claim_ptr_updates[agent_id] = claim_id

                new_meta = _update_meta(state, claim_ptr_updates)

                return {
                    "messages": [AIMessage(content=f"Sparse mesh round {round_idx} done", name="system")],
                    "agent_outputs": new_outputs,
                    "influence": new_influence,
                    "claims": new_claims,
                    "metadata": new_meta,
                    "step": self._step,
                }
            return round_fn

        # ── Synthesis node: highest-influence agent synthesizes ───────

        def synthesis(state: MASState) -> Dict:
            self._step += 1
            self._maybe_snapshot()

            influence   = state["influence"]
            synthesizer = max(influence, key=influence.get) if influence else ids[0]

            # Explicit ordered latest claim IDs for all agents
            all_agent_claim_ids = _ordered_claim_ids(state, ids)
            final_claim_id      = new_id("claim")
            root_claim_id       = _root_claim_from_parents(
                state, all_agent_claim_ids, final_claim_id
            )

            # PATCHED: prior_outputs in new format for synthesis context
            ptr_map = dict((state.get("metadata") or {}).get("latest_claim_by_agent") or {})
            active_agents = [aid for aid in ids if state["agent_outputs"].get(aid)]
            prior_outputs_list = [
                (aid,
                 f"msg_sparse_synth_{aid}",
                 ptr_map.get(aid, ""),
                 state["agent_outputs"][aid])
                for aid in active_agents
            ]

            # PATCHED: AgentContextSpec + build_context replaces hand-built user_content
            spec = AgentContextSpec(
                task=state["task"],
                agent_id=synthesizer,
                agent_role="synthesizer",
                topology="sparse_mesh",
                task_family=self.task_family,
                local_goal=(
                    "Synthesize all agents' final outputs into one coherent answer."
                ),
                subtask_id=final_claim_id,
                step=self._step,
                neighbor_ids=[a for a in adj.get(synthesizer, []) if a != synthesizer],
                prior_outputs=prior_outputs_list,
                available_tools=self._tool_names,
            )
            system_prompt = (
                "You are the emergent leader of a peer-to-peer multi-agent network "
                "(elected by influence). Synthesize all agents' final outputs into "
                "one coherent answer."
            )
            user_content = build_context(spec)

            output = self._call_llm(
                agent_id=synthesizer,
                agent_role="emergent_hub",
                system_prompt=system_prompt,
                user_content=user_content,
                event_type=None,  # post-hoc: extractor detects merge from parent_claim_ids >= 2
                claim_id=final_claim_id,
                claim_depth=MAX_ROUNDS,
                parent_claim_ids=all_agent_claim_ids,
                root_claim_id=root_claim_id,
                visible_neighbors=[a for a in adj.get(synthesizer, []) if a != synthesizer],
            )

            # Store final claim
            new_claims = dict(state["claims"])
            new_claims[final_claim_id] = {
                "agent_id":        synthesizer,
                "claim_id":        final_claim_id,
                "parent_claim_ids": all_agent_claim_ids,
                "root_claim_id":   root_claim_id,
                "depth":           MAX_ROUNDS,
                "text_hash":       text_hash(output),
            }

            new_outputs = dict(state["agent_outputs"])
            new_outputs["final"] = output
            new_influence = dict(state["influence"])
            new_influence[synthesizer] = new_influence.get(synthesizer, 0) + len(all_agent_claim_ids)

            self._record_activation(self._step, [synthesizer])
            new_meta = _update_meta(state, {synthesizer: final_claim_id})

            return {
                "messages": [AIMessage(content=output, name=synthesizer)],
                "agent_outputs": new_outputs,
                "influence": new_influence,
                "claims": new_claims,
                "metadata": new_meta,
                "final_answer": output,
                "step": self._step,
            }

        # ── Wire graph ────────────────────────────────────
        round_names = [f"round_{r}" for r in range(MAX_ROUNDS)]
        for r, rname in enumerate(round_names):
            builder.add_node(rname, make_round_node(r))
        builder.add_node("synthesis", synthesis)

        builder.set_entry_point(round_names[0])
        for i in range(len(round_names) - 1):
            builder.add_edge(round_names[i], round_names[i + 1])
        builder.add_edge(round_names[-1], "synthesis")
        builder.add_edge("synthesis", END)

        return builder.compile()