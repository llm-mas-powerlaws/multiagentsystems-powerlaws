"""
hybrid.py  —  Hybrid Modular Directed Graph topology
------------------------------------------------------
Structure
---------
  Agents divided into M communities. Full mesh within each community.
  Bridge agents (first in each community) relay to global integrator.

  Community_0:  [agents 0..k]  ←→  bridge_0  ─┐
  Community_1:  [agents k+1..2k] ←→  bridge_1  ─┼→  integrator
  Community_2:  [agents 2k+1..3k] ←→ bridge_2  ─┘

Design (paper-faithful)
-----------------------
Topology owns: claim_id, parent_claim_ids, structural lineage, AgentContextSpec.
Topology does NOT own: event_type (except PROPOSE_CLAIM for first speaker),
  merge_*/contradiction_* kwargs, semantic labels in state["claims"].

Cascade structure:
  - First speaker in community: PROPOSE_CLAIM (no parents → structural root)
  - Subsequent speakers: event_type=None (extractor infers from parent_claim_ids)
  - Bridge summary: event_type=None, parent_claim_ids=community_member_cids
  - Integrator: event_type=None, parent_claim_ids=bridge_summary_cids
"""

from __future__ import annotations
import math
from typing import Any, Dict, List, Optional, Tuple
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, END
from .base import BaseTopology, MASState, make_agent_id, new_id, text_hash
from context_builder import build_context, AgentContextSpec
from loggers.schemas import EventType, TopologyName

NUM_COMMUNITIES = 2
INTEGRATOR_ID   = "integrator"


def _partition_agents(agent_ids: List[str], num_communities: int) -> List[List[str]]:
    size = math.ceil(len(agent_ids) / num_communities)
    return [agent_ids[i * size:(i + 1) * size] for i in range(num_communities)]


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


def _update_meta_claims(
    state: MASState,
    updates: Dict[str, str],
    bridge_summary_updates: Optional[Dict[int, str]] = None,
) -> Dict:
    meta = dict(state.get("metadata") or {})
    lcba = dict(meta.get("latest_claim_by_agent") or {})
    lcba.update(updates)
    meta["latest_claim_by_agent"] = lcba
    if bridge_summary_updates:
        bsc = dict(meta.get("bridge_summary_claim_by_community") or {})
        bsc.update({str(k): v for k, v in bridge_summary_updates.items()})
        meta["bridge_summary_claim_by_community"] = bsc
    return meta


class HybridModularTopology(BaseTopology):

    def __init__(self, *, num_communities: int = NUM_COMMUNITIES, **kwargs):
        super().__init__(**kwargs)
        self.num_communities = num_communities
        raw_ids              = [make_agent_id("agent", i) for i in range(self.num_agents)]
        self._communities    = _partition_agents(raw_ids, num_communities)
        self._bridges        = [comm[0] for comm in self._communities if comm]
        self._community_map: Dict[str, int] = {}
        for i, comm in enumerate(self._communities):
            for aid in comm:
                self._community_map[aid] = i

    def name(self) -> TopologyName:
        return TopologyName.HYBRID_MODULAR

    def agent_ids(self) -> List[str]:
        return [make_agent_id("agent", i) for i in range(self.num_agents)] + [INTEGRATOR_ID]

    def edge_list(self) -> List[Tuple[str, str]]:
        edges = []
        for comm in self._communities:
            for a in comm:
                for b in comm:
                    if a != b:
                        edges.append((a, b))
        for bridge in self._bridges:
            edges.append((bridge, INTEGRATOR_ID))
        return edges

    def build_graph(self) -> Any:
        builder = StateGraph(MASState)

        # ── Community node ─────────────────────────────────────────────

        def make_community_node(comm_idx: int, community: List[str]):
            bridge_id = community[0] if community else ""

            def comm_fn(state: MASState) -> Dict:
                new_outputs   = dict(state["agent_outputs"])
                new_influence = dict(state["influence"])
                new_claims    = dict(state["claims"])
                claim_ptr_updates: Dict[str, str] = {}

                for agent_id in community:
                    self._step += 1
                    self._maybe_snapshot()
                    is_bridge = (agent_id == bridge_id)
                    role      = "bridge" if is_bridge else "worker"
                    peers     = [a for a in community if a != agent_id]
                    peers_with_output = [p for p in peers if new_outputs.get(p)]

                    # PROPOSE_CLAIM only for first speaker (no peers yet → structural root)
                    if not peers_with_output:
                        ev_type             = EventType.PROPOSE_CLAIM
                        py_parent_claim_ids = []
                    else:
                        ev_type             = None  # extractor infers revise/contradict/endorse
                        py_parent_claim_ids = _ordered_claim_ids(
                            state, peers_with_output, new_claims
                        )

                    claim_id      = new_id("claim")
                    root_claim_id = _root_claim_from_parents(
                        state, py_parent_claim_ids, claim_id, new_claims
                    )

                    # prior_outputs: full text, no truncation
                    ptr_map = dict((state.get("metadata") or {}).get("latest_claim_by_agent") or {})
                    prior_outputs_list = [
                        (p, f"msg_hybrid_c{comm_idx}_{p}", ptr_map.get(p, ""), new_outputs[p])
                        for p in peers_with_output
                    ]

                    spec = AgentContextSpec(
                        task=state["task"],
                        agent_id=agent_id,
                        agent_role=role,
                        topology="hybrid_modular",
                        task_family=self.task_family,
                        local_goal=(
                            "Summarize your community's consensus for the integrator."
                            if is_bridge else
                            f"Reason using your community {comm_idx} peers' work."
                        ),
                        subtask_id=claim_id,
                        step=self._step,
                        neighbor_ids=peers,
                        prior_outputs=prior_outputs_list,
                        available_tools=self._tool_names,
                        extra_context=f"Community index: {comm_idx}",
                    )
                    user_content = build_context(spec)

                    # FIX 1: visible_neighbors inside _call_llm, not outside
                    output = self._call_llm(
                        agent_id=agent_id,
                        agent_role=role,
                        system_prompt=(
                            f"You are {'the bridge' if is_bridge else 'a member'} "
                            f"of community {comm_idx}."
                        ),
                        user_content=user_content,
                        event_type=ev_type,
                        claim_id=claim_id,
                        claim_depth=1,
                        parent_claim_ids=py_parent_claim_ids,
                        root_claim_id=root_claim_id,
                        visible_neighbors=peers,
                    )

                    new_outputs[agent_id]   = output
                    new_influence[agent_id] = new_influence.get(agent_id, 0) + len(peers)
                    # Structural claims dict — no semantic fields
                    new_claims[claim_id] = {
                        "agent_id":         agent_id,
                        "claim_id":         claim_id,
                        "parent_claim_ids": py_parent_claim_ids,
                        "root_claim_id":    root_claim_id,
                        "community":        comm_idx,  # kept as metadata; DAG ignores it
                        "depth":            1,
                        "text_hash":        text_hash(output),
                    }
                    claim_ptr_updates[agent_id] = claim_id
                    self._record_activation(self._step, [agent_id])

                # ── Bridge summary ─────────────────────────────────────
                self._step += 1
                summary_cid = new_id("claim")

                community_members     = [a for a in community if a != bridge_id]
                community_member_cids = _ordered_claim_ids(state, community_members, new_claims)

                # FIX 3: guard against double-counting bridge's own claim
                bridge_own_cid = claim_ptr_updates.get(bridge_id)
                if bridge_own_cid and bridge_own_cid not in community_member_cids:
                    community_member_cids = [bridge_own_cid] + community_member_cids

                summary_root = _root_claim_from_parents(
                    state, community_member_cids, summary_cid, new_claims
                )

                # prior_outputs: full text, no truncation
                comm_prior = [
                    (aid,
                     f"msg_hybrid_bsum_{aid}",
                     claim_ptr_updates.get(aid, ""),
                     new_outputs.get(aid, ""))
                    for aid in community if new_outputs.get(aid)
                ]

                spec_bridge = AgentContextSpec(
                    task=state["task"],
                    agent_id=bridge_id,
                    agent_role="bridge",
                    topology="hybrid_modular",
                    task_family=self.task_family,
                    local_goal=f"Synthesize community {comm_idx} outputs for the integrator.",
                    subtask_id=summary_cid,
                    step=self._step,
                    neighbor_ids=[INTEGRATOR_ID],
                    prior_outputs=comm_prior,
                    available_tools=self._tool_names,
                    extra_context=f"Community {comm_idx} bridge — relay to integrator.",
                )
                bridge_user_content = build_context(spec_bridge)

                # FIX 1: visible_neighbors inside _call_llm
                bridge_summary_output = self._call_llm(
                    agent_id=bridge_id,
                    agent_role="bridge",
                    system_prompt=(
                        f"You are the bridge of community {comm_idx}. "
                        "Synthesize for the integrator."
                    ),
                    user_content=bridge_user_content,
                    event_type=None,
                    claim_id=summary_cid,
                    claim_depth=2,
                    parent_claim_ids=community_member_cids,
                    root_claim_id=summary_root,
                    visible_neighbors=[INTEGRATOR_ID],
                )

                new_outputs[f"bridge_summary_{comm_idx}"] = bridge_summary_output
                new_influence[bridge_id] = new_influence.get(bridge_id, 0) + len(community)
                new_claims[summary_cid] = {
                    "agent_id":         bridge_id,
                    "claim_id":         summary_cid,
                    "parent_claim_ids": community_member_cids,
                    "root_claim_id":    summary_root,
                    "community":        comm_idx,
                    "depth":            2,
                    "text_hash":        text_hash(bridge_summary_output),
                }
                claim_ptr_updates[bridge_id] = summary_cid
                self._record_activation(self._step, [bridge_id])

                return {
                    "messages":      [AIMessage(content=f"Community {comm_idx} done", name=bridge_id)],
                    "agent_outputs": new_outputs,
                    "influence":     new_influence,
                    "claims":        new_claims,
                    "metadata":      _update_meta_claims(
                        state,
                        updates=claim_ptr_updates,
                        bridge_summary_updates={comm_idx: summary_cid},
                    ),
                    "step":          self._step,
                }
            return comm_fn

        # ── Integrator node ────────────────────────────────────────────

        def integrator(state: MASState) -> Dict:
            self._step += 1
            self._maybe_snapshot()

            bsc_map = state.get("metadata", {}).get("bridge_summary_claim_by_community", {})
            bridge_summary_cids = []
            for i in range(len(self._communities)):
                cid = bsc_map.get(str(i))
                if not cid and i < len(self._bridges):
                    cid = _get_latest_claim(state, self._bridges[i])
                if cid:
                    bridge_summary_cids.append(cid)

            # FIX 2: use integrator_cid as fallback, not a throwaway new_id()
            integrator_cid = new_id("claim")
            root_claim_id  = _root_claim_from_parents(
                state, bridge_summary_cids, integrator_cid
            )

            # prior_outputs: full text, no truncation
            prior_outputs_list = [
                (self._bridges[i] if i < len(self._bridges) else f"bridge_{i}",
                 f"msg_hybrid_int_{i}",
                 bridge_summary_cids[i] if i < len(bridge_summary_cids) else "",
                 state["agent_outputs"].get(f"bridge_summary_{i}", "(none)"))
                for i in range(len(self._communities))
            ]

            spec = AgentContextSpec(
                task=state["task"],
                agent_id=INTEGRATOR_ID,
                agent_role="synthesizer",
                topology="hybrid_modular",
                task_family=self.task_family,
                local_goal="Integrate all community bridge outputs into one final answer.",
                subtask_id=integrator_cid,
                step=self._step,
                neighbor_ids=list(self._bridges),
                prior_outputs=prior_outputs_list,
                available_tools=self._tool_names,
            )
            user_content = build_context(spec)

            # FIX 1: visible_neighbors inside _call_llm
            output = self._call_llm(
                agent_id=INTEGRATOR_ID,
                agent_role="integrator",
                system_prompt=(
                    "You are the global integrator. "
                    "Synthesize community summaries into one final answer."
                ),
                user_content=user_content,
                event_type=None,
                claim_id=integrator_cid,
                claim_depth=3,
                parent_claim_ids=bridge_summary_cids,
                root_claim_id=root_claim_id,
                visible_neighbors=list(self._bridges),
            )

            new_outputs   = dict(state["agent_outputs"])
            new_outputs[INTEGRATOR_ID] = output
            new_influence = dict(state["influence"])
            new_influence[INTEGRATOR_ID] = new_influence.get(INTEGRATOR_ID, 0) + len(self._bridges)
            new_claims    = dict(state["claims"])
            new_claims[integrator_cid] = {
                "agent_id":         INTEGRATOR_ID,
                "claim_id":         integrator_cid,
                "parent_claim_ids": bridge_summary_cids,
                "root_claim_id":    root_claim_id,
                "depth":            3,
                "text_hash":        text_hash(output),
            }
            self._record_activation(self._step, [INTEGRATOR_ID])

            return {
                "messages":      [AIMessage(content=output, name=INTEGRATOR_ID)],
                "agent_outputs": new_outputs,
                "influence":     new_influence,
                "claims":        new_claims,
                "metadata":      _update_meta_claims(
                    state, updates={INTEGRATOR_ID: integrator_cid}
                ),
                "final_answer":  output,
                "step":          self._step,
            }

        # ── Wire graph ─────────────────────────────────────────────────
        comm_nodes = [f"community_{i}" for i in range(len(self._communities))]
        for i, (cname, comm) in enumerate(zip(comm_nodes, self._communities)):
            builder.add_node(cname, make_community_node(i, comm))
        builder.add_node("integrator", integrator)
        builder.set_entry_point(comm_nodes[0])
        for i in range(len(comm_nodes) - 1):
            builder.add_edge(comm_nodes[i], comm_nodes[i + 1])
        builder.add_edge(comm_nodes[-1], "integrator")
        builder.add_edge("integrator", END)
        return builder.compile()