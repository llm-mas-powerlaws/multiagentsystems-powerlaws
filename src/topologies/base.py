"""
src/topologies/base.py

Unchanged except for two additions:

1. set_tool_names(names) — called by graph_runner after get_topology().
   Stores tool names so build_graph() node functions can forward them
   to AgentContextSpec.available_tools.

2. self._tool_names = [] in __init__ — safe default.

Everything else (MASState, canonicalize_event_type, _acall_llm, _call_llm,
_log_event, _log_snapshot, run, _initial_state) is preserved exactly.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
import traceback
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Annotated
import operator

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

from loggers.event_bus import EventBus
from loggers.schemas import (
    AgentEvent,
    EventType,
    TopologyName,
)


# ─────────────────────────────────────────────────────────────
# Reducer helpers
# ─────────────────────────────────────────────────────────────

def _merge_dicts(a: Dict, b: Dict) -> Dict:
    if not a: return b
    if not b: return a
    return {**a, **b}

def _keep_last(a: Any, b: Any) -> Any:
    return b if b is not None else a

def _max_int(a: int, b: int) -> int:
    return max(a or 0, b or 0)


# ─────────────────────────────────────────────────────────────
# Shared state type
# ─────────────────────────────────────────────────────────────

class MASState(TypedDict):
    messages:      Annotated[List[BaseMessage],  operator.add]
    task:          str
    current_agent: Annotated[str,                _keep_last]
    step:          Annotated[int,                _max_int]
    claims:        Annotated[Dict[str, Any],     _merge_dicts]
    subtasks:      Annotated[Dict[str, Any],     _merge_dicts]
    influence:     Annotated[Dict[str, float],   _merge_dicts]
    agent_outputs: Annotated[Dict[str, str],     _merge_dicts]
    final_answer:  Annotated[Optional[str],      _keep_last]
    metadata:      Annotated[Dict[str, Any],     _merge_dicts]


# ─────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────

def make_agent_id(role: str, index: int) -> str:
    return f"{role}_{index:03d}"

def text_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:12]

def new_id(prefix: str = "id") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"

def token_count_estimate(text: str) -> int:
    return max(1, len(text) // 4)


# ─────────────────────────────────────────────────────────────
# Enum normalization
# ─────────────────────────────────────────────────────────────

_VALID_SUPPORT_TYPES = {"agreement", "reuse", "validation", "coalition_support"}

_SUPPORT_TYPE_MAP = {
    "agree": "agreement", "agrees": "agreement", "support": "agreement",
    "supports": "agreement", "endorse": "agreement", "endorses": "agreement",
    "confirm": "agreement", "confirms": "agreement", "concur": "agreement",
    "valid": "validation", "validate": "validation", "validates": "validation",
    "verify": "validation", "verifies": "validation", "verified": "validation",
    "check": "validation", "functional": "validation", "correct": "validation",
    "accurate": "validation", "description": "validation",
    "functional description": "validation",
    "reuse": "reuse", "reuses": "reuse", "reusing": "reuse",
    "recycle": "reuse", "repurpose": "reuse", "prior reasoning": "reuse",
    "coalition": "coalition_support", "coalition_support": "coalition_support",
    "group": "coalition_support", "collective": "coalition_support",
}

def normalize_support_type(raw: Any, event_type: Any = None) -> Optional[str]:
    if raw is None:
        return None
    if raw in _VALID_SUPPORT_TYPES:
        return raw
    lowered = str(raw).lower().strip()
    if lowered in _VALID_SUPPORT_TYPES:
        return lowered
    if lowered in _SUPPORT_TYPE_MAP:
        return _SUPPORT_TYPE_MAP[lowered]
    for keyword, mapped in _SUPPORT_TYPE_MAP.items():
        if keyword in lowered:
            return mapped
    ev = str(event_type).lower() if event_type else ""
    if "endorse" in ev:
        return "agreement"
    if "merge" in ev:
        return "coalition_support"
    return None


_VALID_CLAIM_TYPES = {
    "initial_claim", "intermediate_claim", "final_claim",
    "contradiction", "revision", "merge_output",
}

def normalize_claim_type(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    if raw in _VALID_CLAIM_TYPES:
        return raw
    s = str(raw).split(".")[-1].lower().strip()
    if s in _VALID_CLAIM_TYPES:
        return s
    if "final" in s:      return "final_claim"
    if "initial" in s or "first" in s: return "initial_claim"
    if "contradict" in s: return "contradiction"
    if "revis" in s:      return "revision"
    if "merge" in s:      return "merge_output"
    return "intermediate_claim"


_VALID_SUBTASK_TYPES = {
    "decomposition", "critique", "synthesis",
    "verification", "retrieval", "execution",
}

def normalize_subtask_type(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    if raw in _VALID_SUBTASK_TYPES:
        return raw
    s = str(raw).split(".")[-1].lower().strip()
    if s in _VALID_SUBTASK_TYPES:
        return s
    if "decomp" in s or "split" in s or "break" in s: return "decomposition"
    if "critiqu" in s or "review" in s:               return "critique"
    if "synth" in s or "final" in s:                  return "synthesis"
    if "verif" in s:                                   return "verification"
    if "retriev" in s or "search" in s or "fetch" in s: return "retrieval"
    if "execut" in s or "implement" in s:              return "execution"
    return None


# ─────────────────────────────────────────────────────────────
# Canonical event type
# ─────────────────────────────────────────────────────────────

def canonicalize_event_type(
    hinted_event: Any,
    *,
    parent_claim_ids: List[str],
    claim_type: Optional[str],
    endorsed_claim_id: Optional[str],
    merge_num_inputs: Optional[int],
    target_agent_id: Optional[str],
    parent_subtask_id: Optional[str],
    py_rev_chain: Optional[str],
) -> str:
    # DEPRECATED: Not called by _acall_llm. Kept for legacy _log_event calls
    # (DELEGATE_SUBTASK zero-token events). Event classification for main
    # experiments is done post-hoc by event_extraction/event_extractor.py.
    ev = str(hinted_event).lower().replace("eventtype.", "").strip()
    n  = len(parent_claim_ids)

    if n >= 2:
        return "merge_claims"
    if endorsed_claim_id:
        return "endorse_claim"
    if target_agent_id and parent_subtask_id is not None and "delegate" in ev:
        return "delegate_subtask"
    if "contradict" in ev and n == 1:
        return "contradict_claim"
    if py_rev_chain or (n == 1 and "contradict" not in ev):
        return "revise_claim"
    if claim_type == "final_claim":
        return "finalize_answer"
    return "propose_claim"


# ─────────────────────────────────────────────────────────────
# Retry helper
# ─────────────────────────────────────────────────────────────

async def _call_with_retry(fn, max_retries: int = 5, base_delay: float = 1.0):
    for attempt in range(max_retries):
        try:
            return await fn()
        except Exception as e:
            err = str(e).lower()
            is_rate_limit = any(x in err for x in [
                "429", "rate limit", "ratelimit", "rate_limit",
                "too many requests", "quota", "overloaded",
            ])
            is_transient = any(x in err for x in [
                "timeout", "connection", "502", "503", "504", "server error",
            ])
            if (is_rate_limit or is_transient) and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                if is_rate_limit:
                    delay = max(delay, 5.0)
                await asyncio.sleep(delay)
                continue
            raise


# ─────────────────────────────────────────────────────────────
# Abstract topology
# ─────────────────────────────────────────────────────────────

class BaseTopology(ABC):

    def __init__(
        self,
        *,
        llm,
        bus: EventBus,
        run_id: str,
        benchmark: str,
        task_id: str,
        task_family: str,
        difficulty: str,
        num_agents: int,
        seed: int,
        architecture: str,
        snapshot_every: int = 5,
    ) -> None:
        self.llm            = llm
        self.bus            = bus
        self.run_id         = run_id
        self.benchmark      = benchmark
        self.task_id        = task_id
        self.task_family    = task_family
        self.difficulty     = difficulty
        self.num_agents     = num_agents
        self.seed           = seed
        self.architecture   = architecture
        self.snapshot_every = snapshot_every
        self._step          = 0
        self._tool_names: List[str] = []   # ADDED: populated by graph_runner
        from sparse_activation import ActivationTracker
        self._activation    = ActivationTracker(num_agents=num_agents)

    # ADDED: called by graph_runner after get_topology()
    def set_tool_names(self, tool_names: List[str]) -> None:
        """
        Set the benchmark-native tool names for this topology.
        Called by GraphRunner before topo.run() so node functions can
        forward tool_names to AgentContextSpec.available_tools.
        Same tools for all agents, all roles — policy enforced in tools.py.
        """
        self._tool_names = list(tool_names)

    @abstractmethod
    def name(self) -> TopologyName: ...

    @abstractmethod
    def build_graph(self) -> Any: ...

    @abstractmethod
    def edge_list(self) -> List[Tuple[str, str]]: ...

    def edge_weights(self) -> Dict[str, float]:
        return {}

    def agent_ids(self) -> List[str]:
        return []

    # ── Logging helpers ────────────────────────────────────────

    def _log_event(self, **kwargs) -> None:
        try:
            event = AgentEvent(
                run_id=self.run_id,
                benchmark=self.benchmark,
                task_id=self.task_id,
                task_family=self.task_family,
                difficulty=self.difficulty,
                topology=self.name(),
                architecture=self.architecture,
                num_agents=self.num_agents,
                seed=self.seed,
                step_id=self._step,
                **kwargs,
            )
            self.bus.log_event(event)
        except Exception:
            tb = traceback.format_exc()
            try:
                safe_event = AgentEvent(
                    run_id=self.run_id,
                    benchmark=self.benchmark,
                    task_id=self.task_id,
                    task_family=self.task_family,
                    difficulty=self.difficulty,
                    topology=self.name(),
                    architecture=self.architecture,
                    num_agents=self.num_agents,
                    seed=self.seed,
                    step_id=self._step,
                    agent_id=kwargs.get("agent_id", "unknown"),
                    agent_role=kwargs.get("agent_role", "worker"),
                    event_type="propose_claim",
                    action_success=False,
                )
                self.bus.log_event(safe_event)
            except Exception:
                pass

    def _log_snapshot(
        self,
        step: Optional[int] = None,
        active_agents: Optional[List[str]] = None,
    ) -> None:
        s = step if step is not None else self._step
        extra_kwargs = {}
        if active_agents is not None:
            extra_kwargs = self._activation.snapshot_kwargs(s, active_agents)
        self.bus.log_snapshot(
            run_id=self.run_id,
            step=s,
            topology=self.name(),
            agent_ids=self.agent_ids(),
            edge_list=self.edge_list(),
            edge_weights=self.edge_weights(),
            **extra_kwargs,
        )

    def _maybe_snapshot(
        self, active_agents: Optional[List[str]] = None
    ) -> None:
        if self._step % self.snapshot_every == 0:
            self._log_snapshot(active_agents=active_agents)

    def _record_activation(self, step_idx: int, active: List[str]) -> List[str]:
        return self._activation.record_step(step_idx, active)

    # ── Core LLM call ─────────────────────────────────────────

    async def _acall_llm(
        self,
        agent_id:     str,
        agent_role:   str,
        system_prompt: str,
        user_content:  str,
        event_type:    Any = None,   # hint only — NOT logged; post-hoc extraction owns this
        **event_kwargs,
    ) -> str:
        """
        Thin LLM call + raw trace logging.

        What this does:
          1. Call the LLM with retry
          2. Parse JSON output for text content only (answer, reasoning, confidence)
          3. Extract raw structural fields from event_kwargs (claim_id, parent_claim_ids,
             subtask_id, etc.) — these are Python-computed lineage, not event labels
          4. Log a raw TraceRow to events.jsonl via bus.log()
          5. Return the answer text

        What this does NOT do:
          - canonicalize_event_type()        → event_extractor.py
          - normalize_claim/subtask/support  → event_extractor.py
          - assign merge_id, revision_chain_id, contradiction_group_id → dag_builder.py
          - decide revise vs contradict vs endorse vs merge → event_extractor.py
          - assign claim_status              → event_extractor.py

        event_type kwarg is accepted for backward compatibility with topology
        call sites (e.g. DELEGATE_SUBTASK, PROPOSE_CLAIM) but is stored as a
        hint field only. The post-hoc extractor may override it.
        Topology files should pass event_type=None for non-root claim events.
        """
        import json as _json
        import uuid as _uuid

        t0 = time.time()

        from context_builder import MAX_COMPLETION_TOKENS
        invoke_kwargs = {"max_tokens": MAX_COMPLETION_TOKENS}
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content},
        ]

        async def _do_call():
            # Always use sync invoke via executor to avoid nested event loop issues
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, lambda: self.llm.invoke(messages, **invoke_kwargs)
            )

        try:
            response = await _call_with_retry(_do_call, max_retries=5, base_delay=1.0)
            raw_text = response.content if hasattr(response, "content") else str(response)
            success  = True
        except Exception as e:
            raw_text = ""
            success  = False

        latency_ms = (time.time() - t0) * 1000
        in_tokens  = token_count_estimate(system_prompt + user_content)
        out_tokens = token_count_estimate(raw_text)

        # ── Parse JSON for text content only ─────────────────────────
        # Extract answer/reasoning/confidence from agent output.
        # Do NOT extract event semantics here — that belongs to event_extractor.
        answer     = raw_text
        reasoning  = ""
        confidence = None
        raw_coordination_signals: dict = {}
        raw_claim_id_from_llm:    str  = ""
        raw_parent_claim_ids_from_llm: list = []

        text = raw_text.strip()
        for fence in ("```json", "```"):
            if text.startswith(fence):
                text = text[len(fence):]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            try:
                parsed = _json.loads(text[start:end+1])
                content = parsed.get("content") or {}
                answer     = content.get("answer") or parsed.get("answer") or raw_text
                reasoning  = content.get("reasoning") or parsed.get("reasoning") or ""
                confidence = content.get("confidence") or parsed.get("confidence")
                raw_coordination_signals = parsed.get("coordination_signals") or {}
                # Capture agent-generated claim lineage hints (for provenance only)
                raw_claim_id_from_llm         = parsed.get("claim_id") or ""
                raw_parent_claim_ids_from_llm = parsed.get("parent_claim_ids") or []
            except Exception:
                pass

        # ── Log requested tool calls (single-turn opportunistic protocol) ─
        # _acall_llm does NOT execute tools here. Tool execution is the
        # orchestrator's responsibility and happens outside this primitive.
        #
        # Protocol:
        #   answer      = model output only, never mutated with tool results
        #   tools_used  = tools the agent requested (success=None = not yet run)
        #   evidence_refs = filled by orchestrator after execution
        #
        # Topology nodes call run_tool() after _call_llm() returns, then feed
        # results into the next turn's prior_outputs as evidence context.
        # This is single-turn opportunistic tool use, not an iterative loop.
        tools_used_list: list = []
        evidence_refs_list: list = []

        raw_tool_calls = []
        if start != -1 and end != -1:
            try:
                import json as _json2
                _p = _json2.loads(text[start:end+1])
                raw_tool_calls = _p.get("tool_calls") or []
            except Exception:
                pass

        for tc in raw_tool_calls:
            if not isinstance(tc, dict):
                continue
            tool_name = tc.get("tool_name") or tc.get("name", "")
            if tool_name and tool_name in self._tool_names:
                tools_used_list.append({
                    "tool":    tool_name,
                    "success": None,  # requested but not yet executed
                })
        # evidence_refs populated by orchestrator after execution, not here

        # ── Extract Python-computed structural fields ──────────────────
        # These are topology-computed lineage fields — not event labels.
        # They are logged as-is into the TraceRow for dag_builder to use.
        claim_id         = event_kwargs.pop("claim_id",             None) or new_id("claim")
        parent_claim_ids = event_kwargs.pop("parent_claim_ids",     []) or []
        root_claim_id    = event_kwargs.pop("root_claim_id",        None)
        claim_depth      = event_kwargs.pop("claim_depth",          None)
        subtask_id       = event_kwargs.pop("subtask_id",           None) or new_id("sub")
        parent_subtask_id = event_kwargs.pop("parent_subtask_id",   None)
        root_subtask_id  = event_kwargs.pop("root_subtask_id",      None)
        subtask_depth    = event_kwargs.pop("subtask_depth",        None)
        subtask_status   = event_kwargs.pop("subtask_status",       None)
        subtask_assigned_by = event_kwargs.pop("subtask_assigned_by", None)
        subtask_assigned_to = event_kwargs.pop("subtask_assigned_to", None)
        visible_neighbors   = event_kwargs.pop("visible_neighbors",   []) or []
        # Consume remaining known kwargs silently (not logged — post-hoc pipeline owns them)
        for _drop in ("claim_type", "claim_status", "root_subtask_id",
                      "subtask_type", "subtask_assigned_by", "subtask_assigned_to",
                      "revision_chain_id", "trigger_claim_id", "reason_for_revision",
                      "contradiction_group_id", "merge_id", "merge_parent_claim_ids",
                      "merge_num_inputs", "merge_num_unique_agents",
                      "merge_synthesizer_agent_id", "endorsed_claim_id",
                      "endorsement_reason", "support_type", "confidence_score",
                      "target_agent_id", "num_agents_involved", "local_group_id",
                      "same_subtask_branch", "agent_influence_score_so_far",
                      "agent_degree_so_far", "merge_parent_claim_ids"):
            event_kwargs.pop(_drop, None)

        # event_type hint: store as-is for structural events (DELEGATE, PROPOSE at root)
        # For all other events pass None — event_extractor owns classification.
        event_type_hint = str(event_type).split(".")[-1].lower() if event_type is not None else None

        # ── Log raw TraceRow ──────────────────────────────────────────
        message_id = f"msg_{_uuid.uuid4().hex[:12]}"
        row = {
            # Run metadata
            "run_id":       self.run_id,
            "benchmark":    self.benchmark,
            "task_id":      self.task_id,
            "task_family":  self.task_family,
            "topology":     str(self.name().value) if hasattr(self.name(), "value") else str(self.name()),
            "seed":         self.seed,
            "num_agents":   self.num_agents,
            # Turn metadata
            "step_id":      self._step,
            "timestamp":    time.time(),
            "agent_id":     agent_id,
            "agent_role":   agent_role,
            "role":         agent_role,  # alias for TraceRow compat
            # Identifiers (orchestrator-assigned)
            "message_id":   message_id,
            "neighbor_ids": visible_neighbors,
            # Subtask lineage
            "subtask_id":           subtask_id,
            "parent_subtask_id":    parent_subtask_id,
            "assigned_agent":       subtask_assigned_to,
            "subtask_status":       subtask_status,
            # Claim lineage (Python-computed topology structure)
            "claim_id":          claim_id,
            "parent_claim_ids":  parent_claim_ids,
            "root_claim_id":     root_claim_id,   # None at emit time; dag_builder fills it
            "claim_depth":       claim_depth,
            # Content
            "reasoning_text":    reasoning,
            "final_answer_text": None,  # orchestrator sets this; not role-inferred here
            "message_length_chars": len(raw_text),
            "message_length":       len(raw_text),  # alias kept for backward compat
            "confidence":        float(confidence) if confidence is not None else None,
            # Provenance
            # claims_visible:    topology-computed parent_claim_ids (what agent could see)
            # claims_referenced: LLM-reported claim IDs (what agent chose to cite)
            # These can diverge; extractor uses both as signals.
            # references_used: message IDs cited by agent in provenance field
            "references_used":   (parsed.get("provenance") or {}).get("references_used", [])
                                 if start != -1 and end != -1 and 'parsed' in dir() else [],
            "claims_visible":    parent_claim_ids,
            "claims_referenced": raw_parent_claim_ids_from_llm or [],  # [] if LLM output not JSON
            "tools_used":        tools_used_list,
            "evidence_refs":     evidence_refs_list,
            # Coordination signals (raw from agent — post-hoc extractor reads these)
            "requested_subtask_creation":    raw_coordination_signals.get("requested_subtask_creation", False),
            "proposed_assignee":             raw_coordination_signals.get("proposed_assignee"),
            "synthesis_of_multiple_inputs":  raw_coordination_signals.get("synthesis_of_multiple_inputs", False),
            "explicit_disagreement_with":    raw_coordination_signals.get("explicit_disagreement_with", []),
            "explicit_correction_of":        raw_coordination_signals.get("explicit_correction_of", []),
            "supports_claims":               raw_coordination_signals.get("supports_claims", []),
            # Event type hint (structural events only; None for claim events)
            "event_type_hint":   event_type_hint,
            # Post-hoc fields — None at emit time; filled by event_extractor + dag_builder
            "event_type":              None,
            "claim_status":            None,
            "revision_chain_id":       None,
            "contradiction_group_id":  None,
            "merge_id":                None,
            "secondary_root_claim_ids": [],
            "subtask_depth":           subtask_depth,
            # Perf
            "latency_ms":        round(latency_ms, 2),
            "action_success":    success,
            "tokens_input":      in_tokens,
            "tokens_output":     out_tokens,
        }

        # Write raw trace row
        self.bus.log(row)

        return answer


    def _call_llm(
        self,
        agent_id:      str,
        agent_role:    str,
        system_prompt: str,
        user_content:  str,
        event_type:    Any = None,   # hint only — see _acall_llm docstring
        **event_kwargs,
    ) -> str:
        """Sync wrapper around _acall_llm."""
        coro = self._acall_llm(
            agent_id=agent_id,
            agent_role=agent_role,
            system_prompt=system_prompt,
            user_content=user_content,
            event_type=event_type,
            **event_kwargs,
        )
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(asyncio.run, coro)
                    return future.result()
            else:
                return loop.run_until_complete(coro)
        except RuntimeError:
            return asyncio.run(coro)

    # ── Run entry point ────────────────────────────────────────

    def run(self, task: str, config: Optional[Dict] = None) -> str:
        app    = self.build_graph()
        init   = self._initial_state(task)
        result = app.invoke(init, config=config or {})
        self._log_snapshot(step=self._step)
        if not self._activation._per_step:
            all_ids = self.agent_ids()
            if all_ids:
                for i, aid in enumerate(all_ids):
                    self._activation.record_step(i, [aid])
        self._activation_summary = self._activation.summary()
        return result.get("final_answer", "")

    def _initial_state(self, task: str) -> MASState:
        return MASState(
            messages=[HumanMessage(content=task)],
            task=task,
            current_agent="",
            step=0,
            claims={},
            subtasks={},
            influence={},
            agent_outputs={},
            final_answer=None,
            metadata={},
        )