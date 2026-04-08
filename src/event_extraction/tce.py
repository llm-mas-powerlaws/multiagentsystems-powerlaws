"""
event_extraction/tce.py
------------------------
Total Cognitive Effort (TCE) extraction at multiple granularities.

TCE operationalisations used in the paper:

  TCE_run      = sum(tokens) over all events in one run
  TCE_event    = tokens_total_event for a single event
  TCE_agent    = sum(tokens) for all events by one agent in one run
  TCE_cascade  = sum(tokens) for all events sharing a root_subtask_id
  TCE_wave     = sum(tokens) for all events sharing a revision_chain_id

All return List[float] ready for powerlaw.Fit().
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional


def _iter_events(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def tce_per_run(data_root: Path) -> List[float]:
    """Total tokens spent per run."""
    sizes = []
    for p in sorted(data_root.rglob("events.jsonl")):
        total = sum(
            (ev.get("tokens_total_event") or 0)
            for ev in _iter_events(p)
        )
        if total > 0:
            sizes.append(float(total))
    return sizes


def tce_per_event(data_root: Path) -> List[float]:
    """Token count per individual event (filters zeros)."""
    sizes = []
    for p in sorted(data_root.rglob("events.jsonl")):
        for ev in _iter_events(p):
            t = ev.get("tokens_total_event") or 0
            if t > 0:
                sizes.append(float(t))
    return sizes


def tce_per_agent_per_run(data_root: Path) -> List[float]:
    """Token count per (agent, run) pair."""
    agent_run: Dict[str, float] = defaultdict(float)
    for p in sorted(data_root.rglob("events.jsonl")):
        for ev in _iter_events(p):
            key = f"{ev.get('run_id','')}::{ev.get('agent_id','')}"
            agent_run[key] += ev.get("tokens_total_event") or 0
    return [v for v in agent_run.values() if v > 0]


def tce_per_cascade(data_root: Path) -> List[float]:
    """Token cost of each delegation subtask cascade."""
    cascade_tokens: Dict[str, float] = defaultdict(float)
    for p in sorted(data_root.rglob("events.jsonl")):
        for ev in _iter_events(p):
            root = ev.get("root_subtask_id") or ev.get("subtask_id")
            if root and ev.get("event_type") in ("delegate_subtask", "complete_subtask"):
                cascade_tokens[root] += ev.get("tokens_total_event") or 0
    return [v for v in cascade_tokens.values() if v > 0]


def tce_per_revision_wave(data_root: Path) -> List[float]:
    """Token cost of each revision wave chain."""
    wave_tokens: Dict[str, float] = defaultdict(float)
    for p in sorted(data_root.rglob("events.jsonl")):
        for ev in _iter_events(p):
            chain = ev.get("revision_chain_id")
            if chain:
                wave_tokens[chain] += ev.get("tokens_total_event") or 0
    return [v for v in wave_tokens.values() if v > 0]


def compute_all_tce(data_root: Path) -> Dict[str, List[float]]:
    """Return all TCE observables in one dict."""
    return {
        "tce_per_run":              tce_per_run(data_root),
        "tce_per_event":            tce_per_event(data_root),
        "tce_per_agent_per_run":    tce_per_agent_per_run(data_root),
        "tce_per_cascade":          tce_per_cascade(data_root),
        "tce_per_revision_wave":    tce_per_revision_wave(data_root),
    }
