"""
benchmark_wrappers/marble.py
-----------------------------
Loads tasks from MARBLE (Multi-Agent Reasoning Benchmark for LLM Evaluation).
MARBLE lives at ~/mas-powerlaws/MARBLE.

Actual task file (confirmed by diagnose_benchmarks.py):
  marble/environments/coding_utils/assets/benchmark.jsonl
  100 tasks, keys: topic_category, coordination_category, content, requirements, id

Task families map:
  topic_category -> task_family
  coordination_category -> used to infer difficulty
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from execution.graph_runner import BenchmarkTask

# ── Path resolution ───────────────────────────────────────────────────────
# Priority: env var > sibling of mas-powerlaws > default
_MAS_ROOT = Path(__file__).resolve().parents[2]   # ~/mas-powerlaws
_DEFAULT  = _MAS_ROOT / "MARBLE"
MARBLE_ROOT = Path(os.environ.get("MARBLE_PATH", str(_DEFAULT)))
BENCHMARK_JSONL = MARBLE_ROOT / "marble" / "environments" / "coding_utils" / "assets" / "benchmark.jsonl"


def _family(raw: Dict) -> str:
    cat = str(raw.get("topic_category", "")).lower()
    if "action"    in cat: return "coordination"
    if "strategy"  in cat: return "planning"
    if "puzzle"    in cat: return "reasoning"
    if "db" in cat or "database" in cat: return "coding"
    if "simulation" in cat: return "planning"
    if "rpg"       in cat: return "coordination"
    return "coding"


def _difficulty(raw: Dict) -> str:
    coord = str(raw.get("coordination_category", "")).lower()
    reqs  = str(raw.get("requirements", ""))
    if coord == "test_case": return "easy"
    if len(reqs) > 400:      return "hard"
    return "medium"


def _make_prompt(raw: Dict) -> str:
    content  = str(raw.get("content", "")).strip()
    reqs_raw = raw.get("requirements", "")
    if isinstance(reqs_raw, list):
        reqs_list = reqs_raw
    else:
        try:
            reqs_list = ast.literal_eval(str(reqs_raw))
        except Exception:
            reqs_list = [str(reqs_raw)]
    reqs_str = "\n".join(f"  - {r}" for r in reqs_list if str(r).strip())
    prompt = content
    if reqs_str:
        prompt += f"\n\nRequirements:\n{reqs_str}"
    return prompt.strip()


def load_marble_tasks(
    max_tasks: int = 20,
    jsonl_path: Optional[Path] = None,
) -> List[BenchmarkTask]:
    """Load up to max_tasks tasks from MARBLE's benchmark.jsonl."""
    path = jsonl_path or BENCHMARK_JSONL

    if not path.exists():
        candidates = list(MARBLE_ROOT.rglob("benchmark.jsonl"))
        if candidates:
            path = candidates[0]
        else:
            raise FileNotFoundError(
                f"MARBLE benchmark.jsonl not found at {path}\n"
                f"MARBLE root: {MARBLE_ROOT}\n"
                f"Override with: export MARBLE_PATH=/path/to/MARBLE"
            )

    tasks: List[BenchmarkTask] = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue

            task_id = str(raw.get("id", f"marble_{i:04d}"))
            prompt  = _make_prompt(raw)
            if not prompt:
                continue

            tasks.append(BenchmarkTask(
                task_id=task_id,
                benchmark="MARBLE",
                task_family=_family(raw),
                difficulty=_difficulty(raw),
                prompt=prompt,
                gold_answer=None,
                metadata={
                    "topic_category":        raw.get("topic_category", ""),
                    "coordination_category": raw.get("coordination_category", ""),
                    "source_file":           str(path),
                },
                requires_tools=False,
                requires_synthesis=True,
            ))
            if len(tasks) >= max_tasks:
                break

    if not tasks:
        raise RuntimeError(f"Loaded 0 tasks from {path}.")
    return tasks
