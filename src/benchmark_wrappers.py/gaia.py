"""
benchmark_wrappers/gaia.py
--------------------------
Loads tasks from the GAIA benchmark via HuggingFace datasets.
GAIA: General AI Assistants benchmark — open-ended reasoning + tool use.
HF repo: gaia-benchmark/GAIA

Levels: 1 (easy), 2 (medium), 3 (hard)
We use the 2023 validation split (test labels are hidden).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from execution.graph_runner import BenchmarkTask


def _difficulty(level: Any) -> str:
    if isinstance(level, (int, float)):
        if level <= 1: return "easy"
        if level <= 2: return "medium"
        return "hard"
    s = str(level).lower()
    if s in ("1", "easy", "level 1"): return "easy"
    if s in ("3", "hard", "level 3"): return "hard"
    return "medium"


def _family(row: Dict) -> str:
    """GAIA tasks are primarily QA but can require planning and tool use."""
    q = str(row.get("Question", "")).lower()
    if any(w in q for w in ("steps", "plan", "sequence", "order")): return "planning"
    if any(w in q for w in ("code", "script", "program", "function")): return "coding"
    return "qa"


def load_gaia_tasks(
    max_tasks:   int  = 20,
    split:       str  = "validation",
    use_cache:   bool = True,
) -> List[BenchmarkTask]:
    """
    Load up to max_tasks tasks from GAIA via HuggingFace.

    Parameters
    ----------
    max_tasks   Cap on tasks returned (balanced across difficulty levels).
    split       HF dataset split: "validation" (labels available) or "test".
    use_cache   Pass to datasets.load_dataset for offline use.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        raise ImportError("pip install datasets  (HuggingFace datasets library)")

    ds = load_dataset(
        "gaia-benchmark/GAIA",
        "2023_all",
        split=split,
            )

    tasks: List[BenchmarkTask] = []
    for i, row in enumerate(ds):
        # GAIA schema: Question, Final answer, Level, task_id, Annotator Metadata
        question   = str(row.get("Question", ""))
        gold       = row.get("Final answer", row.get("answer", ""))
        level      = row.get("Level", 2)
        task_id    = str(row.get("task_id", f"gaia_{i:04d}"))

        if not question:
            continue

        # Append any file attachments mention to prompt
        file_name = row.get("file_name", "")
        if file_name:
            question = question + f"\n\n[Attached file: {file_name}]"

        tasks.append(BenchmarkTask(
            task_id=task_id,
            benchmark="GAIA",
            task_family=_family(row),
            difficulty=_difficulty(level),
            prompt=question,
            gold_answer=str(gold) if gold else None,
            metadata={
                "level": level,
                "file_name": file_name,
                "annotator_metadata": row.get("Annotator Metadata", {}),
            },
            requires_tools=bool(file_name),
            requires_synthesis=False,
        ))
        if len(tasks) >= max_tasks:
            break

    return tasks
