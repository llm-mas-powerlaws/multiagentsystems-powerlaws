"""
benchmark_wrappers/realm_bench.py
----------------------------------
Loads tasks from REALM-Bench.
Data: ~/mas-powerlaws/REALM-Bench/datasets/

Confirmed structure (from diagnose_benchmarks.py):
  datasets/
    J1/ J2/ J3/ J4/          <- Job-shop scheduling families
    P1/ P2/ ... P10/          <- Project scheduling families
    Each folder contains disruptions/*.json files

Each JSON file has keys:
  instance_id, base_instance, disruptions, description, objective

Task families:
  J* -> "planning"  (job-shop scheduling under disruption)
  P* -> "coordination"  (project scheduling / resource allocation)

Difficulty inferred from:
  - Number of disruptions
  - Disruption type (weather/power_outage harder than machine_breakdown)
  - Folder family (J4/P10 tend to be harder instances)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from execution.graph_runner import BenchmarkTask

# ── Path resolution ───────────────────────────────────────────────────────
_MAS_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT  = _MAS_ROOT / "REALM-Bench" / "datasets"
REALM_DATASETS_PATH = Path(
    os.environ.get("REALM_DATASETS_PATH", str(_DEFAULT))
)

# Harder disruption types
_HARD_DISRUPTIONS = {"weather_effect", "power_outage", "emergency_shutdown"}
_EASY_DISRUPTIONS = {"machine_breakdown"}


def _family(folder_name: str) -> str:
    """J* folders = planning, P* folders = coordination."""
    if folder_name.upper().startswith("J"):
        return "planning"
    return "coordination"


def _difficulty(raw: Dict, folder_name: str) -> str:
    """
    Infer difficulty from disruption type and folder index.
    J1/P1 = easier instances, J4/P10 = harder instances.
    """
    disruptions = raw.get("disruptions", [])
    if not disruptions:
        return "easy"

    # Check disruption type
    d_type = disruptions[0].get("type", "") if disruptions else ""
    duration = int(disruptions[0].get("duration", 0)) if disruptions else 0

    # High folder numbers = harder benchmark instances
    try:
        num = int("".join(c for c in folder_name if c.isdigit()))
    except (ValueError, TypeError):
        num = 1

    if d_type in _HARD_DISRUPTIONS or duration > 60:
        return "hard"
    if num >= 3 or duration > 25:
        return "medium"
    return "easy"


def _make_prompt(raw: Dict, file_path: Path) -> str:
    """
    Build agent prompt from REALM task fields.

    REALM tasks are scheduling/planning problems with disruptions.
    We give agents the description + objective + disruption details.
    """
    instance_id  = raw.get("instance_id", file_path.stem)
    base_raw = raw.get("base_instance", "")
    base = base_raw if isinstance(base_raw, str) else ""
    description  = str(raw.get("description", "")).strip()
    objective    = str(raw.get("objective", "")).strip()
    disruptions  = raw.get("disruptions", [])

    # Format disruptions
    disrupt_lines = []
    for d in disruptions:
        d_type    = d.get("type", "unknown")
        start     = d.get("start_time", "?")
        duration  = d.get("duration", "?")
        impact    = d.get("impact", "")
        line      = f"  - {d_type}: starts at t={start}, duration={duration}"
        if impact:
            line += f", impact: {impact}"
        disrupt_lines.append(line)
    disrupt_str = "\n".join(disrupt_lines)

    prompt = (
        f"Scheduling problem: {instance_id}\n"
        f"Base instance: {base}\n\n"
    )
    if description:
        prompt += f"Description:\n{description}\n\n"
    if disrupt_str:
        prompt += f"Disruptions:\n{disrupt_str}\n\n"
    if objective:
        prompt += f"Objective:\n{objective}\n\n"
    prompt += (
        "Your task: given the above scheduling problem and disruptions, "
        "produce a revised schedule or coordination plan that minimises "
        "the impact of the disruption and achieves the stated objective."
    )
    return prompt.strip()


def _load_file(path: Path) -> Optional[Dict]:
    """Load and parse one REALM disruption JSON file."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and "instance_id" in raw:
            return raw
    except Exception:
        pass
    return None


def load_realm_tasks(
    max_tasks: int = 20,
    datasets_path: Optional[Path] = None,
) -> List[BenchmarkTask]:
    """
    Load one representative instance per REALM-Bench problem type.

    REALM has 14 problem types (P1-P14, J1-J4 etc). Each has 100 instances
    that are parameterized variations of the same base scenario. Sampling
    all instances gives near-duplicate prompts. Instead we take instance_001
    from each problem folder, giving one genuinely distinct planning task
    per problem type — matching the paper's reported pool of ~14 tasks.

    All tasks map to task_family="planning" to match the 4-type taxonomy:
    qa / coding / reasoning / planning.
    """
    root = datasets_path or REALM_DATASETS_PATH

    if not root.exists():
        raise FileNotFoundError(
            f"REALM-Bench datasets not found at {root}\n"
            f"Override with: export REALM_DATASETS_PATH=/path/to/datasets"
        )

    tasks: List[BenchmarkTask] = []

    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue

        # Find the first valid JSON in any subdirectory of this problem folder
        candidate = None
        for subdir in sorted(folder.iterdir()):
            if not subdir.is_dir():
                continue
            jsons = sorted(subdir.glob("*.json"))
            if jsons:
                candidate = (subdir.name, jsons[0])
                break

        if candidate is None:
            continue

        subdir_name, file_path = candidate
        raw = _load_file(file_path)
        if raw is None:
            continue

        prompt = _make_prompt(raw, file_path)
        if not prompt:
            continue

        task_id = f"{folder.name}_representative"

        tasks.append(BenchmarkTask(
            task_id=task_id,
            benchmark="REALM",
            task_family="planning",   # all REALM tasks = planning
            difficulty=_difficulty(raw, folder.name),
            prompt=prompt,
            gold_answer=None,
            metadata={
                "folder":        folder.name,
                "base_instance": raw.get("base_instance", ""),
                "disruption_types": [
                    d.get("type", "") for d in raw.get("disruptions", [])
                ],
                "source_file": str(file_path),
            },
            requires_tools=False,
            requires_synthesis=True,
        ))

        if len(tasks) >= max_tasks:
            break

    return tasks