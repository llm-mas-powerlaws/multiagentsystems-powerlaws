"""
benchmark_wrappers/swebench.py
-------------------------------
Loads tasks from SWE-bench Verified via HuggingFace.
HF repo: princeton-nlp/SWE-bench_Verified  (500 tasks)
We use 10 tasks (2 easy, 3 medium, 5 hard) as per the portfolio.

SWE-bench tasks are GitHub issues paired with a test suite.
We use the issue description as the prompt; tests as the gold standard.
These tasks are uniquely suited to the tree + hybrid topologies since
they require planner → searcher → coder → reviewer role delegation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from execution.graph_runner import BenchmarkTask


# The 10 task IDs we selected for the portfolio
# (update after running build_manifest.py to get the actual IDs)
SELECTED_TASK_IDS: List[str] = []  # populated by task_curator.py


def _difficulty(row: Dict) -> str:
    """
    SWE-bench doesn't have explicit difficulty labels.
    We proxy via number of changed files or test count in metadata.
    """
    # Some versions have FAIL_TO_PASS counts
    ftp = row.get("FAIL_TO_PASS", row.get("fail_to_pass", []))
    n   = len(ftp) if isinstance(ftp, list) else 1
    if n <= 1:    return "easy"
    if n <= 3:    return "medium"
    return "hard"


def _make_prompt(row: Dict) -> str:
    """Build agent task prompt from SWE-bench issue fields."""
    repo       = row.get("repo", "unknown/repo")
    issue_text = row.get("problem_statement", row.get("text", ""))
    base_commit = row.get("base_commit", "")

    prompt = (
        f"Repository: {repo}\n"
        f"Base commit: {base_commit}\n\n"
        f"Issue description:\n{issue_text}\n\n"
        "Your task: Identify the root cause of this issue and propose a concrete fix. "
        "Specify which files need to be changed and what changes to make."
    )
    return prompt


def load_swebench_tasks(
    max_tasks:  int  = 10,
    split:      str  = "test",
    task_ids:   Optional[List[str]] = None,
) -> List[BenchmarkTask]:
    """
    Load up to max_tasks tasks from SWE-bench Verified.

    Parameters
    ----------
    max_tasks   Number of tasks to load.
    split       HF split: "test" (standard) or "dev".
    task_ids    If provided, load only these specific task IDs.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        raise ImportError("pip install datasets")

    ds = load_dataset(
        "princeton-nlp/SWE-bench_Verified",
        split=split,
            )

    # Build lookup by instance_id if filtering
    target_ids = set(task_ids or SELECTED_TASK_IDS)

    tasks: List[BenchmarkTask] = []
    for row in ds:
        instance_id = str(row.get("instance_id", row.get("id", "")))

        # If we have a target list, filter to it
        if target_ids and instance_id not in target_ids:
            continue

        prompt = _make_prompt(row)
        if not prompt.strip():
            continue

        # Gold: the patch text (for reference; actual scoring uses test suite)
        gold = row.get("patch", row.get("solution", ""))

        tasks.append(BenchmarkTask(
            task_id=instance_id,
            benchmark="SWE-bench",
            task_family="coding",
            difficulty=_difficulty(row),
            prompt=prompt,
            gold_answer=str(gold)[:500] if gold else None,   # truncate large patches
            metadata={
                "repo":        row.get("repo", ""),
                "base_commit": row.get("base_commit", ""),
                "fail_to_pass": row.get("FAIL_TO_PASS", []),
                "pass_to_pass": row.get("PASS_TO_PASS", []),
            },
            requires_tools=True,       # needs code search / file reading
            requires_synthesis=True,   # multi-file changes
        ))

        if len(tasks) >= max_tasks:
            break

    # If no target IDs set, just take first max_tasks
    if not target_ids:
        tasks = tasks[:max_tasks]

    return tasks
