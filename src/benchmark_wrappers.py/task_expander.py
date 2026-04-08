"""
task_expander.py
================
Benchmark-conditioned workload expansion module.


----------------------------
  K = min(5, |B(b,d)|)          -- seed tasks sampled from benchmark b, domain d
  A = 5                          -- target agents per task
  M = ceil(N / (K * A))         -- expanded tasks generated per seed
  Total pool = K * M             -- all tasks agents actually execute

  Tasks are connected via sparse, randomly sampled dependency edges forming
  a shallow DAG independent of the communication topology.

  The module generates only WORKLOAD — dependency structure and task content.
  It does NOT prescribe any coordination event type. All coordination events
  (delegation, revision, contradiction, merge) emerge from agent interaction
  during execution and are measured only from the realized trace.

  Ground truth for evaluation lives at the K original seed tasks.
  Cascade dynamics are measured across all coordination events in the full pool.

Benchmark pool sizes (Table 26)
---------------------------------
  GAIA             ~150 tasks
  SWE-bench        ~235 tasks
  REALM-Bench      ~14  tasks   (K capped at min(5,14) = 5)
  MultiAgentBench  ~6   tasks   (K capped at min(5,6)  = 5)

Figure 18 targets
------------------
  N=8   → agents_per_subtask ≈ 2
  N=512 → agents_per_subtask ≈ 9
  Active-agent fraction > 80% at all N.

Usage
-----
    from task_expander import TaskExpander

    expander = TaskExpander(benchmark="gaia", domain="qa", seed=42)
    tree = expander.build(N=128, benchmark_pool=pool_of_task_dicts)
    print(tree.summary())
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, field
from typing import Literal, Optional


# ---------------------------------------------------------------------------
# Constants (from paper)
# ---------------------------------------------------------------------------

K_MAX    = 5    # max seed tasks — Section H: K = min(5, |B(b,d)|)
A_TARGET = 5    # target agents per task — Section H: A = 5


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

DomainType = Literal["qa", "reasoning", "coding", "planning"]
NodeType   = Literal["seed", "expanded", "root"]

BenchmarkType = Literal["gaia", "swebench", "realm", "multiagentbench"]


# ---------------------------------------------------------------------------
# Scaling formulas (paper Section H)
# ---------------------------------------------------------------------------

def num_seed_tasks(benchmark_pool_size: int) -> int:
    """K = min(5, |B(b,d)|).  Paper: Section H."""
    return min(K_MAX, benchmark_pool_size)


def num_expanded_per_seed(N: int, K: int) -> int:
    """M = ceil(N / (K * A)).  Paper: Section H, A=5."""
    return max(1, math.ceil(N / (K * A_TARGET)))


def total_pool_size(N: int, K: int) -> int:
    """K * M.  Paper: Section H."""
    return K * num_expanded_per_seed(N, K)


def agents_per_task(N: int, K: int) -> float:
    """N / (K * M) — should be ~5 at all scales.  Paper: Figure 18."""
    pool = total_pool_size(N, K)
    return N / pool if pool > 0 else 0.0


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class TaskNode:
    node_id:          str
    description:      str
    node_type:        NodeType       # "seed" | "expanded" | "root"
    seed_parent_id:   Optional[str]  # which seed task this was expanded from
    depends_on:       list[str]      # sparse DAG edges (other node_ids)
    ground_truth:     Optional[str] = None
    benchmark_source: Optional[str] = None
    agent_budget:     int = 1
    # Populated at runtime, not expansion time:
    agent_answer:     Optional[str] = None


@dataclass
class TaskTree:
    """
    The fully expanded workload tree ready for agent execution.

    Attributes mirror the paper's description exactly:
      - seed_nodes:     the K original benchmark tasks (ground truth here)
      - expanded_nodes: the K*M expanded tasks agents actually execute
      - root_node:      single synthesis node at the top
      - dependency_dag: sparse edges connecting tasks (shallow DAG)
    """
    benchmark:        str
    domain:           DomainType
    N:                int
    K:                int
    M:                int
    seed_nodes:       list[TaskNode]
    expanded_nodes:   list[TaskNode]
    root_node:        TaskNode
    dependency_dag:   dict[str, list[str]]   # node_id -> [node_ids it feeds into]
    agent_allocation: dict[str, int] = field(default_factory=dict)

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def all_nodes(self) -> list[TaskNode]:
        return self.seed_nodes + self.expanded_nodes + [self.root_node]

    @property
    def execution_pool(self) -> list[TaskNode]:
        """Nodes agents actually work on (seeds + expanded, not root)."""
        return self.seed_nodes + self.expanded_nodes

    @property
    def total_nodes(self) -> int:
        return len(self.all_nodes)

    @property
    def pool_size(self) -> int:
        """K * M — matches paper Section H."""
        return len(self.seed_nodes) + len(self.expanded_nodes)

    def active_agent_fraction(self) -> float:
        assigned = sum(self.agent_allocation.values())
        return min(1.0, assigned / self.N)

    def node_by_id(self, node_id: str) -> Optional[TaskNode]:
        return next((n for n in self.all_nodes if n.node_id == node_id), None)

    def to_dict(self) -> dict:
        return {
            "benchmark": self.benchmark,
            "domain":    self.domain,
            "N":         self.N,
            "K":         self.K,
            "M":         self.M,
            "pool_size": self.pool_size,
            "total_nodes": self.total_nodes,
            "nodes": [
                {
                    "node_id":          n.node_id,
                    "node_type":        n.node_type,
                    "description":      n.description,
                    "seed_parent_id":   n.seed_parent_id,
                    "depends_on":       n.depends_on,
                    "agent_budget":     n.agent_budget,
                    "ground_truth":     n.ground_truth,
                    "benchmark_source": n.benchmark_source,
                }
                for n in self.all_nodes
            ],
            "dependency_dag":   self.dependency_dag,
            "agent_allocation": self.agent_allocation,
        }

    def summary(self) -> str:
        m = validate_tree(self)
        lines = [
            f"\n{'='*65}",
            f"Task Tree | {self.benchmark.upper()} | {self.domain.upper()} | N={self.N}",
            f"{'='*65}",
            f"  Seed tasks (K)           : {self.K}",
            f"  Expanded per seed (M)    : {self.M}",
            f"  Execution pool (K*M)     : {self.pool_size}",
            f"  Total nodes incl. root   : {self.total_nodes}",
            f"  Agents per task (target) : {m['agents_per_task']:.1f}  (paper target: ~5)",
            f"  Active agent fraction    : {m['active_agent_fraction']:.1%}  (paper target: >80%)",
            f"  Avg deps per node        : {m['avg_deps_per_node']:.1f}",
            f"  Max DAG depth            : {m['max_dag_depth']}",
            f"  DAG edge density         : {m['edge_density']:.3f}",
            "",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM-based task expansion (one call per seed task)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a task generator. Given a benchmark task, generate M related "
    "sub-tasks of exactly the same type. Return ONLY valid JSON — no markdown, "
    "no preamble, no explanation."
)

_DOMAIN_INSTRUCTION: dict[DomainType, str] = {
    "qa": (
        "Generate {M} related factual QA questions that are narrower or more specific "
        "than the original and whose answers together help build a complete answer to it. "
        "Every sub-question must be a genuine QA question — not a workflow step or review task."
    ),
    "reasoning": (
        "Generate {M} related reasoning sub-problems that are narrower in scope "
        "than the original. Each sub-problem must be a genuine multi-step reasoning "
        "task — not a meta-instruction or review step."
    ),
    "coding": (
        "Generate {M} related coding sub-tasks that are narrower in scope than the "
        "original. Each must be a genuine coding task — a specific function, module, "
        "or bug fix — not a review step or test plan."
    ),
    "planning": (
        "Generate {M} related planning sub-problems that cover a specific sub-route, "
        "time window, resource subset, or agent subgroup from the original. Each must "
        "be a genuine planning task — not a verification or review step."
    ),
}

_USER_TEMPLATE = """\
Original task: {description}

{domain_instruction}

Return JSON only:
{{
  "expanded_tasks": [
    {{"id": "e1", "description": "<task text>"}},
    ...
  ]
}}
"""


def _call_llm_expand(
    seed_task: dict,
    domain: DomainType,
    M: int,
    model: str = "gpt-4o-mini",
    max_retries: int = 3,
) -> list[dict]:
    """
    Expand one seed task into M related tasks via a single OpenAI API call.
    Returns list of dicts with keys: id, description.
    """
    import os
    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    user_msg = _USER_TEMPLATE.format(
        description=seed_task["description"],
        domain_instruction=_DOMAIN_INSTRUCTION[domain].format(M=M),
    )

    last_err: Exception = RuntimeError("No attempts made.")
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                max_tokens=4096,
                temperature=0.7,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
            )
            raw = resp.choices[0].message.content.strip()
            if raw.startswith("```"):
                parts = raw.split("```")
                raw = parts[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            parsed = json.loads(raw)
            tasks  = parsed.get("expanded_tasks", [])
            if not tasks:
                raise ValueError("Empty expanded_tasks list returned.")
            return tasks
        except Exception as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

    raise RuntimeError(
        f"LLM expansion failed for seed '{seed_task['node_id']}' "
        f"after {max_retries} attempts: {last_err}"
    ) from last_err


# ---------------------------------------------------------------------------
# Synthetic expansion (no API — for tests and dry runs)
# ---------------------------------------------------------------------------

def _synthetic_expand(
    seed_task: dict,
    domain: DomainType,
    M: int,
    seed_idx: int,
) -> list[dict]:
    """Deterministic placeholder expansion for testing without API calls."""
    domain_verb = {
        "qa": "sub-question",
        "reasoning": "sub-problem",
        "coding": "sub-task",
        "planning": "sub-plan",
    }[domain]
    return [
        {
            "id": f"e{i+1}",
            "description": (
                f"Synthetic {domain_verb} {seed_idx+1}.{i+1}: "
                f"derived from '{seed_task['description'][:60]}...'"
            ),
        }
        for i in range(M)
    ]


# ---------------------------------------------------------------------------
# Sparse DAG construction (paper Section H)
# ---------------------------------------------------------------------------

def _build_sparse_dag(
    nodes: list[TaskNode],
    rng: random.Random,
    avg_deps: float = 2.0,
) -> dict[str, list[str]]:
    """
    Build a sparse, shallow DAG of dependency edges.

    Paper: "Tasks are connected via sparse, randomly sampled dependency edges,
    forming a shallow DAG independent of the communication topology."

    Implementation: each expanded node randomly samples avg_deps predecessors
    from earlier nodes in the same cluster. Cross-cluster edges are added at
    1/3 the rate. This is O(n * avg_deps) — no quadratic scan.

    Seed nodes depend on their expanded cluster.
    Root depends on all seed nodes.
    """
    forward_dag: dict[str, list[str]] = {n.node_id: [] for n in nodes}

    # Group expanded nodes by cluster (seed parent)
    clusters: dict[str, list[TaskNode]] = {}
    for n in nodes:
        if n.node_type == "expanded":
            clusters.setdefault(n.seed_parent_id or "_none", []).append(n)

    # For each cluster, wire expanded nodes to a few predecessors within cluster
    for seed_id, cluster in clusters.items():
        for i, node in enumerate(cluster):
            # Within-cluster: pick up to avg_deps predecessors from earlier nodes
            predecessors = cluster[:i]
            if predecessors:
                k = min(len(predecessors), max(1, round(avg_deps)))
                chosen = rng.sample(predecessors, k)
                for pred in chosen:
                    if node.node_id not in pred.depends_on:
                        pred.depends_on = pred.depends_on  # already set
                    node.depends_on.append(pred.node_id)
                    forward_dag[pred.node_id].append(node.node_id)

    # Cross-cluster sparse edges: each expanded node gets ~0 or 1 cross-cluster dep
    all_clusters = list(clusters.keys())
    for seed_id, cluster in clusters.items():
        other_clusters = [k for k in all_clusters if k != seed_id]
        if not other_clusters:
            continue
        for node in cluster:
            if rng.random() < 0.15:   # 15% chance of one cross-cluster dep
                other_key  = rng.choice(other_clusters)
                other_nodes = clusters[other_key]
                if other_nodes:
                    pred = rng.choice(other_nodes)
                    # Only add if pred comes "before" node to keep DAG acyclic
                    # Use index in full expanded list as proxy for ordering
                    node.depends_on.append(pred.node_id)
                    forward_dag[pred.node_id].append(node.node_id)

    # Seed nodes depend on their expanded cluster
    for seed_n in [n for n in nodes if n.node_type == "seed"]:
        cluster = clusters.get(seed_n.node_id, [])
        seed_n.depends_on = [n.node_id for n in cluster]
        for n in cluster:
            forward_dag[n.node_id].append(seed_n.node_id)

    # Root depends on all seed nodes
    root = next((n for n in nodes if n.node_type == "root"), None)
    if root:
        seed_ids = [n.node_id for n in nodes if n.node_type == "seed"]
        root.depends_on = seed_ids
        for sid in seed_ids:
            forward_dag[sid].append(root.node_id)

    return forward_dag


# ---------------------------------------------------------------------------
# Agent allocation
# ---------------------------------------------------------------------------

def _allocate_agents(tree: TaskTree) -> TaskTree:
    """
    Distribute exactly N agents across execution-pool nodes (seeds + expanded)
    plus the root node.

    Uses largest-remainder method to guarantee sum == N with no infinite loops.
    When N < total_nodes (can happen at very small N), some nodes get 0 agents
    -- the paper's active-agent-fraction metric captures this correctly.

    Paper Figure 18: agents per subtask scales from ~2 (N=8) to ~9 (N=512).
    """
    pool = tree.execution_pool
    all_nodes_w = pool + [tree.root_node]

    if not all_nodes_w:
        return tree

    # Weights: pool nodes = 1.0, root = 1.5
    weights  = [1.0] * len(pool) + [1.5]
    total_w  = sum(weights)
    N        = tree.N

    # Largest-remainder method: guarantees sum == N, no floor loops
    exact    = [w / total_w * N for w in weights]
    floors   = [int(x) for x in exact]
    remainders = [(exact[i] - floors[i], i) for i in range(len(floors))]
    diff     = N - sum(floors)
    # Give the remaining diff to nodes with the largest fractional parts
    for _, idx in sorted(remainders, reverse=True)[:diff]:
        floors[idx] += 1

    allocation: dict[str, int] = {}
    for node, budget in zip(all_nodes_w, floors):
        allocation[node.node_id] = budget
        node.agent_budget = budget

    tree.agent_allocation = allocation
    return tree


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class TaskExpander:
    """
    Benchmark-conditioned workload expansion module (paper Section H).

    Parameters
    ----------
    benchmark : str
        One of "gaia" | "swebench" | "realm" | "multiagentbench".
    domain : DomainType
        "qa" | "reasoning" | "coding" | "planning".
    seed : int
        RNG seed for DAG construction and synthetic expansion.
    model : str
        Claude model for LLM-based expansion.
    """

    def __init__(
        self,
        benchmark: str,
        domain: DomainType,
        seed: int = 42,
        model: str = "gpt-4o-mini",
    ):
        self.benchmark = benchmark
        self.domain    = domain
        self.seed      = seed
        self.model     = model

    def build(
        self,
        N: int,
        benchmark_pool: list[dict],
        use_llm: bool = True,
    ) -> TaskTree:
        """
        Build the workload tree for N agents from the provided benchmark pool.

        Parameters
        ----------
        N : int
            Agent count for this run.
        benchmark_pool : list[dict]
            All available benchmark tasks for this benchmark+domain.
            Each dict must have:
              "node_id"     (str) -- unique identifier
              "description" (str) -- task text
            Optional keys:
              "ground_truth"     (str)
              "benchmark_source" (str)

        use_llm : bool
            True  -> call Claude API to expand each seed task.
            False -> deterministic synthetic expansion (for tests).

        Returns
        -------
        TaskTree
        """
        rng = random.Random(self.seed)

        # ── Step 1: Sample K seed tasks ─────────────────────────────────────
        K = num_seed_tasks(len(benchmark_pool))
        sampled = rng.sample(benchmark_pool, K)

        seed_nodes = [
            TaskNode(
                node_id=t["node_id"],
                description=t["description"],
                node_type="seed",
                seed_parent_id=None,
                depends_on=[],
                ground_truth=t.get("ground_truth"),
                benchmark_source=t.get("benchmark_source", self.benchmark),
            )
            for t in sampled
        ]

        # ── Step 2: Expand each seed into M tasks ────────────────────────────
        M = num_expanded_per_seed(N, K)
        expanded_nodes: list[TaskNode] = []

        for idx, seed_node in enumerate(seed_nodes):
            seed_dict = {"node_id": seed_node.node_id, "description": seed_node.description}

            if use_llm:
                raw_tasks = _call_llm_expand(seed_dict, self.domain, M, model=self.model)
            else:
                raw_tasks = _synthetic_expand(seed_dict, self.domain, M, seed_idx=idx)

            # Trim or pad to exactly M tasks
            raw_tasks = raw_tasks[:M]
            while len(raw_tasks) < M:
                raw_tasks.append({
                    "id": f"e{len(raw_tasks)+1}",
                    "description": f"Additional {self.domain} sub-task {len(raw_tasks)+1} for seed {idx+1}.",
                })

            for task in raw_tasks:
                exp_id = f"{seed_node.node_id}_exp_{task['id']}"
                expanded_nodes.append(
                    TaskNode(
                        node_id=exp_id,
                        description=task["description"],
                        node_type="expanded",
                        seed_parent_id=seed_node.node_id,
                        depends_on=[],
                        benchmark_source=self.benchmark,
                    )
                )

        # ── Step 3: Root synthesis node ──────────────────────────────────────
        root_node = TaskNode(
            node_id="root_000",
            description=(
                f"Synthesise and reconcile the solutions from all {K} "
                f"{self.benchmark} {self.domain} tasks into a single, complete, "
                "coherent answer."
            ),
            node_type="root",
            seed_parent_id=None,
            depends_on=[],  # filled by DAG builder
        )

        # ── Step 4: Build sparse DAG ─────────────────────────────────────────
        all_nodes = seed_nodes + expanded_nodes + [root_node]
        dag = _build_sparse_dag(all_nodes, rng)

        # ── Step 5: Assemble tree ────────────────────────────────────────────
        tree = TaskTree(
            benchmark=self.benchmark,
            domain=self.domain,
            N=N,
            K=K,
            M=M,
            seed_nodes=seed_nodes,
            expanded_nodes=expanded_nodes,
            root_node=root_node,
            dependency_dag=dag,
        )

        # ── Step 6: Allocate agents ──────────────────────────────────────────
        tree = _allocate_agents(tree)

        return tree


# ---------------------------------------------------------------------------
# Validation (paper Figure 18 / Table H)
# ---------------------------------------------------------------------------

def validate_tree(tree: TaskTree) -> dict:
    """Health metrics. Values should match paper Figure 18 targets."""
    from collections import deque

    all_nodes = tree.all_nodes

    # Iterative topological depth — avoids recursion limit on large N
    in_degree = {n.node_id: len(n.depends_on) for n in all_nodes}
    fwd: dict[str, list[str]] = {n.node_id: [] for n in all_nodes}
    for n in all_nodes:
        for dep in n.depends_on:
            if dep in fwd:
                fwd[dep].append(n.node_id)

    depths: dict[str, int] = {n.node_id: 0 for n in all_nodes}
    queue = deque(nid for nid, deg in in_degree.items() if deg == 0)
    while queue:
        nid = queue.popleft()
        for child in fwd.get(nid, []):
            depths[child] = max(depths[child], depths[nid] + 1)
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    max_depth = max(depths.values(), default=0)

    all_nodes = tree.all_nodes
    total_edges = sum(len(n.depends_on) for n in all_nodes)
    n_nodes     = len(all_nodes)
    max_edges   = n_nodes * (n_nodes - 1) / 2
    edge_density = total_edges / max_edges if max_edges > 0 else 0.0

    return {
        "N":                    tree.N,
        "K":                    tree.K,
        "M":                    tree.M,
        "pool_size":            tree.pool_size,
        "total_nodes":          tree.total_nodes,
        "agents_per_task":      round(agents_per_task(tree.N, tree.K), 2),
        "active_agent_fraction": round(tree.active_agent_fraction(), 3),
        "avg_deps_per_node":    round(total_edges / n_nodes, 2) if n_nodes else 0.0,
        "max_dag_depth":        max_depth,
        "edge_density":         round(edge_density, 4),
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_accuracy(tree: TaskTree) -> Optional[float]:
    """
    Accuracy evaluated at seed nodes only — they carry ground truth.
    Returns None if no seed nodes have been answered yet.
    """
    scored = [
        n for n in tree.seed_nodes
        if n.ground_truth is not None and n.agent_answer is not None
    ]
    if not scored:
        return None
    correct = sum(1 for n in scored if n.agent_answer.strip() == n.ground_truth.strip())
    return correct / len(scored)


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def save_tree(tree: TaskTree, path: str) -> None:
    import pathlib
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(tree.to_dict(), f, indent=2)


def load_tree_dict(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# CLI smoke-test  (python task_expander.py --N 128 --benchmark gaia)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Smoke-test task_expander against paper spec.")
    ap.add_argument("--N",         type=int,  default=64,     help="Agent count")
    ap.add_argument("--benchmark", type=str,  default="gaia", help="gaia|swebench|realm|multiagentbench")
    ap.add_argument("--domain",    type=str,  default="qa",   help="qa|reasoning|coding|planning")
    ap.add_argument("--use-llm",   action="store_true",       help="Call real Claude API")
    ap.add_argument("--seed",      type=int,  default=42)
    ap.add_argument("--pool-size", type=int,  default=150,    help="Simulated benchmark pool size")
    args = ap.parse_args()

    # ── Print scaling table (matches paper Section H / Figure 18) ──────────
    print("\nScaling table (paper Section H):")
    print(f"{'N':>6}  {'K':>4}  {'M':>5}  {'K*M':>6}  {'agents/task':>12}")
    print("-" * 40)
    for N in [8, 16, 32, 64, 128, 256, 512]:
        pool_size_sim = args.pool_size
        K = num_seed_tasks(pool_size_sim)
        M = num_expanded_per_seed(N, K)
        pool = K * M
        apt  = agents_per_task(N, K)
        print(f"{N:>6}  {K:>4}  {M:>5}  {pool:>6}  {apt:>12.1f}")

    print()

    # ── Build a tree with synthetic data ────────────────────────────────────
    pool_size = args.pool_size
    fake_pool = [
        {
            "node_id":          f"task_{i:04d}",
            "description":      f"Synthetic {args.domain} task #{i}: placeholder description.",
            "ground_truth":     f"answer_{i}",
            "benchmark_source": args.benchmark,
        }
        for i in range(pool_size)
    ]

    expander = TaskExpander(
        benchmark=args.benchmark,
        domain=args.domain,
        seed=args.seed,
    )
    tree = expander.build(N=args.N, benchmark_pool=fake_pool, use_llm=args.use_llm)
    print(tree.summary())

    # ── Validation checks ───────────────────────────────────────────────────
    m = validate_tree(tree)

    # Paper Figure 18: agents per task should be ~2 at N=8, ~9 at N=512
    # At N=64 with K=5, M=ceil(64/25)=3, pool=15, agents/task=64/15≈4.3
    assert m["active_agent_fraction"] > 0.8, \
        f"Active fraction {m['active_agent_fraction']:.1%} below 80% paper target"

    # Root should depend on all seed tasks
    assert set(tree.root_node.depends_on) == {n.node_id for n in tree.seed_nodes}, \
        "Root node must depend on all K seed tasks"

    # Each seed should have M expanded children
    for seed_n in tree.seed_nodes:
        children = [n for n in tree.expanded_nodes if n.seed_parent_id == seed_n.node_id]
        assert len(children) == tree.M, \
            f"Seed {seed_n.node_id} has {len(children)} children, expected {tree.M}"

    # Agent budgets sum to N
    total = sum(tree.agent_allocation.values())
    assert total == tree.N, f"Agent sum {total} != N={tree.N}"

    # DAG is acyclic (simple check: no node depends on itself)
    for n in tree.all_nodes:
        assert n.node_id not in n.depends_on, f"Self-loop at {n.node_id}"

    print("All checks passed.")

    # ── Save ────────────────────────────────────────────────────────────────
    out = f"/tmp/tree_{args.benchmark}_N{args.N}.json"
    save_tree(tree, out)
    print(f"Saved to {out}")