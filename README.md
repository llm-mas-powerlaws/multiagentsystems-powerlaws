# Do Agent Societies Develop Intellectual Elites?
### The Hidden Power Laws of Collective Cognition in LLM Multi-Agent Systems

**Kavana Venkatesh · Jiaming Cui** — Virginia Tech

[![Paper](https://img.shields.io/badge/arXiv-2604.02674-b31b1b.svg)](https://arxiv.org/abs/2604.02674)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Status: Active Development](https://img.shields.io/badge/status-active%20development-orange)

> **⚠️ This repository is under active development.** Code and data are being made available continuously. Not everything is in its final state yet. We are working toward a modular, reproducible release soon.

---

## What This Is

This is the codebase for our empirical study of coordination dynamics in LLM multi-agent systems. We analyze over **1.5 million coordination events** across tasks, topologies, and agent scales to uncover three coupled laws governing collective cognition:

- **H1 — Heavy-tailed cascades.** Coordination events follow truncated power-law distributions (α̂ ∈ [2.1, 2.7]) across all observables — delegation, revision, contradiction, merge, and Total Cognitive Effort (TCE).
- **H2 — Intellectual elites.** Cognitive effort concentrates in a small subset of agents through preferential attachment; the top-10% share grows by +24pp from N=8 to N=512.
- **H3 — Scaling of extremes.** The largest coordination cascades grow systematically with agent society size, tracking the EVT prediction ⟨x_max⟩ ∝ N^γ with γ̂ ≈ 0.85 for TCE.

All three arise from the same mechanism: an **integration bottleneck** where cascade expansion scales with system size but consolidation (merge) does not. We also introduce **Deficit-Triggered Integration (DTI)**, a targeted intervention that monitors imbalance and triggers integration, improving performance precisely where coordination fails.

---

## Paper

> *Do Agent Societies Develop Intellectual Elites? The Hidden Power Laws of Collective Cognition in LLM Multi-Agent Systems*  
> Kavana Venkatesh, Jiaming Cui  
> Preprint — [arXiv:2604.02674](https://arxiv.org/abs/2604.02674)

---

## Experimental Setup

| Dimension | Values |
|---|---|
| Agent scales | N ∈ {8, 16, 32, 64, 128, 256, 512} |
| Topologies | chain, star, tree, full_mesh, sparse_mesh, hybrid_modular, dynamic_reputation |
| Benchmarks | GAIA (QA/reasoning), SWE-bench (coding), REALM-Bench (planning), MultiAgentBench (coordination) |
| Seeds | 5 per configuration |
| Total events | ~1.5M coordination events |

Agents share a common LLM, prompt, tools, and task instances. Workloads are scaled using a benchmark-conditioned task expansion module (paper Appendix H). All coordination events are extracted post-hoc from raw traces — no event labels are injected into the agents.

---

## Repository Structure

```
mas-powerlaws/
├── scripts/
│   ├── smoke_test.py        # End-to-end test: imports → LLM → run → pipeline → power-law fit
│   └── run_sweep.py         # Full experimental sweep runner
│
├── src/
│   ├── context_builder.py              # Agent prompt assembly (BASE + topology + task + state)
│   ├── sparse_activation.py            # Activation tracking per topology
│   │
│   ├── prompts/
│   │   ├── base_prompt.py              # Fixed base prompt (same across all conditions)
│   │   ├── topology_addenda.py         # Per-topology communication constraints
│   │   └── task_addenda.py             # Per-task-family domain guidance
│   │
│   ├── loggers/
│   │   ├── event_bus.py                # Writes events.jsonl + snapshots.jsonl per run
│   │   ├── trace_schema.py             # TraceRow schema + AGENT_OUTPUT_FORMAT
│   │   └── schemas.py                  # RunConfig, RunMetadata, TopologyName, etc.
│   │
│   ├── event_extraction/
│   │   └── event_extractor.py          # Post-hoc event type classification from raw traces
│   │
│   ├── observables/
│   │   ├── dag_builder.py              # Builds subtask tree, claim DAG, cascades from traces
│   │   └── cascade_metrics.py          # Computes 5 power-law observables + agent influence
│   │
│   ├── metrics/
│   │   └── inequality.py               # Gini, top-k share, effective N
│   │
│   ├── tail_fitting/
│   │   └── powerlaw_fit.py             # MLE fitting: PL, TPL, LN, exp; LR tests; CCDF
│   │
│   ├── analysis/
│   │   └── run_pipeline.py             # annotate → DAG → observables → fit per run dir
│   │
│   ├── tools/
│   │   └── tools.py                    # Web search, calculator, python_exec, file reader
│   │
│   ├── topologies/
│   │   ├── base.py                     # BaseTopology: LLM call, trace logging, LangGraph state
│   │   ├── chain.py                    # Sequential pipeline
│   │   ├── star.py                     # Hub-and-spoke
│   │   ├── tree.py                     # Hierarchical tree
│   │   ├── full_mesh.py                # Fully connected mesh
│   │   ├── sparse_mesh.py              # Sparse random mesh
│   │   ├── hybrid.py                   # Modular bridge-integrator
│   │   └── dynamic_reputation.py       # Reputation-weighted routing
│   │
│   ├── execution/
│   │   └── graph_runner.py             # Orchestrates one run: task tree → topology → traces
│   │
│   └── benchmark_wrappers/
│       ├── task_expander.py            # Benchmark-conditioned workload expansion (paper §H)
│       ├── gaia.py
│       ├── swebench.py
│       ├── realm_bench.py
│       └── multiagentbench.py
│
└── data/                               # Run outputs (not committed)
    └── sweep/
        └── {benchmark}/{topology}/n{N}/s{seed}/{task_id}/
            ├── events.jsonl            # Raw TraceRows (one per agent turn)
            ├── task_tree.json          # Task expansion tree
            └── run_metadata.json       # H2 metrics + scores
```

---

## How It Works

### 1. Task Expansion

Before each run, `TaskExpander` expands a benchmark task into a tree of K×M interdependent subtasks with a sparse dependency DAG (paper §H). Agents work over these subtasks, not a single flat prompt. This keeps coordination demand balanced across agent scales.

### 2. Agent Execution

Each topology runs a LangGraph graph where agents call `_call_llm()`, which logs a `TraceRow` to `events.jsonl` for every agent turn. No event type labels are injected into the agent. The raw output JSON is logged as-is.

### 3. Post-Hoc Extraction

After a run, `event_extractor.annotate_event_types()` classifies each row into one of:
`propose_claim` · `revise_claim` · `contradict_claim` · `merge_claims` · `endorse_claim` · `delegate_subtask`

Then `dag_builder.build_all()` reconstructs the subtask tree and claim DAG, assigning `root_claim_id`, `claim_depth`, and cascade membership to every row.

### 4. Power-Law Observables

`cascade_metrics.extract_all_observables()` computes five sample lists from the annotated traces:

| Observable | Measures |
|---|---|
| Delegation cascade size | Subtask subtree size rooted at each delegation event |
| Revision wave size | Number of `revise_claim` events per revision chain |
| Contradiction burst size | Distinct agents contradicting the same parent claim |
| Merge fan-in | Number of parent claims per merge event |
| TCE | Total coordination events per root-centered cascade |

These are pooled across runs and fitted with truncated power law, log-normal, and exponential models by MLE, with Vuong likelihood-ratio tests for model comparison.

---

## Quickstart

### Requirements

```bash
python >= 3.11
pip install -r requirements.txt
# Key packages: langchain-openai, langgraph, powerlaw, scipy, numpy, pandas
```

### Environment

```bash
cp .env.example .env
# Add your OPENAI_API_KEY to .env
export $(grep -v '^#' .env | xargs)
```

### Smoke test (verify the full pipeline works)

```bash
cd mas-powerlaws
python scripts/smoke_test.py
```

This runs 45 checks: module imports, LLM connectivity, tools, a full chain N=8 run, events.jsonl validation, post-hoc pipeline, power-law fitting, all 7 topology graph builds, a mini sweep (4 topologies × 2 seeds), log inspection, and a pooled fit.

### Run a small subset

```bash
python scripts/run_sweep.py \
    --benchmarks gaia \
    --topologies chain star full_mesh \
    --scales 8 16 \
    --seeds 0 1
```

### Full sweep

```bash
python scripts/run_sweep.py --workers 4
# Resume after interruption:
python scripts/run_sweep.py --workers 4 --resume
```

Output goes to `data/sweep/`. Each run directory contains `events.jsonl`, `task_tree.json`, and `run_metadata.json`. A `sweep_manifest.jsonl` tracks every completed run and `sweep_errors.jsonl` logs failures.

---

## Key Design Decisions

**No event labels in prompts.** Agents are never told to emit `revise_claim` or `merge_claims`. All coordination event types are assigned post-hoc by `event_extractor.py` from the raw output structure — this is essential for uncontaminated measurement.

**Shared configuration across all conditions.** Same LLM, same base prompt, same tools, same task instances across all topologies, scales, and seeds. Differences in coordination emerge from topology structure alone.

**Trace-first architecture.** `events.jsonl` is the ground truth. The claim DAG, cascade metrics, power-law fits, and H2 metrics are all derived from it post-hoc with no LLM calls.

**TaskExpander scales workload with N.** K = min(5, |B|) seed tasks, M = ⌈N/(K×5)⌉ expanded tasks per seed, sparse dependency DAG. Agents per subtask stays ~5 across all N.

---

## Citation

```bibtex
@article{venkatesh2026powerlaws,
  title     = {Do Agent Societies Develop Intellectual Elites? The Hidden Power Laws of Collective Cognition in LLM Multi-Agent Systems},
  author    = {Venkatesh, Kavana and Cui, Jiaming},
  journal   = {arXiv preprint arXiv:2604.02674},
  year      = {2026}
}
```

---

## Status

This repository is under active development as the full sweep is being run and data is being cleaned up. Things that are complete and working:

- Full pipeline from raw LLM traces to power-law fits
- All 7 topology implementations
- Post-hoc event extraction and claim DAG builder
- Power-law fitting with MLE and model comparison
- Smoke test (45/45 passing)
- Full sweep runner with resume, parallel workers, progress tracking

Things still being updated:

- Data release (runs being collected)
- Visualization scripts
- DTI intervention implementation
- Per-benchmark evaluators beyond GAIA exact-match
- Full benchmark wrapper integrations

---

## License

MIT
