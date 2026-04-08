# Do Agent Societies Develop Intellectual Elites?
### The Hidden Power Laws of Collective Cognition in LLM Multi-Agent Systems

**Kavana Venkatesh · Jiaming Cui** — Virginia Tech

[![Paper](https://img.shields.io/badge/arXiv-2604.02674-b31b1b.svg)](https://arxiv.org/abs/2604.02674)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Status: Active Development](https://img.shields.io/badge/status-active%20development-orange)

> **This repository is under active development.** We are working toward full release of all our data and code soon for reproducibility. Please monitor this space for updates.

---

## Abstract

Large Language Model (LLM) multi-agent systems are increasingly deployed as interacting agent societies, yet scaling these systems often yields diminishing or unstable returns, the causes of which remain poorly understood. We present the first large-scale empirical study of coordination dynamics in LLM-based multi-agent systems, introducing an atomic event-level formulation that reconstructs reasoning as cascades of coordination. Analyzing over **1.5 Million** interactions across tasks, topologies, and scales, we uncover *three* coupled laws: coordination follows heavy-tailed cascades, concentrates via preferential attachment into intellectual elites, and produces increasingly frequent extreme events as system size grows. We show that these effects are coupled through a single structural mechanism: an **integration bottleneck**, in which coordination expansion scales with system size while consolidation does not, producing large but weakly integrated reasoning processes. To test this mechanism, we introduce **Deficit-Triggered Integration (DTI)**, which selectively increases integration under imbalance. DTI improves performance precisely where coordination fails, without suppressing large-scale reasoning. Together, our results establish quantitative laws of collective cognition and identify coordination structure as a fundamental, previously unmeasured axis for understanding and improving scalable multi-agent intelligence.



---

## Paper

> *Do Agent Societies Develop Intellectual Elites? The Hidden Power Laws of Collective Cognition in LLM Multi-Agent Systems*  
> Kavana Venkatesh, Jiaming Cui  
> [arXiv:2604.02674](https://arxiv.org/abs/2604.02674)

---

## Key Findings

**H1 — Heavy-tailed coordination cascades.** Coordination event sizes follow truncated power-law distributions with α̂ ∈ (2, 3) across all observables (delegation, revision, contradiction, merge, TCE), tasks, topologies, and agent scales. Truncated power law provides a significantly better fit than log-normal and exponential alternatives (p < 0.05, Vuong test).

**H2 — Emergence of intellectual elites.** Cognitive effort concentrates endogenously in a small subset of agents through preferential attachment in claim selection. The top-10% excess share reaches +24pp at large N, and this gap widens systematically with scale — elite formation is a scale-amplified structural property, not a finite-size artifact.

**H3 — Scaling of extreme coordination cascades.** The expected maximum cascade size grows as ⟨x_max⟩ ∝ N^γ, with γ̂ ≈ 0.85 for TCE, closely matching the EVT prediction γ_th = 1/(α̂ − 1) ≈ 0.82. Larger agent societies produce qualitatively larger cascades, not merely more of them.

**Integration bottleneck.** Expansion-driving primitives (delegation, contradiction) scale strongly with N while consolidation (merge) does not, progressively widening the gap between generative and integrative coordination. This asymmetry is the structural explanation for non-monotonic scaling in LLM MAS.

**DTI intervention.** Deficit-Triggered Integration monitors the expansion–integration imbalance per active cascade and triggers merge integration when the deficit exceeds a condition-specific threshold. DTI reduces excess tail mass, moderates elite concentration, and improves task success in high-imbalance regimes, while preserving the heavy-tailed coordination structure that enables large-scale reasoning.

---

## Experimental Setup

| Dimension | Values |
|---|---|
| Agent scales (N) | 8, 16, 32, 64, 128, 256, 512 |
| Communication topologies | chain, star, tree, fully connected, sparse mesh, hybrid modular, dynamic reputation |
| Benchmarks | GAIA (QA / reasoning), SWE-bench (coding), REALM-Bench (planning), MultiAgentBench (coordination) |
| Seeds | 5 per configuration |
| Total coordination events | ~1.5M |

All agents share a common LLM, prompt, tool set, and task instances. Coordination event types are classified post-hoc from raw interaction traces — no event labels are injected into agent prompts. Workloads are scaled with N using a benchmark-conditioned task expansion module (Appendix H) that generates dependency-structured task trees without prescribing any coordination pattern.

---

## Repository Structure

```
src/
├── context_builder.py              # Agent prompt assembly
├── prompts/                        # Base prompt, topology addenda, task addenda
├── loggers/                        # Event bus, trace schema, run metadata
├── event_extraction/               # Post-hoc coordination event classification
├── observables/                    # Claim DAG construction, cascade metrics
├── metrics/                        # Inequality statistics (Gini, top-k share)
├── tail_fitting/                   # MLE power-law fitting and model comparison
├── analysis/                       # Per-run analysis pipeline
├── tools/                          # Agent tool implementations
├── topologies/                     # Seven communication topology implementations
├── execution/                      # Run orchestration and task tree integration
└── benchmark_wrappers/             # Benchmark loaders and task expansion module
```

---

## How It Works

**1. Task expansion.** Before each run, `TaskExpander` expands a benchmark task into K×M interdependent subtasks with a sparse dependency DAG (paper Appendix H), keeping coordination demand balanced across agent scales.

**2. Agent execution.** Each topology runs a LangGraph graph. Every agent turn produces a `TraceRow` written to `events.jsonl`. No event type labels are injected — agents operate purely on task content and prior outputs visible through their topology.

**3. Post-hoc extraction.** `event_extractor` classifies each row into one of: `propose_claim`, `revise_claim`, `contradict_claim`, `merge_claims`, `endorse_claim`, `delegate_subtask`. `dag_builder` then reconstructs the subtask tree and claim DAG, assigning cascade membership, root claim IDs, and depths to every row.

**4. Power-law observables.** Five sample distributions are extracted per run — delegation cascade size, revision wave size, contradiction burst size, merge fan-in, and TCE — and fitted with truncated power law, log-normal, and exponential models by MLE with Vuong likelihood-ratio tests for model comparison.

---

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env          # add OPENAI_API_KEY
export $(grep -v '^#' .env | xargs)
```

**Running the sweep:**

```bash
# Subset
python scripts/run_sweep.py --benchmarks gaia --topologies chain star --scales 8 16 --seeds 0 1

# Full sweep
python scripts/run_sweep.py --workers 4

# Resume after interruption
python scripts/run_sweep.py --workers 4 --resume
```

Run outputs are written to `data/sweep/{benchmark}/{topology}/n{N}/s{seed}/{task_id}/` and include raw event traces (`events.jsonl`), the task tree (`task_tree.json`), and computed metrics (`run_metadata.json`).

---

## Citation

```bibtex
@article{venkatesh2026powerlaws,
  title   = {Do Agent Societies Develop Intellectual Elites? The Hidden Power Laws of Collective Cognition in LLM Multi-Agent Systems},
  author  = {Venkatesh, Kavana and Cui, Jiaming},
  journal = {arXiv preprint arXiv:2604.02674},
  year    = {2026}
}
```

---

## License

MIT
