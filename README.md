# OmniRCA

**LLM-powered multi-agent root cause analysis for microservice incidents.**

> *"Can an LLM agent, given standard SRE diagnostic procedures and raw telemetry — but no labels or ground truth — correctly identify the root cause of a microservice incident?"*

**Result: 95.7% top-1 accuracy (22/23 runs) across 10 distinct fault types on a 20-service microservice topology.**

---

## What It Does

OmniRCA takes a plain-language incident report (time window + observable symptom) and autonomously diagnoses the root cause by reasoning over raw telemetry: syscalls, metrics, distributed traces, and application logs. It uses six specialist LLM agents in a ReAct (Reason + Act) loop, each building on the previous agent's findings.

**Input** — only this, nothing more:
```
"Multiple services are timing out and requests are failing.
 This started around 1pm on Jan 1 and lasted about an hour."
```

**Output:**
```json
{
  "root_cause_service": "backend_6",
  "fault_category": "cascading_timeout",
  "confidence": "HIGH",
  "evidence": [
    "backend_6 syscall p99 latency z-score = 64.2 (baseline: 120μs, fault: 8100μs)",
    "backend_6 anomaly onset 3min before all downstream victims (temporal)",
    "4 downstream services show cascading timeout errors after backend_6 degraded (traces)",
    "No anomalous upstream neighbors — structural root cause signal (graph)"
  ],
  "propagation_path": "backend_6 → database_1 → backend_0 → frontend_1"
}
```

The system is fully blind: it never sees ground-truth labels, fault IDs, or any pre-computed anomaly annotations. Every diagnosis is derived purely from raw telemetry and general SRE knowledge.

---

## Architecture

```
Incident Query (time window + symptom)
         │
         ▼
┌─────────────────────────────────────────────────────┐
│                    MainAgent                        │
│              (sequential orchestrator)              │
└─────────────────────────────────────────────────────┘
         │
         ├─► DataDetective      → rank all services by syscall z-score
         │
         ├─► GraphExplorer      → use topology to separate cause from victim
         │
         ├─► FaultTyper         → classify fault type via RAG over FAULT_ENCYCLOPEDIA
         │
         ├─► EvidenceCollector  → corroborate with traces, metrics, logs
         │
         ├─► TemporalAnalyst    → determine which service's anomaly started first
         │
         └─► JudgeAgent         → synthesize findings → final verdict + confidence
```

Agents share a `ctx` dict that accumulates findings end-to-end. Each agent reads prior findings before reasoning, so later agents have full context.

---

## Agents

### 1. DataDetective
Scans all services using syscall z-scores against a 1-hour pre-fault baseline. Returns a ranked list of anomalous services and a breakdown of which syscall sub-channels are elevated (read/write/mmap/fsync/etc.).

### 2. GraphExplorer
Builds a fused graph (static architecture + trace-active edges during fault window). Walks upstream from top-ranked services — a service with no anomalous upstream neighbor is the root cause candidate. Resolves the victim/cause confusion that pure statistical ranking misses.

### 3. FaultTyper
Classifies the fault type by querying a FAISS-indexed fault encyclopedia (general SRE fault archetypes, no dataset-specific names). Retrieves a multi-step Standard Operating Procedure (SOP) for the identified fault.

### 4. EvidenceCollector
Independently verifies the top candidate against each telemetry modality. Requires syscall evidence plus at least one of: trace latency anomaly, memory slope, or error burst. Prevents false positives from single-signal noise.

### 5. TemporalAnalyst
Computes anomaly onset time per service using change-point detection. Returns `STRONG_EVIDENCE` (≥1 min lead), `WEAK_EVIDENCE` (<1 min), or `AMBIGUOUS` — serves as a tiebreaker when graph topology is ambiguous.

### 6. JudgeAgent
Evaluates consistency across all five preceding agents. Applies conflict resolution rules, assigns final confidence (HIGH/MEDIUM/LOW), and generates a human-readable evidence summary.

---

## Tools (24 total)

| Domain | Tools |
|---|---|
| **Syscall** | `syscall_multi_service_compare`, `syscall_sub_channel_analysis`, `compute_anomaly_score`, `detect_memory_slope` |
| **Graph** | `get_service_dependencies`, `build_call_path`, `get_dynamic_graph`, `get_propagation_candidates`, `get_fused_graph_summary`, `detect_causal_order` |
| **Metrics** | `query_metrics`, `cross_correlate_services` |
| **Traces** | `query_traces`, `compare_trace_latency`, `detect_error_burst` |
| **Logs** | `query_logs` |
| **RCA** | `search_fault_knowledge`, `classify_fault_pattern`, `check_sop`, `explain_evidence`, `finalize_rca` |

---

## Supported Fault Types

| Category | Faults |
|---|---|
| `data_integrity` | stale_cache, api_version_mismatch, data_corruption, data_race_condition |
| `resource_exhaustion` | memory_leak, thread_pool_exhaustion, disk_io_saturation |
| `cascading_deadlock` | cascading_timeout, transaction_deadlock |
| `auth` | authentication_failure |

---

## Evaluation Results

| Metric | Value |
|---|---|
| **Top-1 Root Cause Accuracy** | 22 / 23 = **95.7%** |
| **Top-3 Coverage** | 23 / 23 = **100%** |
| **Fault Category Accuracy** | 7 / 23 = 30.4% |
| **Mean Latency** | ~48.4 seconds |
| **HIGH-confidence calibration** | 18 / 19 = **94.7%** |

The single failure (memory_leak, run P1) was caused by GraphExplorer's topology rule over-weighting database services. Fix: make the database-priority rule fault-category-aware and suppress it for `memory_leak`.

---

## Project Structure

```
omnirca/                       ← repo root (this folder)
├── agent/
│   ├── prompts.py             # LLM system prompts for each agent
│   ├── react_agent.py         # ReAct loop implementation
│   └── tool_schema.py         # OpenAI function-calling schemas + dispatcher
├── agents/
│   ├── main_agent.py          # Orchestrator — runs 6 agents sequentially
│   ├── base_agent.py          # SubAgent base class (ReAct loop)
│   ├── data_detective.py
│   ├── graph_explorer.py
│   ├── fault_typer.py
│   ├── evidence_collector.py
│   ├── temporal_analyst.py
│   └── judge_agent.py
├── tools/
│   ├── syscall_tools.py       # Primary signal tools
│   ├── graph_tools.py
│   ├── metric_tools.py
│   ├── trace_tools.py
│   ├── log_tools.py
│   ├── rca_tools.py           # RAG search, SOP lookup, evidence explanation
│   └── base.py
├── data_layer/
│   ├── loader.py              # CSV loader (strips is_anomaly to prevent leakage)
│   ├── graph_loader.py        # Architecture graph + fused graph
│   ├── temporal.py            # Anomaly onset detection
│   └── kv_store.py
├── rag/
│   ├── retriever.py           # FAISS semantic search
│   ├── indexer.py
│   └── index/                 # Pre-built FAISS index
├── sops/
│   ├── sop_library.py         # 10 canonical SOPs (one per fault category)
│   └── sop_generator.py
├── voting/
│   └── self_consistency.py    # 3-trajectory majority voting
├── evaluation/
│   ├── test_harness.py        # Blind evaluation runner
│   ├── metrics.py
│   ├── query_builder.py
│   ├── ablation.py
│   └── stress_analysis.py
├── tests/                     # Phase-by-phase unit tests
├── config.py                  # All paths, thresholds, signal weights
├── llm_client.py              # LLM provider abstraction (OpenAI / Azure)
└── logger.py
```

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/hemant-iitkgp/omnirca.git
cd omnirca
pip install -e .
```

`pip install -e .` installs the `omnirca` package in editable mode so all `from omnirca.xxx import ...` statements resolve correctly from anywhere.

### 2. Set your LLM credentials

```bash
cp .env.example .env
```

Edit `.env` with your Azure OpenAI credentials (it is gitignored and never committed):

```ini
AZURE_OPENAI_KEY=your-api-key
AZURE_OPENAI_ENDPOINT=https://your-resource.cognitiveservices.azure.com/
AZURE_DEPLOYMENT=gpt-4o-2
```

`llm_client.py` loads this file automatically via `python-dotenv`.

### 3. Prepare your dataset

Place your telemetry data **one level above the repo** (i.e., `../datasets_complex/`):

```
datasets_complex/       ← sibling of the cloned omnirca/ repo
├── metrics.csv
├── syscalls.csv
├── logs.csv
├── traces.csv
├── faults.csv
├── architecture.pkl         # NetworkX DiGraph
└── FAULT_ENCYCLOPEDIA.md    # General fault archetypes
```

`config.py` resolves `DATA_DIR = Path(__file__).parent.parent / "datasets_complex"` — no changes needed if you follow this layout.

### 4. Run

```bash
# Single custom query (edit run.py to set your window + symptom)
python run.py

# Single blind evaluation run (requires faults.csv)
python -m evaluation.test_harness

# Full 23-fault stress battery
python -m evaluation._run_stress_battery
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| **Syscalls as primary signal** | Only service-local and fault-type-specific; not transmitted downstream to victims like metrics and traces are |
| **Sequential agents, not parallel** | Each agent depends on the previous; parallel execution degrades reasoning quality |
| **Temperature = 0.0** | Deterministic output for reproducible evaluation |
| **Ground truth stripped at load time** | `is_anomaly` column removed from all CSVs — prevents any label leakage |
| **Multi-modal corroboration required** | Syscalls + ≥1 other modality before setting HIGH confidence |
| **GraphExplorer overrides DataDetective** | A high-z-score service is often a *victim* receiving load from the actual root cause — topology-aware reasoning corrects this in ~80% of multi-service faults |

---

## Anomaly Detection

Z-score per service per metric channel against a rolling pre-fault baseline:

```
z = (x_fault − μ_baseline) / σ_baseline
```

- Baseline: 60 minutes before fault start
- Skip last 5 minutes of baseline to avoid early-symptom contamination
- Composite score: weighted sum across all channels (weights in `config.py`)

Primary signal weights (empirically calibrated):
```python
"syscall_avg_duration_us": 4.0   # z = 30–70 on all fault types
"syscall_p99_duration_us": 4.0   # z = 48–70 on all fault types
"syscall_error_rate":      2.0   # ratio-based (std ≈ 0 makes z unstable)
"memory_slope":            1.5   # linear regression in MB/s
```

---

## Citation

```
OmniRCA: LLM-Powered Multi-Agent Root Cause Analysis for Microservice Incidents
Hemant Agarwal, IIT Kharagpur, 2025
```
