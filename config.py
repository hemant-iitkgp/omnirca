"""
Central configuration for OmniRCA.
All paths, thresholds, and empirically calibrated constants live here.
"""
import os
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT      = Path(__file__).parent            # omnirca/
# OMNIRCA_DATA_DIR lets a run point at a different dataset without editing code.
DATA_DIR          = Path(os.getenv("OMNIRCA_DATA_DIR",
                                   str(PROJECT_ROOT.parent / "datasets_complex")))
ARCH_PKL          = DATA_DIR / "architecture.pkl"
FAULT_ENCYCLOPEDIA = DATA_DIR / "FAULT_ENCYCLOPEDIA.md"

METRICS_CSV  = DATA_DIR / "metrics.csv"
LOGS_CSV     = DATA_DIR / "logs.csv"
TRACES_CSV   = DATA_DIR / "traces.csv"
SYSCALLS_CSV = DATA_DIR / "syscalls.csv"

# ── LLM (Phase 3+) ───────────────────────────────────────────────────────────
LLM_PROVIDER      = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL         = os.getenv("LLM_MODEL", "gpt-4o")
TEMPERATURE_VOTES = [0.3, 0.5, 0.7]   # for N=3 self-consistency (Phase 5)
MAX_STEPS         = 15

# ── Anomaly detection ────────────────────────────────────────────────────────
ANOMALY_THRESHOLD     = float(os.getenv("OMNIRCA_ANOMALY_THRESHOLD", 2.5))
                               # z-score threshold — "this service is anomalous"
BASELINE_MINUTES      = int(os.getenv("OMNIRCA_BASELINE_MINUTES", 60))
                               # baseline window length (minutes before fault start)
BASELINE_SKIP_MINUTES = int(os.getenv("OMNIRCA_BASELINE_SKIP", 5))
                               # skip the N minutes immediately preceding the fault
                               # to avoid baseline contamination by early fault symptoms

# ── Primary detection panel ──────────────────────────────────────────────────
# Columns in syscalls.csv that DataDetective ranks services on.  A service's
# score is the STRONGEST (max) z-score across this panel, so the ranking works
# whatever the dataset's dominant channel happens to be.
# datasets_complex: syscall duration is the only real signal -> just the two.
# RCAEval: resource channels carry the fault, latency alone misses CPU faults.
PRIMARY_CHANNELS = [c for c in os.getenv(
    "OMNIRCA_PRIMARY_CHANNELS", "avg_duration_us,p99_duration_us").split(",") if c]

# ── Graph fusion (Phase 2) ───────────────────────────────────────────────────
STATIC_GRAPH_WEIGHT  = 1.0
DYNAMIC_GRAPH_WEIGHT = 2.0
TEMPORAL_BONUS       = 0.20   # +20% propagation score for earlier-onset service

# ── Signal weights for composite anomaly score ───────────────────────────────
# Empirically calibrated from assess.py:
#   - syscall columns: z = 30–70 on all 10 faults  → highest weight
#   - metric columns:  z = 0.00–0.07 on all faults  → minimal weight
#   - log errors:      0 entries in all fault windows → near-zero weight
#   - trace latency:   max z = 0.64, only 10/20 services → low weight
SIGNAL_WEIGHTS = {
    "syscall_avg_duration_us": 4.0,   # z-score; confirmed z=30–70 all faults
    "syscall_p99_duration_us": 4.0,   # z-score; confirmed z=48–70 all faults
    "syscall_error_rate":      2.0,   # RATIO not z (std≈0 makes z unstable); 15× real
    "memory_slope":            1.5,   # linear regression in MB/s; only metric signal
    "cpu_percent":             0.5,   # z-score; near-zero but audited
    "latency_ms":              0.5,   # z-score; near-zero backup
    "trace_latency":           0.5,   # z-score; max 0.64, only 10 of 20 services
    "log_error_density":       0.2,   # count delta per minute; 0 in all fault windows
}

# Severity thresholds for composite score output labels
SEVERITY_HIGH   = 10.0
SEVERITY_MEDIUM = 3.0

# ── Service topology (confirmed from architecture.pkl inspection) ─────────────
# auth_service (id=18) is called by ALL 18 non-logging services
# logging (id=19) is called by ALL 19 services — never a root cause
# cache_1 (id=15) has ZERO incoming edges — isolated, never a root cause
NEVER_ROOT_CAUSE = {"logging"}   # agent should never conclude logging is root cause
