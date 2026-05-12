"""
omnirca/evaluation/query_builder.py — Builds blind evaluation queries (Phase 7).

The fairness contract (plan.md):
  • The agent receives ONLY: time window, ONE affected/visible service name,
    and a natural-language symptom description.
  • The agent does NOT receive: root_cause_service, fault_type, faults.csv,
    labels.csv, or any dataset-specific metadata.
  • Symptom descriptions are written as a real SRE alert would read —
    they describe observable effects without naming the fault category.

Query format (as specified by user):
    "Incident window: {t_start} → {t_end}
     Reported affected service: {affected_service}
     Symptom: {natural_language_description}
     Investigate and identify the root cause."

For each fault, ONE affected service is chosen:
  - Prefer a visible VICTIM (not root cause) when multiple services are affected.
  - When only one service is affected, that service (which is also the root cause)
    is reported — the agent still must confirm via data and classify the fault.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Service ID → name mapping (from plan.md, confirmed against architecture.pkl)
# ─────────────────────────────────────────────────────────────────────────────

_ID_TO_NAME: dict[int, str] = {
    0:  "frontend_0",
    1:  "frontend_1",
    2:  "api_gateway",
    3:  "backend_0",
    4:  "backend_1",
    5:  "backend_2",
    6:  "backend_3",
    7:  "backend_4",
    8:  "backend_5",
    9:  "backend_6",
    10: "backend_7",
    11: "database_0",
    12: "database_1",
    13: "database_2",
    14: "cache_0",
    15: "cache_1",
    16: "message_queue_0",
    17: "message_queue_1",
    18: "auth_service",
    19: "logging",
}


# ─────────────────────────────────────────────────────────────────────────────
# Per-fault query definitions
# Each entry specifies:
#   reported_service_id : int — the ONE visible affected service in the query
#                          Prefer victim over root cause when possible.
#   symptom             : str — natural-language SRE alert text
#                          MUST NOT mention fault category or root cause service.
# ─────────────────────────────────────────────────────────────────────────────

_FAULT_QUERIES: dict[int, dict] = {
    # ── Fault 0: stale_cache | root=cache_0 (id=14) | affected=[14,18,19]
    # Report: auth_service (18) — victim, not root
    0: {
        "reported_service_id": 18,  # auth_service (victim)
        "symptom": (
            "auth_service is reporting elevated error counts, yet individual "
            "requests are completing unusually quickly. Multiple downstream teams "
            "are seeing data inconsistencies — responses are being served but contain "
            "outdated values that fail downstream validation. Correct tokens are "
            "occasionally being rejected as invalid despite successful authentication."
        ),
    },

    # ── Fault 1: transaction_deadlock | root=database_2 (id=13) | affected=[13,4,6]
    # Report: backend_1 (id=4) — victim, not root
    1: {
        "reported_service_id": 4,  # backend_1 (victim)
        "symptom": (
            "backend_1 is timing out on write operations. Database transactions "
            "are hanging indefinitely before eventually failing. The service has "
            "entered a degraded state where writes queue up without completing. "
            "Services dependent on this data path have stopped processing requests "
            "successfully."
        ),
    },

    # ── Fault 2: api_version_mismatch | root=backend_6 (id=9) | affected=[9,12,13,18]
    # Report: database_1 (id=12) — victim, not root
    2: {
        "reported_service_id": 12,  # database_1 (victim)
        "symptom": (
            "database_1 is rejecting incoming connections at a high rate. Connection "
            "establishment attempts are failing immediately rather than timing out, "
            "which suggests a protocol-level rejection rather than resource exhaustion. "
            "Services that rely on this data store are receiving connection errors and "
            "cannot proceed with their operations."
        ),
    },

    # ── Fault 3: authentication_failure | root=auth_service (id=18) | affected=[18,19]
    # Report: auth_service (18) — only real non-logging victim = root cause here
    3: {
        "reported_service_id": 18,  # auth_service (root = only real victim)
        "symptom": (
            "auth_service is under severe stress. System-wide authentication is "
            "failing — all dependent services are unable to authenticate their "
            "internal requests. Credential lookup operations are spiking without "
            "a corresponding increase in successful authentications. Users cannot "
            "log in and inter-service calls are being rejected."
        ),
    },

    # ── Fault 4: data_corruption | root=backend_7 (id=10) | affected=[10]
    # Report: backend_7 (10) — only affected service = root cause
    4: {
        "reported_service_id": 10,  # backend_7 (only service, also root)
        "symptom": (
            "backend_7 is producing invalid outputs. Write operations are failing "
            "with data integrity errors. The service is generating outputs that "
            "downstream consumers are rejecting during validation. The failure "
            "pattern suggests corruption is occurring at the write stage, not "
            "during transmission."
        ),
    },

    # ── Fault 5: disk_io_saturation | root=database_2 (id=13) | affected=[13,18,19]
    # Report: auth_service (18) — victim, not root
    5: {
        "reported_service_id": 18,  # auth_service (victim)
        "symptom": (
            "auth_service latency has increased dramatically. All I/O-bound "
            "operations across the system are significantly slower than baseline. "
            "The degradation is steady rather than spiky, suggesting a resource "
            "saturation problem at the storage tier rather than a transient spike. "
            "Database query times are increasing monotonically."
        ),
    },

    # ── Fault 6: memory_leak | root=backend_4 (id=7) | affected=[7]
    # Report: backend_4 (7) — only affected service = root cause
    6: {
        "reported_service_id": 7,  # backend_4 (only service, also root)
        "symptom": (
            "backend_4 memory usage has been climbing steadily for the past "
            "15 minutes with no sign of levelling off. Response latency is "
            "gradually increasing in proportion to the memory growth. The service "
            "has not crashed but performance continues to degrade linearly over time."
        ),
    },

    # ── Fault 7: data_race_condition | root=backend_0 (id=3) | affected=[3]
    # Report: backend_0 (3) — only affected service = root cause
    7: {
        "reported_service_id": 3,  # backend_0 (only service, also root)
        "symptom": (
            "backend_0 is producing inconsistent results under concurrent load. "
            "Simultaneous requests are interfering with each other, causing sporadic "
            "failures when multiple operations access shared internal state at the "
            "same time. The failures are non-deterministic and difficult to reproduce "
            "in isolation."
        ),
    },

    # ── Fault 8: thread_pool_exhaustion | root=backend_2 (id=5) | affected=[5]
    # Report: backend_2 (5) — only affected service = root cause
    8: {
        "reported_service_id": 5,  # backend_2 (only service, also root)
        "symptom": (
            "backend_2 is refusing new incoming connections. The service appears "
            "to have exhausted its worker thread capacity — incoming requests are "
            "queuing with no response. The service process is alive and healthy "
            "but cannot accept new work. Existing in-flight requests are "
            "completing normally."
        ),
    },

    # ── Fault 9: cascading_timeout | root=backend_6 (id=9) | affected=[9]
    # Report: backend_6 (9) — only affected service = root cause
    9: {
        "reported_service_id": 9,  # backend_6 (only service, also root)
        "symptom": (
            "backend_6 response times have increased approximately 10x from "
            "baseline. All services that depend on backend_6 are now timing out "
            "waiting for responses. The slowness is cascading through the entire "
            "dependency chain, causing system-wide degradation. The root slowdown "
            "appears to originate at this single service."
        ),
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# FaultQuery dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FaultQuery:
    """A single evaluation query for one fault."""
    fault_id:          int
    t_start:           str
    t_end:             str
    reported_service:  str          # ONE visible affected service (may = root)
    symptom:           str          # natural-language description
    query_text:        str          # the full text passed to the agent
    # Ground truth (loaded by harness, NEVER passed to agent)
    fault_type:                  str   # raw CSV type label (e.g. "stale_cache")
    ground_truth_service:        str
    ground_truth_category:       str   # normalised SOP key
    ground_truth_category_group: str   # functional | non-functional


# ─────────────────────────────────────────────────────────────────────────────
# QueryBuilder
# ─────────────────────────────────────────────────────────────────────────────

# Normalize the CSV category names to match SOP library keys
_CATEGORY_NORM: dict[str, str] = {
    "authentication_failure": "auth_failure",
    "stale_cache":            "stale_cache",
    "transaction_deadlock":   "transaction_deadlock",
    "api_version_mismatch":   "api_version_mismatch",
    "data_corruption":        "data_corruption",
    "disk_io_saturation":     "disk_io_saturation",
    "memory_leak":            "memory_leak",
    "data_race_condition":    "data_race_condition",
    "thread_pool_exhaustion": "thread_pool_exhaustion",
    "cascading_timeout":      "cascading_timeout",
}

# Broad category group per fault type (from plan.md)
_CATEGORY_GROUP: dict[str, str] = {
    "stale_cache":            "functional",
    "transaction_deadlock":   "functional",
    "api_version_mismatch":   "functional",
    "auth_failure":           "functional",
    "authentication_failure": "functional",
    "data_corruption":        "functional",
    "data_race_condition":    "functional",
    "disk_io_saturation":     "non-functional",
    "memory_leak":            "non-functional",
    "thread_pool_exhaustion": "non-functional",
    "cascading_timeout":      "non-functional",
}


class QueryBuilder:
    """
    Builds evaluation queries from faults.csv metadata.

    The QueryBuilder reads ground truth ONLY for query construction.
    The harness controls what the agent sees — the query_text never
    contains root_cause_service or fault_type explicitly.
    """

    def __init__(self, faults_csv_path: str) -> None:
        import pandas as pd
        self._df = pd.read_csv(faults_csv_path)

    def build(self, fault_id: int) -> FaultQuery:
        """Build a FaultQuery for the given fault_id."""
        row = self._df[self._df["fault_id"] == fault_id].iloc[0]

        t_start = str(row["start_time"])
        t_end   = str(row["end_time"])
        root_id = int(row["root_cause_service"])
        root_service = _ID_TO_NAME[root_id]
        raw_category = str(row["type"])
        category = _CATEGORY_NORM.get(raw_category, raw_category)
        category_group = _CATEGORY_GROUP.get(raw_category, "unknown")

        spec = _FAULT_QUERIES[fault_id]
        reported_service = _ID_TO_NAME[spec["reported_service_id"]]
        symptom = spec["symptom"]

        query_text = (
            f"Incident window: {t_start} → {t_end}\n"
            f"Reported affected service: {reported_service}\n\n"
            f"Symptom description:\n{symptom}\n\n"
            f"Investigate and identify the root cause service and fault type."
        )

        return FaultQuery(
            fault_id=fault_id,
            t_start=t_start,
            t_end=t_end,
            reported_service=reported_service,
            symptom=symptom,
            query_text=query_text,
            fault_type=raw_category,
            ground_truth_service=root_service,
            ground_truth_category=category,
            ground_truth_category_group=category_group,
        )

    def build_all(self) -> list[FaultQuery]:
        """Build queries for all 10 faults."""
        return [self.build(i) for i in range(10)]

    @property
    def num_faults(self) -> int:
        return len(self._df)
