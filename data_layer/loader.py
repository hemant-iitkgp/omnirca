"""
DataLoader singleton.

Loads all four CSVs once at startup, strips the forbidden 'is_anomaly' column,
and builds the service name ↔ id bidirectional map used by all tools.
"""
import pandas as pd
from ..config import METRICS_CSV, LOGS_CSV, TRACES_CSV, SYSCALLS_CSV

# Columns that must never be exposed to the agent
_FORBIDDEN_COLS = {"is_anomaly"}


class DataLoader:
    """Singleton. Call get_loader() to get the shared instance."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    # ── Public load ──────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load all CSVs into memory. Idempotent — safe to call multiple times."""
        if self._loaded:
            return

        print("[loader] Loading datasets_complex CSVs into memory …")
        self.metrics  = pd.read_csv(METRICS_CSV,  parse_dates=["timestamp"])
        self.logs     = pd.read_csv(LOGS_CSV,     parse_dates=["timestamp"])
        self.traces   = pd.read_csv(TRACES_CSV,   parse_dates=["start_time"])
        self.syscalls = pd.read_csv(SYSCALLS_CSV, parse_dates=["timestamp"])

        # Strip forbidden columns from every DataFrame
        for attr in ("metrics", "logs", "traces", "syscalls"):
            df = getattr(self, attr)
            to_drop = [c for c in df.columns if c in _FORBIDDEN_COLS]
            if to_drop:
                setattr(self, attr, df.drop(columns=to_drop))
                print(f"[loader]  Stripped {to_drop} from {attr}.csv")

        # Build service id ↔ name maps (metrics.csv is the authoritative source)
        svc = self.metrics[["service_id", "service_name"]].drop_duplicates()
        self.id_to_name: dict[int, str] = \
            svc.set_index("service_id")["service_name"].to_dict()
        self.name_to_id: dict[str, int] = \
            {v: k for k, v in self.id_to_name.items()}
        self.service_names: list[str] = sorted(self.name_to_id.keys())

        self._loaded = True
        print(f"[loader] Ready. {len(self.service_names)} services, "
              f"{len(self.metrics):,} metric rows, "
              f"{len(self.logs):,} log rows, "
              f"{len(self.traces):,} trace spans, "
              f"{len(self.syscalls):,} syscall rows.")

    # ── Service resolution helpers ───────────────────────────────────────────

    def resolve_service(self, service: "int | str") -> tuple[int, str]:
        """Accept either a service name or id; return (id, name) pair."""
        if isinstance(service, int):
            if service not in self.id_to_name:
                raise ValueError(f"Unknown service id {service}. "
                                 f"Valid ids: {sorted(self.id_to_name)}")
            return service, self.id_to_name[service]
        if service not in self.name_to_id:
            raise ValueError(f"Unknown service {service!r}. "
                             f"Valid names: {self.service_names}")
        return self.name_to_id[service], service


# ── Module-level singleton accessor ─────────────────────────────────────────

_loader = DataLoader()


def get_loader() -> DataLoader:
    """Return the singleton DataLoader, loading CSVs on first call."""
    _loader.load()
    return _loader
