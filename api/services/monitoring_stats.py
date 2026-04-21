"""In-memory monitoring statistics for the KlarKI audit pipeline.

Singleton pattern — import `stats` to record events from any module.
All counters reset when the API process restarts (no persistence).
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field
from threading import Lock
from typing import Any


@dataclass
class _NodeStats:
    invocations: int = 0
    total_duration_s: float = 0.0
    errors: int = 0

    @property
    def avg_duration_s(self) -> float:
        return self.total_duration_s / self.invocations if self.invocations else 0.0


class MonitoringStats:
    """Thread-safe in-memory counters for pipeline observability."""

    def __init__(self) -> None:
        self._lock = Lock()
        self.started_at: float = time.time()

        # Audit pipeline counters
        self.total_audits: int = 0
        self.successful_audits: int = 0
        self.failed_audits: int = 0
        self.total_pipeline_duration_s: float = 0.0

        # Per-stage duration lists (for avg/p95 calculation)
        self._stage_durations: dict[str, list[float]] = defaultdict(list)

        # LangGraph node stats keyed by node name
        self._graph_nodes: dict[str, _NodeStats] = {}

        # Active audits in progress (audit_id → start_time)
        self._active: dict[str, float] = {}

    # ── Pipeline recording ────────────────────────────────────────────────────

    def audit_started(self, audit_id: str) -> None:
        with self._lock:
            self.total_audits += 1
            self._active[audit_id] = time.time()

    def audit_completed(self, audit_id: str) -> None:
        with self._lock:
            start = self._active.pop(audit_id, time.time())
            self.successful_audits += 1
            self.total_pipeline_duration_s += time.time() - start

    def audit_failed(self, audit_id: str) -> None:
        with self._lock:
            start = self._active.pop(audit_id, time.time())
            self.failed_audits += 1
            self.total_pipeline_duration_s += time.time() - start

    def record_stage(self, stage: str, duration_s: float) -> None:
        with self._lock:
            self._stage_durations[stage].append(duration_s)

    # ── LangGraph node recording ──────────────────────────────────────────────

    def record_graph_node(self, node: str, duration_s: float, error: bool = False) -> None:
        with self._lock:
            if node not in self._graph_nodes:
                self._graph_nodes[node] = _NodeStats()
            n = self._graph_nodes[node]
            n.invocations += 1
            n.total_duration_s += duration_s
            if error:
                n.errors += 1

    # ── Snapshot for API response ─────────────────────────────────────────────

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            uptime_s = time.time() - self.started_at
            total = self.total_audits or 1  # avoid division by zero

            stage_stats = {}
            for stage, durations in self._stage_durations.items():
                if durations:
                    sorted_d = sorted(durations)
                    p95_idx = max(0, int(len(sorted_d) * 0.95) - 1)
                    stage_stats[stage] = {
                        "count": len(durations),
                        "avg_s": round(sum(durations) / len(durations), 2),
                        "p95_s": round(sorted_d[p95_idx], 2),
                        "min_s": round(sorted_d[0], 2),
                        "max_s": round(sorted_d[-1], 2),
                    }

            graph_node_stats = {}
            for node, ns in self._graph_nodes.items():
                graph_node_stats[node] = {
                    "invocations": ns.invocations,
                    "avg_duration_s": round(ns.avg_duration_s, 2),
                    "total_duration_s": round(ns.total_duration_s, 2),
                    "errors": ns.errors,
                }

            avg_pipeline = (
                round(self.total_pipeline_duration_s / (self.successful_audits + self.failed_audits), 2)
                if (self.successful_audits + self.failed_audits)
                else 0.0
            )

            return {
                "uptime_s": round(uptime_s, 1),
                "pipeline": {
                    "total": self.total_audits,
                    "successful": self.successful_audits,
                    "failed": self.failed_audits,
                    "active": len(self._active),
                    "success_rate": round(self.successful_audits / total * 100, 1),
                    "avg_duration_s": avg_pipeline,
                },
                "stages": stage_stats,
                "graph_nodes": graph_node_stats,
            }


# Global singleton — import this everywhere
stats = MonitoringStats()
