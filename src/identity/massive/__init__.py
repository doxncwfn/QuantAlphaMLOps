"""
Massive Point-in-Time Identity Backfill Subsystem.
==================================================
Modular components for HTTP querying, per-key pacing, atomic caching,
resumable checkpointing, telemetry reconciliation, 3-level date strategy,
and multi-threaded worker pool orchestration.
"""

from src.identity.massive.backfill import BackfillEngine
from src.identity.massive.cache import CacheManager
from src.identity.massive.checkpoint import CheckpointManager
from src.identity.massive.client import MassiveClient
from src.identity.massive.date_strategy import RepresentativeDateStrategy
from src.identity.massive.rate_limiter import PerKeyRateLimiter
from src.identity.massive.telemetry import WorkerTelemetry
from src.identity.massive.worker import MassiveWorker
from src.identity.massive.worker_pool import ConcurrentKeyWorkerPool

__all__ = [
    "BackfillEngine",
    "CacheManager",
    "CheckpointManager",
    "ConcurrentKeyWorkerPool",
    "MassiveClient",
    "MassiveWorker",
    "PerKeyRateLimiter",
    "RepresentativeDateStrategy",
    "WorkerTelemetry",
]
