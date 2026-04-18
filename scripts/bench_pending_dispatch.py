"""Micro-benchmark for the ``pending_dispatch`` orchestrator stage.

Stage 18 in ``docs/messengers_router_refactor_plan.md`` is gated on
``stage_timings["pending_dispatch"]`` exceeding 20ms at p50 (quoting the
plan: *"worth it ONLY if Stage 14 measurements show the pending dispatch is
>20ms at p50. Otherwise skip."*). This script measures the no-pending
fast-path, which is by far the most common case in production traffic.

Usage:
    PYTHONPATH=. venv/bin/python scripts/bench_pending_dispatch.py [N]

Defaults to 1000 iterations. Prints p50, p90, p99 and max in milliseconds.
"""

from __future__ import annotations

import asyncio
import statistics
import sys
from time import perf_counter

from messengers_router.memory import MemoryStore
from messengers_router.mess_types import SessionState
from messengers_router.orchestrator import OrchestratorContext, pending_dispatch
from messengers_router.services import Services


async def _one_call(services: Services, memory: MemoryStore) -> float:
    state = SessionState(session_id="bench-pending")
    ctx = OrchestratorContext(text="привет", state=state)
    t0 = perf_counter()
    await pending_dispatch(ctx, services=services, memory=memory)
    return (perf_counter() - t0) * 1000.0  # ms


async def _run(n: int) -> list[float]:
    services = Services()
    memory = MemoryStore()
    # warm-up to populate any lazy imports / caches
    for _ in range(10):
        await _one_call(services, memory)
    samples: list[float] = []
    for _ in range(n):
        samples.append(await _one_call(services, memory))
    return samples


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    samples = asyncio.run(_run(n))
    samples.sort()
    p50 = statistics.median(samples)
    p90 = samples[int(0.90 * len(samples))]
    p99 = samples[int(0.99 * len(samples))]
    print(f"pending_dispatch no-pending fast-path over {n} iterations:")
    print(f"  p50 = {p50:.3f} ms")
    print(f"  p90 = {p90:.3f} ms")
    print(f"  p99 = {p99:.3f} ms")
    print(f"  max = {max(samples):.3f} ms")
    print(f"  min = {min(samples):.3f} ms")
    print(f"  mean = {statistics.mean(samples):.3f} ms")
    stage18_threshold = 20.0
    verdict = "IMPLEMENT Stage 18" if p50 > stage18_threshold else "SKIP Stage 18"
    print(f"\nPlan gate (>{stage18_threshold}ms at p50): {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
