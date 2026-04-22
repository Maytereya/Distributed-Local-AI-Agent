import asyncio

from agent_logic_2 import llama_func_call as lf
from schedule_ttl_cache import AsyncListTTLStaleCache


def run(coro):
    return asyncio.run(coro)


def _build_cache(clock: dict[str, float], *, fresh: int = 30, stale: int = 600, negative: int = 15) -> AsyncListTTLStaleCache:
    return AsyncListTTLStaleCache(
        fresh_ttl_seconds=fresh,
        stale_ttl_seconds=stale,
        negative_ttl_seconds=negative,
        max_keys=100,
        logger=lf.logger,
        log_events=False,
        name="schedule_cache",
        time_func=lambda: clock["ts"],
    )


def test_llama_schedule_cache_hit(monkeypatch):
    clock = {"ts": 1000.0}
    cache = _build_cache(clock, fresh=30, stale=600, negative=15)
    calls = {"count": 0}

    def fake_schedule(_surname: str):
        calls["count"] += 1
        return [{"fio": "Дразнин Дмитрий", "schedule": {"г. Самара, пр. Ленина, 5": []}}]

    monkeypatch.setattr(lf, "_SCHEDULE_CACHE", cache)
    monkeypatch.setattr(lf, "find_doctor_schedule", fake_schedule)

    first = run(lf.find_doctor_schedule_async("Дразнин"))
    second = run(lf.find_doctor_schedule_async("Дразнин"))

    assert calls["count"] == 1
    assert isinstance(first, list) and first
    assert isinstance(second, list) and second


def test_llama_schedule_cache_miss_after_fresh_ttl(monkeypatch):
    clock = {"ts": 2000.0}
    cache = _build_cache(clock, fresh=10, stale=600, negative=5)
    calls = {"count": 0}

    def fake_schedule(_surname: str):
        calls["count"] += 1
        return [{"fio": "Дразнин Дмитрий", "schedule": {"г. Самара, пр. Ленина, 5": []}}]

    monkeypatch.setattr(lf, "_SCHEDULE_CACHE", cache)
    monkeypatch.setattr(lf, "find_doctor_schedule", fake_schedule)

    run(lf.find_doctor_schedule_async("Дразнин"))
    assert calls["count"] == 1

    clock["ts"] += 11
    run(lf.find_doctor_schedule_async("Дразнин"))
    assert calls["count"] == 2


def test_llama_schedule_cache_returns_stale_on_error(monkeypatch):
    clock = {"ts": 3000.0}
    cache = _build_cache(clock, fresh=10, stale=120, negative=5)
    calls = {"count": 0}
    fail = {"enabled": False}

    def fake_schedule(_surname: str):
        calls["count"] += 1
        if fail["enabled"]:
            raise RuntimeError("source unavailable")
        return [{"fio": "Дразнин Дмитрий", "schedule": {"г. Самара, пр. Ленина, 5": []}}]

    monkeypatch.setattr(lf, "_SCHEDULE_CACHE", cache)
    monkeypatch.setattr(lf, "find_doctor_schedule", fake_schedule)

    first = run(lf.find_doctor_schedule_async("Дразнин"))
    assert isinstance(first, list) and first
    assert calls["count"] == 1

    fail["enabled"] = True
    clock["ts"] += 11
    second = run(lf.find_doctor_schedule_async("Дразнин"))

    assert calls["count"] == 3
    assert isinstance(second, list) and second


def test_llama_schedule_negative_cache_ttl(monkeypatch):
    clock = {"ts": 4000.0}
    cache = _build_cache(clock, fresh=30, stale=120, negative=5)
    calls = {"count": 0}

    def fake_schedule(_surname: str):
        calls["count"] += 1
        return []

    monkeypatch.setattr(lf, "_SCHEDULE_CACHE", cache)
    monkeypatch.setattr(lf, "find_doctor_schedule", fake_schedule)

    first = run(lf.find_doctor_schedule_async("Дразнин"))
    assert first == []
    assert calls["count"] == 1

    clock["ts"] += 2
    second = run(lf.find_doctor_schedule_async("Дразнин"))
    assert second == []
    assert calls["count"] == 1

    clock["ts"] += 4
    third = run(lf.find_doctor_schedule_async("Дразнин"))
    assert third == []
    assert calls["count"] == 2
