"""Half-open reporting windows and minute coverage, independent of transport."""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone


def period_window(period: str, now: datetime, start: datetime | None = None):
    if now.tzinfo is None:
        raise ValueError("timezone_required")
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    if period == "day":
        bounds = now - timedelta(days=1), now
    elif period == "today":
        bounds = midnight, now
    elif period == "week":
        end = midnight - timedelta(days=midnight.weekday())
        bounds = end - timedelta(days=7), end
    elif period == "month":
        end = midnight.replace(day=1)
        bounds = (end - timedelta(days=1)).replace(day=1), end
    elif period == "all" and start is not None:
        bounds = start, now
    else:
        pieces = period.split("..")
        try:
            first = datetime.strptime(pieces[0], "%Y-%m-%d").replace(tzinfo=now.tzinfo)
            last = datetime.strptime(pieces[-1], "%Y-%m-%d").replace(tzinfo=now.tzinfo)
        except ValueError as exc:
            raise ValueError("invalid_period") from exc
        if len(pieces) > 2 or last < first:
            raise ValueError("invalid_period")
        bounds = first, last + timedelta(days=1)
    return tuple(value.astimezone(timezone.utc) for value in bounds)


def covered_samples(samples, start: datetime, end: datetime, interval: int):
    """One valid observation per slot; gaps break an observed outage series."""
    seconds = max(0, (end - start).total_seconds())
    expected = math.ceil(seconds / interval)
    slots = {}
    rejected = 0
    for sample in sorted(samples, key=lambda item: item["_dt"]):
        dt = sample["_dt"]
        if not start <= dt < end:
            continue
        channels = sample.get("channels") or []
        if not channels or any(type(item.get("ok")) is not bool for item in channels):
            rejected += 1
            continue
        slot = int((dt - start).total_seconds() // interval)
        copy = dict(sample)
        ok = sum(item["ok"] for item in channels)
        copy["state"] = "ok" if ok == len(channels) else "partial" if ok else "down"
        slots[slot] = copy
    longest = streak = 0
    previous = None
    for slot, sample in sorted(slots.items()):
        if sample["state"] == "down":
            streak = streak + 1 if previous is not None and slot == previous + 1 else 1
            longest = max(longest, streak)
        else:
            streak = 0
        previous = slot
    observed = sum(min(interval, seconds - slot * interval) for slot in slots)
    return {
        "samples": [item for _, item in sorted(slots.items())],
        "expected_samples": expected,
        "missing_samples": max(0, expected - len(slots)),
        "invalid_samples": rejected,
        "observed_seconds": observed,
        "coverage_percent": round(observed * 100 / seconds, 2) if seconds else None,
        "max_down_streak_samples": longest,
    }
