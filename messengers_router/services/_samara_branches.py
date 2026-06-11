"""Единый источник «филиалов Самары» с устойчивым fallback.

Зачем (Диалог #232 и весь класс «один Гагарина 64»): список филиалов Самары и то,
какой что делает (анализы/ЭКГ/УЗИ/приём) — СТАТИКА. Но `/regions` тянется только
live (in-memory TTL, на диск НЕ персистится). При частичном/медленном ответе
`/regions` parent-иерархия приходит неполной → производный город не выводится →
все филиалы кроме «Гагарина 64» (у неё «Самара» прямо в name) отваливаются →
бот выдаёт одну Гагарину.

Решение: live `/regions` остаётся источником, но:
  1) при ЗДОРОВОЙ загрузке самарский срез персистится на диск (last-good снапшот);
  2) при частичной/пустой загрузке отдаётся last-good снапшот, иначе seed из репо.
Так бот НИКОГДА не схлопывается на один филиал — worst case это полный статический
список (31 заборная точка), а не одинокая/неверная Гагарина.

Строки имеют ту же форму, что `/regions` (city/addressForSite/analysis/ecg/usi/
doctorService/phone/work_time), поэтому текут через `address_info` без изменений.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Самарских филиалов ~32. Сигнатура деградации /regions — схлопывание до ОДНОГО
# («Гагарина 64», единственная с «Самара» в name, выживает без производного города).
# Порог намеренно консервативный (≤1 → fallback): ловит точный репорт-кейс и не
# мешает тестам, которые мокают 2-3 региона. Last-good снапшот даёт богатый fallback;
# при необходимости порог можно поднять (тогда обновить мок-фикстуры тестов).
SAMARA_MIN_BRANCHES = 2

_SEED_PATH = Path(__file__).resolve().parent.parent / "data" / "nonbookable_points.json"
_SNAPSHOT_FILENAME = "samara_branches_snapshot.json"

# Кэши в памяти, чтобы не читать диск на каждый запрос.
_seed_cache: list[dict[str, Any]] | None = None


def _is_samara_region(r: dict[str, Any]) -> bool:
    """Единый Самара-фильтр (S2: централизован из 3 копий address_info/core).

    NB: подстрочная лазейка «самара in name/addressForSite» пока сохранена для
    обратной совместимости (её снятие — отдельная стадия). Именно она делает
    «Гагарину 64» привилегированной; fallback ниже нейтрализует последствия.
    """
    if not isinstance(r, dict):
        return False
    # Lazy import — избегаем циклов с core/city.
    from .core import _is_samara_city_value  # noqa: PLC0415
    from ..russian_nlu import normalize_ru  # noqa: PLC0415

    if _is_samara_city_value(str(r.get("city") or "")):
        return True
    name = normalize_ru(str(r.get("name") or ""))
    addr = normalize_ru(str(r.get("addressForSite") or ""))
    return "самара" in name or "самара" in addr


def samara_subset(regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Самарский срез из полного /regions (все города)."""
    return [r for r in (regions or []) if _is_samara_region(r)]


def _analysis_count(rows: list[dict[str, Any]] | None) -> int:
    return sum(1 for r in (rows or []) if isinstance(r, dict) and bool(r.get("analysis")))


def is_healthy_samara(rows: list[dict[str, Any]] | None) -> bool:
    """Правдоподобен ли самарский срез: не схлопнут до ≤1 филиала.

    По размеру среза (а не по analysis-флагу): ecg/usi-only фикстуры тоже валидны,
    и точная сигнатура деградации — выживание одной «Гагарины».
    """
    return sum(1 for r in (rows or []) if isinstance(r, dict)) >= SAMARA_MIN_BRANCHES


def _snapshot_path() -> Path | None:
    try:
        from agent_logic_2.nayka_api import cache_paths  # noqa: PLC0415

        return cache_paths.resolve_cache_data_dir() / _SNAPSHOT_FILENAME
    except Exception:
        return None


def persist_snapshot(rows: list[dict[str, Any]]) -> None:
    """Атомарно сохраняет ЗДОРОВЫЙ самарский срез как last-good снапшот."""
    if not is_healthy_samara(rows):
        return
    path = _snapshot_path()
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(list(rows), ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
    except Exception as exc:  # pragma: no cover - диск best-effort
        logger.warning("persist_snapshot failed: %s", exc)


def load_snapshot() -> list[dict[str, Any]]:
    path = _snapshot_path()
    if path is None or not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return [r for r in data if isinstance(r, dict)] if isinstance(data, list) else []
    except Exception:
        return []


def load_seed() -> list[dict[str, Any]]:
    """Seed-список филиалов Самары из committed `data/nonbookable_points.json`.

    Это существующий статический файл клиники (32 филиала, точные адреса/телефоны/
    графики, флаги has_analysis/has_ekg). Маппим его строки в форму /regions, чтобы
    fallback тёк через address_info. usi/doctorService в файле не отслеживаются
    (он про walk-in заборные точки) → False; их добирает last-good снапшот живого
    /regions. Cold-start fallback покрывает главный кейс — walk-in анализы/ЭКГ.
    """
    global _seed_cache
    if _seed_cache is not None:
        return _seed_cache
    rows: list[dict[str, Any]] = []
    try:
        data = json.loads(_SEED_PATH.read_text(encoding="utf-8"))
        samara = data.get("Самара") if isinstance(data, dict) else None
        for r in samara or []:
            if not isinstance(r, dict):
                continue
            addr = str(r.get("address") or "").strip()
            if not addr:
                continue
            rows.append({
                "id": f"seed_{len(rows) + 1}",
                "city": "Самара",
                "name": addr,
                "addressForSite": addr,
                "analysis": bool(r.get("has_analysis", True)),
                "ecg": bool(r.get("has_ekg", False)),
                "usi": False,
                "doctorService": False,
                "phone": str(r.get("phone") or "").strip(),
                "work_time": str(r.get("work_time") or "").strip(),
            })
    except Exception:
        rows = []
    _seed_cache = rows
    return _seed_cache


def resolve_samara_branches(live_regions: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    """Самарские филиалы с fallback: live(здоровый) → last-good снапшот → seed.

    Возвращает (строки, источник). Персист last-good делается в `_ensure_regions_loaded`
    на свежем fetch (естественный TTL-троттлинг), не здесь. Если все источники
    неправдоподобны — отдаёт то, что есть от live (≤1), и пусть гард деградации в
    address_info честно сообщит о проблеме.

    :param live_regions: полный список /regions (все города) из живой загрузки
    :return: (самарские строки-филиалы, метка источника)
    """
    live_samara = samara_subset(live_regions)
    if is_healthy_samara(live_samara):
        return live_samara, "live"
    snap = load_snapshot()
    if is_healthy_samara(snap):
        return snap, "snapshot"
    seed = load_seed()
    if is_healthy_samara(seed):
        return seed, "seed"
    return live_samara, "degraded_partial"


def fallback_branches() -> list[dict[str, Any]]:
    """Статический fallback самарских филиалов: last-good снапшот → seed.

    Используется в address_info ТОЛЬКО для walk-in анализов/ЭКГ при схлопывании
    живого /regions (см. гейт там). Возвращает [] если ни снапшота, ни seed нет.
    """
    snap = load_snapshot()
    if is_healthy_samara(snap):
        return snap
    seed = load_seed()
    return seed if is_healthy_samara(seed) else []
