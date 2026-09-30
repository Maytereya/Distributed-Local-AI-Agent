"""Город филиала из дерева ``parent`` справочника ``/regions`` — одна функция на весь код.

Прод-бэкенд МИС (medserver-egisz) НЕ отдаёт поле ``city``: город закодирован деревом
``parent`` (``Все → <…область> → <Город> → <филиалы>``), а у филиала адрес без города
(«ул.Победы, 126»). Кто судит о Самаре по ``city`` или по «самара» в названии, теряет
почти все самарские филиалы.

Живёт здесь, а не в роутере, потому что нужен обоим слоям: роутеру
(``messengers_router.services.core._ensure_regions_loaded``, BUG-A, ``88727d8``) и
сборщику цен по врачам (``api_price.update_doctor_prices``). У сборщика была своя
копия старого признака — и с переезда на medserver-egisz он каждый день писал пустой
``doctor_prices``: «подходящих врачей не нашёл» в каждом ответе о цене приёма
(BUG-2026-09-25-NO-DOCTORS-FOUND-FALSE).
"""

from __future__ import annotations

import re
from typing import Any


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").lower().replace("ё", "е").strip())


def derive_region_city(region: dict[str, Any], byid: dict[Any, dict[str, Any]]) -> str:
    """Город филиала из иерархии ``parent``.

    Поднимаемся по ``parent`` до ноды-города (её родитель — «…область»/«…край»/
    «…республика» или корень «Все») и возвращаем её ``name``. Так филиал-спутник
    (напр. «Пирогова, 4» под «Новокуйбышевск») получает СВОЙ город, а не «Самара».

    :param region: регион/филиал из ``/regions``
    :param byid: индекс ``id -> регион`` по всему списку (для резолва parent)
    :return: имя города или собственное имя региона (фолбэк)
    """
    node = region
    seen: set[Any] = set()
    for _ in range(8):  # дерево мелкое; гард от циклов/битых ссылок
        pid = node.get("parent")
        if pid is None or pid in seen:
            break
        seen.add(pid)
        parent = byid.get(pid)
        if not isinstance(parent, dict):
            break
        pname = _norm(parent.get("name"))
        if "област" in pname or "край" in pname or "республик" in pname or pname == "все":
            # `node` — нода-города (её родитель — регион/страна) → его имя = город.
            return str(node.get("name") or "").strip()
        node = parent
    return str(region.get("name") or "").strip()


def inject_region_cities(regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Проставляет каждому региону производный ``city`` из иерархии ``parent``.

    Идемпотентна; уже заданный ``city`` (если бэкенд его дал) сохраняется.

    :param regions: сырой список регионов от ``api_nayka.site_regions``
    :return: тот же список с проставленным ``city`` у каждого региона
    """
    if not isinstance(regions, list):
        return regions
    byid: dict[Any, dict[str, Any]] = {r.get("id"): r for r in regions if isinstance(r, dict)}
    for r in regions:
        if isinstance(r, dict) and not str(r.get("city") or "").strip():
            r["city"] = derive_region_city(r, byid)
    return regions
