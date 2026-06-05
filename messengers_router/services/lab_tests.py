"""Модуль домена лабораторных услуг.

Содержит pilot-миграцию логики подбора анализов и выдачи результатов
из ``services_legacy``. Публичный API для внешних вызовов по-прежнему
идёт через ``Services``.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any
from urllib.parse import quote_from_bytes

from agent_logic_2.nayka_api import api_nayka, api_price

from ..policies import handoff_message
from ._common import (
    _as_int,
    _get_first_present,
    _normalise_input,
    _service_fallback,
)
from ._prices_helpers import (
    _PRICE_SERVICE_ALIASES,
    SAMARA_PRICE_REGION_ID,
    _rank_price_rows,
    _resolve_multi_price_items,
)

if TYPE_CHECKING:
    from .core import Services


def _cp1251_urlencode(value: str) -> str:
    """Кодирует параметр под контракт ссылки Nayka Lab.

    :param value: исходное строковое значение
    :return: URL-safe строка в кодировке Windows-1251
    """

    raw = str(value or "").strip().encode("cp1251", errors="replace")
    return quote_from_bytes(raw, safe="")


def _extract_result_query_fields(entities: dict[str, Any], query: str) -> dict[str, Any]:
    """Собирает поля, необходимые для поиска результата анализа.

    :param entities: извлечённые сущности роутера
    :param query: исходный текст запроса пользователя
    :return: словарь с фамилией, годом, филиалом, номером и языком
    """

    _ = query
    surname = _get_first_present(entities, ["surname", "result_surname"])
    filial = _get_first_present(entities, ["filial", "result_filial"])
    year_raw = entities.get("year")
    number_raw = entities.get("number")

    if number_raw is None:
        number_raw = entities.get("order_id")

    year = _as_int(year_raw)
    number = _as_int(number_raw)
    return {
        "surname": str(surname or "").strip(),
        "year": year,
        "filial": str(filial or "").strip(),
        "number": number,
        "lang": _get_first_present(entities, ["lang", "result_lang"]) or "ru",
    }


def _build_public_result_link(fields: dict[str, Any]) -> str | None:
    """Строит публичную ссылку на результат анализа.

    :param fields: нормализованные поля результата
    :return: готовая ссылка или ``None``, если данных недостаточно
    """

    surname = str(fields.get("surname") or "").strip()
    filial = str(fields.get("filial") or "").strip()
    year = _as_int(fields.get("year"))
    number = _as_int(fields.get("number"))
    if not surname or not filial or year is None or number is None:
        return None
    return (
        "https://naykalab.ru/getanaliz.php"
        f"?fam={_cp1251_urlencode(surname)}"
        f"&year={year}"
        f"&nom={_cp1251_urlencode(filial)}"
        f"&nom2={number}"
        "&fast=1"
    )


def _test_assist_clarify_response(entities: dict[str, Any], *, note: str) -> dict[str, Any]:
    """Возвращает безопасный fallback для подбора анализов.

    :param entities: текущие сущности роутера
    :param note: диагностическая пометка источника
    :return: payload TEST_ASSIST без handoff_required
    """

    return {
        "tests": [],
        "promos": [],
        "message": (
            "Уточните, пожалуйста, какие симптомы, жалобы или цель обследования вас интересуют, "
            "и я помогу подобрать анализы."
        ),
        "note": note,
        "entities_used": entities,
    }


async def test_assist(self: "Services", query: str, entities: dict[str, Any]) -> dict[str, Any]:
    """Подбирает анализы по запросу пользователя через прайс лаборатории.

    :param self: экземпляр сервисного слоя
    :param query: текст запроса пользователя
    :param entities: извлечённые NLU-сущности
    :return: payload со списком анализов или уточняющим сообщением
    """

    test_name = _get_first_present(entities, ["test_name", "service_name"]) or query
    needle = _normalise_input(test_name)
    if not needle:
        return _test_assist_clarify_response(entities, note="test_assist: no test query")

    try:
        price_rows = await asyncio.to_thread(api_price.load_price_by_region, SAMARA_PRICE_REGION_ID)
    except Exception:
        return _test_assist_clarify_response(entities, note="test_assist source unavailable")
    rows = [p for p in price_rows if isinstance(p, dict)]

    # Мульти-услуговый запрос (через `,` / `;` / ` и ` / `+` / `/`): резолвим
    # каждую услугу через каталог с алиасами (ОАК→общий анализ крови и т.п.),
    # отдаём LLM реальные цены ВСЕХ найденных позиций. LLM не сможет
    # «дофантазировать» недостающие (защищено правилом промпта). Однопредметные
    # запросы и пробельные перечисления идут стандартным single-bag путём ниже.
    multi_items = _resolve_multi_price_items(str(query or ""), rows)
    if multi_items:
        merged: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in multi_items:
            for row in item.get("prices") or []:
                if not isinstance(row, dict):
                    continue
                code = _normalise_input(str(row.get("serviceHomecode") or row.get("homecode") or ""))
                name_key = _normalise_input(str(row.get("serviceName") or row.get("name") or ""))
                key = code or name_key
                if not key or key in seen:
                    continue
                seen.add(key)
                merged.append(row)
        if merged:
            return {
                "tests": merged,
                "promos": [],
                "note": f"test_assist: priceByRegion({SAMARA_PRICE_REGION_ID}, multi-item)",
                "entities_used": entities,
            }

    matches = _rank_price_rows(rows, test_name, limit=10)

    # Алиас одиночного запроса (BUG-2026-06-02-09): _rank_price_rows ранжирует по
    # литералу, поэтому аббревиатура («ОАК») не находит позиции, названные полной
    # формой («Общий анализ крови (...)») — бот показывал только урезанные
    # «ОАК (...)» / капиллярные варианты. Подмешиваем семейство по ТОЧНОЙ фразе
    # алиас-цели (оак→«общий анализ крови») и показываем его ПЕРВЫМ. Multi-word
    # фраза алиаса → без шума (3-буквенное «оак» цепляло бы «трОАКарная» и т.п.).
    alias_targets = _PRICE_SERVICE_ALIASES.get(needle, ())
    if alias_targets:

        def _row_key(row: dict[str, Any]) -> str:
            return _normalise_input(
                str(row.get("serviceHomecode") or row.get("serviceName") or row.get("name") or "")
            )

        seen = {_row_key(r) for r in matches}
        family: list[dict[str, Any]] = []
        for tgt in alias_targets:
            tgt_norm = _normalise_input(tgt)
            if len(tgt_norm) < 6:
                continue
            for row in _rank_price_rows(rows, tgt, limit=10):
                name_norm = _normalise_input(str(row.get("serviceName") or row.get("name") or ""))
                if tgt_norm not in name_norm:
                    continue
                k = _row_key(row)
                if k and k not in seen:
                    seen.add(k)
                    family.append(row)
        if family:
            matches = (family + matches)[:12]

    if not matches:
        return _test_assist_clarify_response(
            entities,
            note=f"test_assist: no matches ({SAMARA_PRICE_REGION_ID})",
        )

    return {
        "tests": matches,
        "promos": [],
        "note": f"test_assist: priceByRegion({SAMARA_PRICE_REGION_ID})",
        "entities_used": entities,
    }


# Бот не видит готового результата (HTTP 404 от resultForPatient ИЛИ пустой data).
# ВАЖНО: 404 НЕ означает однозначно «не готов» — это может быть «не найден / неверные
# данные / нет в этом medserver». BUG-2026-06-04-05: прод-бот ходит в ТЕСТОВЫЙ medserver
# (172.16.0.246/medserver-test), где реальных пациентов нет → 404 даже на готовые
# результаты. Поэтому НЕ утверждаем «Результат пока не готов» как факт (это
# дезинформация) — честно перечисляем возможные причины, просим проверить данные и
# предлагаем оператора. Авто-эскалации на оператора при 404 по-прежнему нет.
_RESULT_NOT_READY_TEXT = (
    "Не нашёл готовый результат по этим данным. Возможно, он ещё не готов, "
    "либо неточно указаны фамилия, год рождения, код или номер анализа — "
    "проверьте, пожалуйста. Если результат уже должен быть готов, напишите «оператор»."
)


async def test_result_status(self: "Services", query: str, entities: dict[str, Any]) -> dict[str, Any]:
    """Проверяет готовность результата анализа и формирует ссылку на него.

    :param self: экземпляр сервисного слоя
    :param query: текст запроса пользователя
    :param entities: извлечённые NLU-сущности
    :return: payload с готовностью результата или fallback-ответом
    """

    _ = self

    def _result_fallback(
        note: str,
        message: str = handoff_message("service_error_results"),
    ) -> dict[str, Any]:
        """Формирует единый fallback при ошибке получения результатов.

        :param note: диагностическая пометка
        :param message: пользовательское сообщение handoff
        :return: fallback-payload с ``ready=False``
        """

        return _service_fallback(
            note=note,
            handoff_message=message,
            entities=entities,
            reason="test_result_fallback",
            extra={"ready": False},
        )

    fields = _extract_result_query_fields(entities, query)
    missing = [key for key in ("surname", "year", "filial", "number") if not fields.get(key)]
    if missing:
        return {
            "ready": False,
            "note": "missing_result_fields",
            "missing_fields": missing,
            "entities_used": entities,
        }

    try:
        api_resp = await asyncio.to_thread(
            api_nayka.site_result_for_patient,
            surname=fields["surname"],
            year=int(fields["year"]),
            filial=fields["filial"],
            number=int(fields["number"]),
            lang=fields["lang"],
            with_time=None,
        )
    except Exception as exc:
        return _result_fallback(f"resultForPatient failed: {exc}")

    if not isinstance(api_resp, dict) or not api_resp.get("ok"):
        # HTTP 404 = по этим данным готового результата нет (ещё в работе / не найден).
        # Это бизнес-состояние, а не сбой сервера → честный «в работе», БЕЗ оператора.
        # Реальные сбои (5xx, таймаут, нет соединения → status_code 5xx/None) уходят в
        # operator-fallback ниже.
        if isinstance(api_resp, dict) and api_resp.get("status_code") == 404:
            return {
                "ready": False,
                "note": "result_not_ready",
                "result_preview": _RESULT_NOT_READY_TEXT,
                "entities_used": entities,
            }
        return _result_fallback(f"resultForPatient error: {api_resp}")

    payload = api_resp.get("data")
    if not payload:
        return {
            "ready": False,
            "note": "result_not_found_or_not_ready",
            "result_payload": payload,
            "result_preview": _RESULT_NOT_READY_TEXT,
            "entities_used": entities,
        }

    link = _build_public_result_link(fields)
    if not link:
        return _result_fallback(
            "result_link_build_failed",
            handoff_message("service_error_result_link"),
        )

    return {
        "ready": True,
        "note": "result_link_constructed",
        "result_payload": payload,
        "result_preview": "Ссылка на результат сформирована.",
        "result_links": [link],
        "entities_used": entities,
    }
