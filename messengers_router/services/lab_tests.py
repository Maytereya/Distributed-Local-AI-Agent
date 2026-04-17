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

if TYPE_CHECKING:
    from ..services_legacy import Services


def _legacy_module():
    """Лениво импортирует legacy-модуль, чтобы не создать цикл импортов.

    :return: модуль ``messengers_router.services_legacy``
    """

    from .. import services_legacy as legacy

    return legacy


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

    legacy = _legacy_module()
    _ = query
    surname = legacy._get_first_present(entities, ["surname", "result_surname"])
    filial = legacy._get_first_present(entities, ["filial", "result_filial"])
    year_raw = entities.get("year")
    number_raw = entities.get("number")

    if number_raw is None:
        number_raw = entities.get("order_id")

    year = legacy._as_int(year_raw)
    number = legacy._as_int(number_raw)
    return {
        "surname": str(surname or "").strip(),
        "year": year,
        "filial": str(filial or "").strip(),
        "number": number,
        "lang": legacy._get_first_present(entities, ["lang", "result_lang"]) or "ru",
    }


def _build_public_result_link(fields: dict[str, Any]) -> str | None:
    """Строит публичную ссылку на результат анализа.

    :param fields: нормализованные поля результата
    :return: готовая ссылка или ``None``, если данных недостаточно
    """

    legacy = _legacy_module()
    surname = str(fields.get("surname") or "").strip()
    filial = str(fields.get("filial") or "").strip()
    year = legacy._as_int(fields.get("year"))
    number = legacy._as_int(fields.get("number"))
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

    legacy = _legacy_module()
    test_name = legacy._get_first_present(entities, ["test_name", "service_name"]) or query
    needle = legacy._normalise_input(test_name)
    if not needle:
        return _test_assist_clarify_response(entities, note="test_assist: no test query")

    try:
        price_rows = await asyncio.to_thread(api_price.load_price_by_region, legacy.SAMARA_PRICE_REGION_ID)
    except Exception:
        return _test_assist_clarify_response(entities, note="test_assist source unavailable")
    matches = legacy._rank_price_rows([p for p in price_rows if isinstance(p, dict)], test_name, limit=10)

    if not matches:
        return _test_assist_clarify_response(
            entities,
            note=f"test_assist: no matches ({legacy.SAMARA_PRICE_REGION_ID})",
        )

    return {
        "tests": matches,
        "promos": [],
        "note": f"test_assist: priceByRegion({legacy.SAMARA_PRICE_REGION_ID})",
        "entities_used": entities,
    }


async def test_result_status(self: "Services", query: str, entities: dict[str, Any]) -> dict[str, Any]:
    """Проверяет готовность результата анализа и формирует ссылку на него.

    :param self: экземпляр сервисного слоя
    :param query: текст запроса пользователя
    :param entities: извлечённые NLU-сущности
    :return: payload с готовностью результата или fallback-ответом
    """

    legacy = _legacy_module()
    _ = self

    def _result_fallback(
        note: str,
        message: str = legacy.handoff_message("service_error_results"),
    ) -> dict[str, Any]:
        """Формирует единый fallback при ошибке получения результатов.

        :param note: диагностическая пометка
        :param message: пользовательское сообщение handoff
        :return: fallback-payload с ``ready=False``
        """

        return legacy._service_fallback(
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
        return _result_fallback(f"resultForPatient error: {api_resp}")

    payload = api_resp.get("data")
    if not payload:
        return {
            "ready": False,
            "note": "result_not_found_or_not_ready",
            "result_payload": payload,
            "result_preview": "По указанным данным результаты пока не найдены или еще не готовы.",
            "entities_used": entities,
        }

    link = _build_public_result_link(fields)
    if not link:
        return _result_fallback(
            "result_link_build_failed",
            legacy.handoff_message("service_error_result_link"),
        )

    return {
        "ready": True,
        "note": "result_link_constructed",
        "result_payload": payload,
        "result_preview": "Ссылка на результат сформирована.",
        "result_links": [link],
        "entities_used": entities,
    }
