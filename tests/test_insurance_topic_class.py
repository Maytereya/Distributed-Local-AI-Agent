"""Вопрос об обслуживании по ДМС / ОМС — тема, которую решает LLM (класс К26, BUG-2026-10-28-K26-DMS-OMS-TOPIC).

Было (выгрузки диалогов 08.10 и 26.10, 25 реплик): «ДМС» правила и сущности брали за услугу или тему подбора — «Дмс» → «не
понял», «Можно ли сдать анализы по полису ДМС?» → «напишите названия анализов», «Как записаться на процедуру по ДМС» → «к
какому врачу?». Решение владельца 28.10 (ответ клиники): как получить помощь по ДМС — ответ операторов; ОМС, конкретный врач —
оператор, оффером «перевести?» (ответ владельца 28.10 на вопросы К26, п. 2); цену по ДМС / ОМС не называть (путь цены, С3).

Инвариант класса под `insurance_topic = on`: реплику со словом страхования (кандидат — слова из данных клиники) относит к
варианту LLM; «как получить помощь по ДМС» — памятка из данных клиники и оффер оператора вопросом; ОМС, конкретный врач,
гарантийное письмо и отдел ДМС — «ответит оператор» текстом из данных, тоже оффером, не переводом сразу; «да» на любой
из офферов — перевод следующим ходом; цена, «ни одна» и сбой LLM — ответ
как без выключателя, и лимит решений хода не тратится. Так же и в идущей записи: «по дмс» на вопрос о филиале — не слот
услуги («Госпитализация в терапевтический стационар»). Реплика без слова страхования LLM темы не зовёт; медвопрос тему не
отдаёт. off — как было, без вызова LLM.
Через обработчик эндпоинта; LLM — подделка настоящего типа.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from evalbench import world as ev_world
from messengers_router import choice, insurance_topic, llm_runtime
from tests._llm_helpers import _CLASSIFIER_MARK
from tests.test_service_answer_class import _LLM, _REGIONS, _classifier, answer_world  # noqa: F401 — фикстура

_MARK = "спрашивает ли пациент об обслуживании по полису ДМС или ОМС"
_DATA = yaml.safe_load((Path(insurance_topic.__file__).resolve().parent / "data" / "insurance_topic.yaml").read_text(encoding="utf-8"))
_MEMO = _DATA["dms_memo"]
_OFFER = _DATA["dms_offer"]

# Как получить помощь по ДМС — синтетика по образцу реплик выгрузок.
DMS_GENERAL = [
    "Дмс",
    "Записаться по дмс",
    "Можно ли сдать анализы по полису ДМС?",
    "здравствуйте, ваша клиника работает по ДМС?",
    "Как записаться на процедуру по ДМС",
    "что взять с собой на приём по дмс?",
]
# Номер варианта LLM → ключ текста «ответит оператор» в данных клиники.
OPERATOR = [
    ("Принимает ли доктор Петрова по ДМС?", 2, "dms_doctor"),
    ("Куда прислать гарантийное письмо от страховой?", 3, "dms_case"),
    ("подскажите, отдел дмс рассмотрел мой протокол?", 3, "dms_case"),
    ("По омс", 4, "oms"),
    ("Или бесплатно можно?", 4, "oms"),
]
AS_BEFORE = ["Сколько стоит приём терапевта по ДМС?", "Записаться по дмс", "По омс"]
NOT_CANDIDATES = ["сколько стоит ферритин", "хочу записаться к кардиологу", "спасибо большое"]


def _pick(*picks: int) -> str:
    return json.dumps({"picks": list(picks)})


def _turns(world_dir, monkeypatch, replies: list[str], *, switch: str, answers: dict[str, Any], cfg_extra: dict | None = None) -> list[dict[str, Any]]:
    """Ходы одной сессии через эндпоинт: ответ, перевод, оффер вопросом, телеметрия и места вызовов LLM каждого хода."""

    from messengers_router.endpoint import MessengerGenerateRequest, messenger_generate_once
    from messengers_router.memory import MemoryStore
    from messengers_router.runtime_config import config as cfg
    from messengers_router.services import Services
    from messengers_router.services import _prepare_select_llm

    fake = _LLM(answers)
    memory = MemoryStore()
    session = f"ins-{switch}-{replies[0]}"
    out: list[dict[str, Any]] = []

    async def one(reply: str) -> dict[str, Any]:
        req = MessengerGenerateRequest(session_id=session, text=reply, debug=True)
        body = json.loads((await messenger_generate_once(req, memory=memory, services=Services())).body)
        return {"text": str(body.get("text") or ""), "handoff": bool(body.get("handoff"))}

    with ev_world.FrozenWorld(world_dir, "day") as world, monkeypatch.context() as patch:
        patch.setattr(cfg, "MR_INSURANCE_TOPIC", switch, raising=False)
        for key, value in (cfg_extra or {}).items():
            patch.setattr(cfg, key, value, raising=False)
        patch.setattr(llm_runtime, "_BREAKER", llm_runtime._Breaker())
        patch.setattr(llm_runtime, "_OLLAMA_CLIENT", fake)
        patch.setitem(_prepare_select_llm._kb_cache, "memos", ())

        async def regions(_self):
            return list(_REGIONS)

        patch.setattr(Services, "_ensure_regions_loaded", regions)
        for reply in replies:
            choice._CACHE.clear()
            before = len(fake.calls)
            row = asyncio.run(one(reply))
            row["telemetry"] = world.take_telemetry() or {}
            row["calls"] = fake.calls[before:]
            row["pending_offer"] = bool(memory.get(session).last_entities.get("_operator_offer_pending"))
            out.append(row)
    return out


def _turn(world_dir, monkeypatch, reply: str, **kwargs) -> dict[str, Any]:
    return _turns(world_dir, monkeypatch, [reply], **kwargs)[0]


def _answers(insurance: Any, classifier_label: str = "OTHER") -> dict[str, Any]:
    return {_CLASSIFIER_MARK: _classifier(classifier_label), _MARK: insurance}


@pytest.mark.parametrize("reply", DMS_GENERAL + [r for r, _, _ in OPERATOR])
def test_off_is_as_before_without_llm(reply, answer_world, monkeypatch):  # noqa: F811
    out = _turn(answer_world, monkeypatch, reply, switch="off", answers=_answers(_pick(1)))

    assert _MARK not in out["calls"], out["calls"]
    assert "insurance_topic" not in out["telemetry"], out["telemetry"]


@pytest.mark.parametrize("reply", DMS_GENERAL)
def test_how_to_get_care_by_dms_is_the_clinic_memo_with_operator_offer(reply, answer_world, monkeypatch):  # noqa: F811
    out = _turn(answer_world, monkeypatch, reply, switch="on", answers=_answers(_pick(1)))

    assert out["telemetry"].get("insurance_topic") == "dms", out["telemetry"]
    assert out["text"] == f"{_MEMO}\n\n{_OFFER}", out["text"]
    assert not out["handoff"] and out["pending_offer"], out


@pytest.mark.parametrize(("reply", "number"), [("Записаться по дмс", 1)] + [(r, n) for r, n, _ in OPERATOR])
def test_yes_to_the_offer_hands_off(reply, number, answer_world, monkeypatch):  # noqa: F811
    offer, yes = _turns(answer_world, monkeypatch, [reply, "да"], switch="on", answers=_answers(_pick(number)))

    assert offer["pending_offer"] and not offer["handoff"], offer
    assert yes["handoff"], yes
    assert _MARK not in yes["calls"], yes["calls"]  # «да» на оффер — ответ на вопрос бота, не новая тема


@pytest.mark.parametrize(("reply", "number", "key"), OPERATOR)
def test_oms_named_doctor_and_dms_paperwork_offer_the_operator_with_clinic_text(reply, number, key, answer_world, monkeypatch):  # noqa: F811
    out = _turn(answer_world, monkeypatch, reply, switch="on", answers=_answers(_pick(number)))

    assert out["telemetry"].get("insurance_topic") == key, out["telemetry"]
    assert out["text"] == _DATA["operator_offer"][key], out["text"]
    assert not out["handoff"] and out["pending_offer"], out  # оффер вопросом, не перевод сразу (ответ владельца 28.10, п. 2)


@pytest.mark.parametrize("reply", AS_BEFORE)
@pytest.mark.parametrize("insurance", [_pick(5), _pick(), ConnectionError("тест: LLM недоступна"), "не json"])
def test_price_none_and_failure_answer_as_off(reply, insurance, answer_world, monkeypatch):  # noqa: F811
    off = _turn(answer_world, monkeypatch, reply, switch="off", answers=_answers(_pick(1)))
    on = _turn(answer_world, monkeypatch, reply, switch="on", answers=_answers(insurance))

    assert _MARK in on["calls"], on["calls"]
    assert "insurance_topic" not in on["text"]
    assert (on["text"], on["handoff"]) == (off["text"], off["handoff"]), (reply, off["text"], on["text"])


def test_turn_decision_is_refunded_when_the_topic_does_not_answer(answer_world, monkeypatch):  # noqa: F811
    out = _turn(answer_world, monkeypatch, "Сколько стоит приём терапевта по ДМС?", switch="on", answers=_answers(_pick(5)))

    assert out["telemetry"].get("insurance_topic") == "price", out["telemetry"]
    assert out["telemetry"].get("insurance_topic_refund") == 1, out["telemetry"]


@pytest.mark.parametrize("reply", NOT_CANDIDATES)
def test_reply_without_insurance_word_does_not_ask_llm(reply, answer_world, monkeypatch):  # noqa: F811
    out = _turn(answer_world, monkeypatch, reply, switch="on", answers=_answers(_pick(1)))

    assert _MARK not in out["calls"], out["calls"]
    assert "insurance_topic" not in out["telemetry"], out["telemetry"]


_BOOKING = "хочу записаться к терапевту"


def test_dms_inside_booking_is_the_memo_not_a_service_slot(answer_world, monkeypatch):  # noqa: F811
    booking, dms = _turns(answer_world, monkeypatch, [_BOOKING, "по дмс"], switch="on", answers=_answers(_pick(1), "APPOINTMENT"))

    assert "Какой филиал" in booking["text"], booking["text"]
    assert dms["text"] == f"{_MEMO}\n\n{_OFFER}" and dms["pending_offer"], dms


@pytest.mark.parametrize("switch", ["off", "on"])
def test_booking_reply_without_insurance_word_stays_in_booking(switch, answer_world, monkeypatch):  # noqa: F811
    _, branch = _turns(answer_world, monkeypatch, [_BOOKING, "пр. Ленина, 5"], switch=switch, answers=_answers(_pick(1), "APPOINTMENT"))

    assert _MARK not in branch["calls"], branch["calls"]
    assert _MEMO not in branch["text"], branch["text"]


def test_medical_advice_keeps_its_template(answer_world, monkeypatch):  # noqa: F811
    out = _turn(
        answer_world, monkeypatch, "второй день болит живот, к кому можно пойти по полису?", switch="on",
        answers=_answers(_pick(1), classifier_label="MEDICAL_ADVICE"),
    )

    assert _MARK not in out["calls"], out["calls"]
    assert _MEMO not in out["text"], out["text"]


def test_memo_text_is_clinic_data_not_code(answer_world, monkeypatch):  # noqa: F811
    edited = {**_DATA, "dms_memo": "По ДМС: текст клиники для проверки."}
    monkeypatch.setattr(insurance_topic, "_data", lambda: edited)
    out = _turn(answer_world, monkeypatch, "Дмс", switch="on", answers=_answers(_pick(1)))

    assert out["text"] == f"По ДМС: текст клиники для проверки.\n\n{_OFFER}", out["text"]
    source = Path(insurance_topic.__file__).read_text(encoding="utf-8")
    assert "учёной степени" not in source and "15 минут" not in source  # факты — в данных клиники, не в коде


@pytest.mark.parametrize("path", ["messengers_router/prompts/choose_insurance.txt", "app_data/prompts/mr_choose_insurance.txt"])
def test_prompt_is_the_same_in_both_locations(path):
    root = Path(__file__).resolve().parents[1]
    assert (root / path).read_text(encoding="utf-8") == (root / "messengers_router/prompts/choose_insurance.txt").read_text(encoding="utf-8")
