"""Подбор анализов — только по данным хода (BUG-2026-10-21-WALKIN-LLM-FREE-TEXT; принципы 1–3).

Класс: ход TEST_ASSIST, которому не нашлось готового ответа, отвечал свободным текстом LLM поверх строк прайса,
найденных по буквам реплики, и утверждал о клинике то, чего в данных нет: «подготовка обычно включает специальную
диету» к строке «Прото-скрин» (g0783), «анализ на магний доступен» при пустом поиске (g0190), анализы плода как
«комплекс к материнству» (g0266). Под `topic_by_llm` (ветки `rule_test_result`, `rule_prepare`) таких ходов больше.

Инвариант под `test_assist_from_data = on`: свободного текста на ходе TEST_ASSIST нет; в ответе — только названия и
сроки выбранных LLM строк прайса (печатает код, без цены); нет данных, сбой LLM, мусор — честный оффер оператора.
Под off — как было (рендер-подделка здесь «выдумывает», и на базе инвариант не держится).

Под `test_assist_price = on` (решение владельца 26.10) у выбранных строк — цена: каждая сумма ответа — сумма той строки
прайса, чьё название напечатано рядом (проверка — независимым детектором стенда `evalbench.prices`), выбор LLM тот же.

Через обработчик эндпоинта; LLM — подделка настоящего типа (`GenerateResponse`).
"""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from evalbench import world as ev_world
from messengers_router import choice, llm_runtime, test_assist_answer
from messengers_router.service_answer import NO_DATA_TEXT
from tests._llm_helpers import _CLASSIFIER_MARK
from tests.test_service_answer_class import _LLM, _REGIONS, _classifier, _number_of, verdict

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "mini_prod" / "snapshots"
_DOCTORS = Path(__file__).resolve().parent / "fixtures" / "doctors_excerpt" / "doctors_20261008.jsonl"
_TA_MARK = "об анализах и исследованиях ТОЛЬКО по прайсу клиники"
_RENDER_MARK = "Ты помощник клиники в мессенджере"
# Свойство вопроса об услуге из каталога (П1 на ходе `topic_by_llm`) решает LLM — как на проде: «есть ли такой анализ».
_FACET_MARK = "понять, что именно пациент хочет узнать об анализе или услуге"
_FACET_AVAILABILITY = json.dumps({"picks": [2]})
FABRICATED = "Подготовка обычно включает специальную диету и отказ от некоторых лекарств. Анализ доступен в нашей клинике."
OFFER = "Перевести?"
# Ветки правил, у которых тему решает LLM, — как на проде (деплой 16): с ними до свободного текста доходит больше ходов.
_PROD_TOPIC_BY_LLM = (
    "rule_test_assist,rule_doctor_info_specialty,rule_schedule,rule_analysis_lookup_ambiguous,rule_website_help,"
    "rule_address_work_hours,rule_test_result,rule_prepare"
)

# Формулировки класса, которые на базе доходят до свободного текста: есть ли анализ (каталог нашёл, поиск не нашёл),
# подбор под цель, вопрос об анализах вообще, условие сдачи.
CLASS_REPLIES = [
    "Есть ли у вас анализ на антитела к кори?",
    "Нужен комплекс анализов для подготовки к беременности, и мужу тоже",
    "Анализ на магний у вас есть?",
    "Какие анализы сдать при постоянной усталости?",
    "Сколько обычно ждать результаты анализов?",
    "Есть ли у вас анализ на кальпротектин в жидком кале?",
]


@pytest.fixture(scope="module")
def ta_world(tmp_path_factory) -> Path:
    source = tmp_path_factory.mktemp("ta_src")
    for sub in ("price_by_region", "price_units"):
        shutil.copytree(_FIXTURE / sub, source / sub)
    (source / "service_info").mkdir()
    (source / "service_info" / "service_info_20261008.jsonl").write_text("", encoding="utf-8")
    shutil.copy2(_DOCTORS, source / _DOCTORS.name)
    root = ev_world.build_world(source, "20261008", tmp_path_factory.mktemp("ta_world") / "world")
    (root / "captured").mkdir()
    (root / "captured" / "kb_main_index.json").write_text("[]", encoding="utf-8")
    return root


def turns(
    world_dir, monkeypatch, replies: list[str], *, switch: str, answers: dict[str, Any], cfg_extra: dict[str, Any] | None = None, wrap: Any = None
) -> list[dict[str, Any]]:
    """Ходы одной сессии; у каждого — текст, телеметрия, вызовы LLM, промпты и висящий оффер оператора. `wrap` — обёртка
    подделки LLM (оракул разметки хода `tests.spans_oracle.SpansOracle`: уточнения позиции)."""

    from messengers_router.endpoint import MessengerGenerateRequest, messenger_generate_once
    from messengers_router.memory import MemoryStore
    from messengers_router.runtime_config import config as cfg
    from messengers_router.services import Services
    from messengers_router.services import _prepare_select_llm

    memory = MemoryStore()
    session = f"ta-{switch}-{replies[0]}"
    out: list[dict[str, Any]] = []

    async def run(reply: str) -> dict[str, Any]:
        req = MessengerGenerateRequest(session_id=session, text=reply, debug=True)
        body = json.loads((await messenger_generate_once(req, memory=memory, services=Services())).body)
        return {"text": str(body.get("text") or ""), "handoff": bool(body.get("handoff"))}

    with ev_world.FrozenWorld(world_dir, "day") as world, monkeypatch.context() as patch:
        patch.setattr(cfg, "MR_TOPIC_BY_LLM", _PROD_TOPIC_BY_LLM, raising=False)
        patch.setattr(cfg, "MR_TEST_ASSIST_FROM_DATA", switch, raising=False)
        for key, value in (cfg_extra or {}).items():
            patch.setattr(cfg, key, value, raising=False)
        patch.setattr(llm_runtime, "_BREAKER", llm_runtime._Breaker())
        patch.setitem(_prepare_select_llm._kb_cache, "memos", ())

        async def regions(_self):
            return list(_REGIONS)

        patch.setattr(Services, "_ensure_regions_loaded", regions)
        for reply in replies:
            fake = _LLM(answers)
            patch.setattr(llm_runtime, "_OLLAMA_CLIENT", wrap(fake) if wrap is not None else fake)
            choice._CACHE.clear()
            result = asyncio.run(run(reply))
            result["telemetry"] = world.take_telemetry() or {}
            result["calls"] = list(fake.calls)
            result["prompts"] = dict(fake.prompts)
            result["pending_offer"] = bool(memory.get(session).last_entities.get("_operator_offer_pending"))
            result["entities"] = dict(memory.get(session).last_entities)
            out.append(result)
    return out


def turn(world_dir, monkeypatch, reply: str, **kwargs: Any) -> dict[str, Any]:
    return turns(world_dir, monkeypatch, [reply], **kwargs)[0]


def _answers(ta_answer: Any) -> dict[str, Any]:
    return {_CLASSIFIER_MARK: _classifier("TEST_ASSIST"), _FACET_MARK: _FACET_AVAILABILITY, _TA_MARK: ta_answer, _RENDER_MARK: FABRICATED}


def _pick(*names: str, outcome: str = "none"):
    return lambda prompt: verdict(outcome, picks=tuple(_number_of(prompt, name) for name in names))


# --------------------------------------------------------------------------- база: класс существует


@pytest.mark.parametrize("reply", CLASS_REPLIES)
def test_off_is_free_text_as_before(reply, ta_world, monkeypatch):
    out = turn(ta_world, monkeypatch, reply, switch="off", answers=_answers(verdict("no_data")))

    assert out["telemetry"].get("answer_source") == "llm_free", out["telemetry"]
    assert _TA_MARK not in out["calls"]
    assert FABRICATED in out["text"]  # на базе свободный текст печатается как есть — класс жив


# --------------------------------------------------------------------------- инвариант под on


@pytest.mark.parametrize("reply", CLASS_REPLIES)
def test_no_data_is_an_honest_offer_not_free_text(reply, ta_world, monkeypatch):
    out = turn(ta_world, monkeypatch, reply, switch="on", answers=_answers(verdict("no_data")))

    assert out["telemetry"].get("answer_source") == "test_assist_data", out["telemetry"]
    assert out["telemetry"].get("ta_answer") == "no_data", out["telemetry"]
    assert _RENDER_MARK not in out["calls"]  # свободный текст не писался
    assert FABRICATED not in out["text"]
    assert out["text"].startswith(NO_DATA_TEXT) and OFFER in out["text"], out["text"]
    assert out["pending_offer"] and not out["handoff"]


def test_empty_pick_without_outcome_is_no_data(ta_world, monkeypatch):
    out = turn(ta_world, monkeypatch, CLASS_REPLIES[1], switch="on", answers=_answers(verdict()))

    assert out["telemetry"].get("ta_answer") == "no_data", out["telemetry"]
    assert out["text"].startswith(NO_DATA_TEXT) and out["pending_offer"]


def test_picked_rows_are_printed_from_the_price(ta_world, monkeypatch):
    out = turn(ta_world, monkeypatch, CLASS_REPLIES[0], switch="on", answers=_answers(_pick("Вирус кори Ig G", "Вирус кори Ig M")))

    assert out["telemetry"].get("answer_source") == "test_assist_data", out["telemetry"]
    assert out["text"].startswith(test_assist_answer.PICKED_HEADING), out["text"]
    # Название и срок — из строки прайса (срок по правилу печати сроков), не из текста LLM; цены нет.
    assert "— Вирус кори Ig G — срок готовности 2–3 дня" in out["text"], out["text"]
    assert "— Вирус кори Ig M — срок готовности до 10 дней" in out["text"], out["text"]
    assert "руб" not in out["text"] and FABRICATED not in out["text"]
    assert not out["pending_offer"]


def test_partly_prints_rows_and_offers_the_rest(ta_world, monkeypatch):
    out = turn(ta_world, monkeypatch, CLASS_REPLIES[5], switch="on", answers=_answers(_pick("Кальпротектин в кале", outcome="partly")))

    assert out["telemetry"].get("ta_answer") == "partly", out["telemetry"]
    assert "— Кальпротектин в кале — срок готовности 1–9 дней" in out["text"], out["text"]
    assert test_assist_answer.PARTLY_TAIL in out["text"] and OFFER in out["text"]
    assert out["pending_offer"] and FABRICATED not in out["text"]


def test_partly_without_rows_is_no_data(ta_world, monkeypatch):
    out = turn(ta_world, monkeypatch, CLASS_REPLIES[5], switch="on", answers=_answers(verdict("partly")))

    assert out["telemetry"].get("ta_answer") == "no_data", out["telemetry"]
    assert out["text"].startswith(NO_DATA_TEXT)


def test_question_about_analyses_in_general_asks_for_names(ta_world, monkeypatch):
    out = turn(ta_world, monkeypatch, CLASS_REPLIES[4], switch="on", answers=_answers(verdict("need_name")))

    assert out["telemetry"].get("ta_answer") == "need_name", out["telemetry"]
    assert out["text"] == test_assist_answer.NEED_NAME_TEXT
    assert not out["pending_offer"]


@pytest.mark.parametrize("failure", [ConnectionError("тест: LLM недоступна"), "не json", verdict(picks=(99,))])
def test_llm_failure_or_garbage_is_an_offer_not_free_text(failure, ta_world, monkeypatch):
    out = turn(ta_world, monkeypatch, CLASS_REPLIES[0], switch="on", answers=_answers(failure))

    assert out["telemetry"].get("answer_source") == "test_assist_data", out["telemetry"]
    assert out["telemetry"].get("ta_answer") == "failed", out["telemetry"]
    assert _RENDER_MARK not in out["calls"] and FABRICATED not in out["text"]
    assert out["text"].startswith(NO_DATA_TEXT) and out["pending_offer"]


def test_decision_budget_refusal_is_an_offer(ta_world, monkeypatch):
    # Решение хода до ответа — свойство вопроса об услуге из каталога; на ответ подбора анализов лимита не остаётся. Разбор фраз
    # подделка не отвечает: сбой вызова лимит не тратит (решение владельца 26.10), поэтому лимит — одно решение.
    out = turn(
        ta_world, monkeypatch, CLASS_REPLIES[0], switch="on", answers=_answers(verdict("no_data")), cfg_extra={"MR_LLM_DECISION_TURN_CALLS": 1}
    )

    assert out["telemetry"].get("ta_answer") == "failed", out["telemetry"]
    assert out["telemetry"].get("ta_fail") == "budget", out["telemetry"]
    assert _RENDER_MARK not in out["calls"] and out["text"].startswith(NO_DATA_TEXT)


# --------------------------------------------------------------------------- кандидаты и контекст


def test_row_the_turn_tool_missed_is_a_candidate(ta_world, monkeypatch):
    # Поиск подбора анализов ищет по всей реплике и «магний» в вопросе не находит; пул выбора строки цены — находит.
    out = turn(ta_world, monkeypatch, CLASS_REPLIES[2], switch="on", answers=_answers(_pick("Mg+ (магний)")))

    assert out["telemetry"].get("ta_tool_rows") == 0, out["telemetry"]
    assert "— Mg+ (магний) — срок готовности 1–2 дня" in out["text"], out["text"]


def test_catalog_guess_is_a_candidate_without_confirm(ta_world, monkeypatch):
    # Суд «да/нет» по нечёткой догадке каталога на этом ходе пропущен (он тратил решения хода), но сама догадка — кандидат
    # ответа: поиск по реплике её строку находит не всегда («уровень железа в крови» → только «CA 15-3 (молочная железа)»).
    from messengers_router.services import Services

    async def fuzzy(_self, raw_text_or_name, **_kwargs):
        return {"status": "fuzzy", "canonical": "Железо в сыворотке", "query": str(raw_text_or_name)}

    monkeypatch.setattr(Services, "match_catalog_service", fuzzy)
    out = turn(ta_world, monkeypatch, "Есть анализ на ЖС?", switch="on", answers=_answers(_pick("Железо в сыворотке")))

    assert _number_of(out["prompts"][_TA_MARK], "Железо в сыворотке") == 1  # первой — до строк поиска по реплике
    assert "— Железо в сыворотке — срок готовности" in out["text"], out["text"]
    assert "Похоже, вы имели в виду" not in out["text"]
    assert test_assist_answer.CATALOG_GUESS_KEY not in out["entities"]  # служебный ключ хода в состояние диалога не попадает


def test_previous_bot_message_is_the_context_of_a_short_reply(ta_world, monkeypatch):
    answers = _answers(_pick("Вирус кори Ig G", "Вирус кори Ig M"))
    first, second = turns(ta_world, monkeypatch, [CLASS_REPLIES[0], "А есть ли такой анализ для ребёнка?"], switch="on", answers=answers)

    assert "Предыдущее сообщение бота: «(диалог только начался)»" in first["prompts"][_TA_MARK]
    assert "Предыдущее сообщение бота: «В прайсе клиники есть: — Вирус кори" in second["prompts"][_TA_MARK], second["prompts"]


# --------------------------------------------------------------------------- цена у выбранных строк (`test_assist_price`)
#
# Решение владельца 26.10 (топ-5 разбора 95 цен, п. 1): у выбранных строк — цена, кодом, из той же строки прайса;
# частичный ответ — цены найденных строк + «остального в прайсе нет — уточнит оператор». Инвариант класса (принцип 1):
# каждая сумма ответа — сумма той строки прайса, название которой напечатано рядом, и строки — только выбранные LLM;
# выбор LLM от выключателя не зависит (промпт тот же — цены модель не видит); давность прайса — как у пути цены.

_PRICE = {"MR_TEST_ASSIST_PRICE": "on"}


def _catalog():
    from evalbench.prices import PriceCatalog, _read_jsonl

    return PriceCatalog(_read_jsonl(next((_FIXTURE / "price_by_region").glob("price_region_3_*.jsonl"))))


def _listed(prompt: str) -> list[tuple[int, str]]:
    out = []
    for line in prompt.splitlines():
        head, _, rest = line.partition(". ")
        if head.isdigit():
            out.append((int(head), rest.split(" — срок готовности")[0]))
    return out


def _first_two(outcome: str = "none"):
    return lambda prompt: verdict(outcome, picks=tuple(number for number, _name in _listed(prompt)[:2]))


@pytest.mark.parametrize("reply", CLASS_REPLIES)
@pytest.mark.parametrize("outcome", ["none", "partly"])
def test_every_printed_price_is_the_price_of_its_own_picked_row(reply, outcome, ta_world, monkeypatch):
    out = turn(ta_world, monkeypatch, reply, switch="on", answers=_answers(_first_two(outcome)), cfg_extra=_PRICE)

    picked = {name for _number, name in _listed(out["prompts"].get(_TA_MARK, ""))[:2]}
    pairs = _catalog().pairs(out["text"])
    assert all(pair.status == "matched" for pair in pairs), (pairs, out["text"])  # ни одной суммы без своей строки
    assert {pair.name for pair in pairs} <= picked, (pairs, picked)  # строки — только выбранные LLM
    if picked:
        assert {pair.name for pair in pairs} == picked, out["text"]  # у каждой выбранной строки — её цена
        assert test_assist_answer.PICKED_TAIL not in out["text"]  # «цену подскажу по названию» рядом с ценой — нет
        assert "Цены актуальны для г. Самара" in out["text"]
    assert FABRICATED not in out["text"] and _RENDER_MARK not in out["calls"]


def test_priced_rows_show_name_price_and_deadline_of_the_row(ta_world, monkeypatch):
    out = turn(
        ta_world, monkeypatch, CLASS_REPLIES[0], switch="on", answers=_answers(_pick("Вирус кори Ig G", "Вирус кори Ig M")), cfg_extra=_PRICE
    )

    assert "— Вирус кори Ig G — 800 руб., срок готовности 2–3 дня" in out["text"], out["text"]
    assert "— Вирус кори Ig M — 930 руб., срок готовности до 10 дней" in out["text"], out["text"]
    assert test_assist_answer.PRICED_TAIL in out["text"]
    assert out["telemetry"].get("ta_priced") == 2, out["telemetry"]
    assert not out["pending_offer"]


def test_priced_partly_prints_found_prices_and_offers_the_rest(ta_world, monkeypatch):
    out = turn(
        ta_world, monkeypatch, CLASS_REPLIES[5], switch="on", answers=_answers(_pick("Кальпротектин в кале", outcome="partly")), cfg_extra=_PRICE
    )

    assert "— Кальпротектин в кале — 2 730 руб., срок готовности 1–9 дней" in out["text"], out["text"]
    # Решение владельца 26.10 (вариант 3): «точно в таком виде нет» — до строк и цен, чтобы цена не читалась ответом «да».
    assert out["text"].startswith(test_assist_answer.PRICED_PARTLY_HEADING), out["text"]
    assert test_assist_answer.PICKED_HEADING not in out["text"]
    assert test_assist_answer.PRICED_PARTLY_TAIL in out["text"]
    assert out["text"].endswith(OFFER) and out["pending_offer"]  # вопрос-оффер — последним, после приписки о ценах


def test_priced_full_answer_keeps_the_plain_heading(ta_world, monkeypatch):
    out = turn(ta_world, monkeypatch, CLASS_REPLIES[0], switch="on", answers=_answers(_pick("Вирус кори Ig G")), cfg_extra=_PRICE)

    assert out["text"].startswith(test_assist_answer.PICKED_HEADING), out["text"]
    assert test_assist_answer.PRICED_PARTLY_HEADING not in out["text"]


@pytest.mark.parametrize("ta_answer", [verdict("no_data"), verdict("need_name"), ConnectionError("тест: LLM недоступна")])
def test_no_rows_no_price(ta_answer, ta_world, monkeypatch):
    out = turn(ta_world, monkeypatch, CLASS_REPLIES[0], switch="on", answers=_answers(ta_answer), cfg_extra=_PRICE)

    assert "руб" not in out["text"], out["text"]
    assert "ta_priced" not in out["telemetry"]


def test_llm_sees_the_same_options_with_and_without_price(ta_world, monkeypatch):
    # Цена — оформление ответа, не довод выбора: промпт тот же, повтор из кэша под on берёт те же ответы модели.
    answers = _answers(_pick("Вирус кори Ig G"))
    off = turn(ta_world, monkeypatch, CLASS_REPLIES[0], switch="on", answers=answers)
    on = turn(ta_world, monkeypatch, CLASS_REPLIES[0], switch="on", answers=answers, cfg_extra=_PRICE)

    assert off["prompts"][_TA_MARK] == on["prompts"][_TA_MARK]
    assert "руб" not in on["prompts"][_TA_MARK]


@pytest.mark.parametrize(("age", "expect"), [(18, None), (5, "по данным на")])
def test_price_age_follows_the_price_path(age, expect, ta_world, monkeypatch):
    # Срез прайса старше недели — путь цены переводит на оператора; здесь — строки без цен, как под off. 3–7 дней — дата.
    import datetime

    from agent_logic_2.nayka_api import snapshots

    stale = snapshots.Snapshot(name="price_region_3", rows=[], date=datetime.date(2026, 10, 8) - datetime.timedelta(days=age), status="stale", age_days=age)
    monkeypatch.setattr(snapshots, "last", lambda prefix: stale if prefix.startswith("price_region") else None)
    out = turn(ta_world, monkeypatch, CLASS_REPLIES[0], switch="on", answers=_answers(_pick("Вирус кори Ig G")), cfg_extra=_PRICE)

    if expect is None:
        assert "руб" not in out["text"] and test_assist_answer.PICKED_TAIL in out["text"], out["text"]
    else:
        assert "— Вирус кори Ig G — 800 руб." in out["text"] and expect in out["text"], out["text"]


@pytest.mark.parametrize("path", ["messengers_router/prompts/choose_test_assist_answer.txt", "app_data/prompts/mr_choose_test_assist_answer.txt"])
def test_partly_asks_for_all_variants_like_none(path):
    # Карточка п. 1 (26.10): g0388 «антитела к аннексину суммарные» — в списке IgG и IgM, LLM выбрала одну: «все варианты» промпт
    # просил только у точного ответа. С ценой одна строка из двух вариантов — неполный ответ ценой (принцип 2: сомнение → список).
    text = (Path(__file__).resolve().parents[1] / path).read_text(encoding="utf-8")
    partly = next(line for line in text.splitlines() if line.startswith('- "partly"'))
    assert "выбери их все, до 6" in partly
