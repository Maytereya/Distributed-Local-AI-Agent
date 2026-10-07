"""Памятки подготовки — как есть; «с наркозом или без?» — сначала спросить (07.10).

Вопрос владельца 07.10: «как подготовиться к колоноскопии» → пересказ LLM памятки базы знаний
с ошибками («Пикопреп» при дневной колоноскопии — «2 пакетика за 8–6 часов» вместо «первый в
21:00 накануне», схема утренней колоноскопии — «по схеме» без схемы). Памяток ФКС в базе
знаний три (без наркоза, с наркозом, ФКС + ФГДС с наркозом), бот брал одну наугад.

Решение владельца 07.10: памятка — как есть, без цен; длинный ответ режет шлюз; есть
варианты «с наркозом / без» — сначала спросить.

Инварианты: пациент получает текст памятки клиники целиком, без LLM; при вариантах — вопрос,
ответ выбирает памятку; «не знаю» и повторное «да/нет» — обе; другая тема снимает вопрос.
"""

from __future__ import annotations

import asyncio

import pytest

from messengers_router import router as router_mod
from messengers_router.memory import MemoryStore
from messengers_router.mess_types import SessionState
from messengers_router.response_builder import PREPARE_VARIANT_PENDING_KEY
from messengers_router.services import Services
from messengers_router.services import core as svc_mod
from messengers_router.services import prepare as prepare_mod
from messengers_router.services._prepare_select_llm import SOURCE_KB, SOURCE_MIS, PrepareMemo

_WITHOUT = PrepareMemo(SOURCE_KB, "ПАМЯТКА ФКС БЕЗ НАРКОЗА ", "ФКС без наркоза: Пикопреп 1-й пакетик в 21:00 накануне.")
_WITH = PrepareMemo(SOURCE_KB, "ПАМЯТКА ФКС с наркозом", "ФКС с наркозом: за 3 часа не пить, нужен сопровождающий.")
_COMBINED = PrepareMemo(SOURCE_KB, "ПАМЯТКА  ФКС + ФГДС с наркозом", "ФКС и ФГДС с наркозом: анализы заранее.")
_GASTRO = PrepareMemo(SOURCE_KB, "ПАМЯТКА Гастроскопия (без наркоза)", "Гастроскопия без наркоза: не есть 8 часов.")
_KB = (_GASTRO, _WITHOUT, _WITH, _COMBINED)


@pytest.fixture
def services(monkeypatch):
    picked: dict[str, PrepareMemo] = {"memo": _WITHOUT}

    async def select(_question, memos, **_kwargs):
        return (picked["memo"],)

    async def no_llm(*_args, **_kwargs):
        raise AssertionError("памятку не переписываем")

    monkeypatch.setattr(svc_mod.api_service_info, "load_service_info", lambda: [])
    monkeypatch.setattr(prepare_mod, "kb_patient_memos", lambda: _KB)
    monkeypatch.setattr(prepare_mod, "select_prepare_memos", select)
    monkeypatch.setattr("messengers_router.llm_runtime.generate_text", no_llm)
    svc = Services()
    svc.ensure_background_refresh_started = lambda: None
    svc.picked = picked
    return svc


def _dialog(services, *turns: str) -> tuple[list[str], SessionState]:
    state, memory = SessionState(session_id="prep-variants"), MemoryStore()

    async def go():
        answers = []
        for text in turns:
            out = [env async for env in router_mod.patient_routing_stream(text, state, services, memory)]
            answers.append("".join(env.text for env in out if env.text))
        return answers

    return asyncio.run(go()), state


def test_procedure_with_variants_asks_about_anesthesia_first(services):
    (answer,), state = _dialog(services, "как подготовиться к колоноскопии")

    assert answer == prepare_mod._ANESTHESIA_QUESTION
    assert state.last_entities.get(PREPARE_VARIANT_PENDING_KEY)


@pytest.mark.parametrize(
    ("reply", "shown", "hidden"),
    [
        ("с наркозом", [_WITH], [_WITHOUT]),
        ("во сне", [_WITH], [_WITHOUT]),
        ("без наркоза", [_WITHOUT], [_WITH]),
        ("без", [_WITHOUT], [_WITH]),
        ("не знаю", [_WITHOUT, _WITH], []),
    ],
)
def test_reply_picks_the_memo_as_is(services, reply, shown, hidden):
    (_, answer), state = _dialog(services, "как подготовиться к колоноскопии", reply)

    for memo in shown:
        assert memo.text in answer
    for memo in hidden:
        assert memo.text not in answer
    assert not state.last_entities.get(PREPARE_VARIANT_PENDING_KEY)


def test_yes_or_no_is_asked_again_once_then_both(services):
    (_, again, both), _ = _dialog(services, "как подготовиться к колоноскопии", "да", "нет")

    assert "«с наркозом» или «без наркоза»" in again
    assert _WITH.text in both and _WITHOUT.text in both


def test_other_topic_drops_the_question(services):
    (_, answer), state = _dialog(services, "как подготовиться к колоноскопии", "адреса филиалов")

    assert not state.last_entities.get(PREPARE_VARIANT_PENDING_KEY)
    assert _WITH.text not in answer and _WITHOUT.text not in answer


@pytest.mark.parametrize(("query", "memo"), [("как подготовиться к колоноскопии под наркозом", _WITH), ("подготовка к фкс без наркоза", _WITHOUT)])
def test_anesthesia_in_the_question_skips_the_question(services, query, memo):
    (answer,), _ = _dialog(services, query)

    assert memo.text in answer
    assert prepare_mod._ANESTHESIA_QUESTION not in answer


def test_procedure_without_variants_goes_as_is(services):
    services.picked["memo"] = _COMBINED

    (answer,), _ = _dialog(services, "как подготовиться к фкс и фгдс")

    assert answer == f"Памятка клиники «{_COMBINED.title}»:\n{_COMBINED.text}"


def test_long_mis_memo_is_not_compressed(services):
    # 31 памятка МИС длиннее 3500 символов раньше ужималась правилами; длинный ответ режет шлюз.
    text = " ".join(f"Шаг {i}: соблюдайте рекомендации врача." for i in range(1, 200))
    services.picked["memo"] = PrepareMemo(SOURCE_MIS, "Цитомегаловирус соскоб [кач.]", text)

    (answer,), _ = _dialog(services, "подготовка к соскобу на цитомегаловирус")

    assert len(text) > 3500
    assert text in answer
