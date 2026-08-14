"""П1 (прод 31.07, 09.08): воронка TEST_ASSIST глотала НАЗВАННЫЕ анализы.

Симптом: пациент называет анализ («С-реактивный белок», «спермограмма»,
«липидограмма», второй ход в списке), бот отвечает воронкой подбора «Для какой
цели хотите подобрать анализы? Например: проверить щитовидку…». Кластер №2
триажа за 14 дней: 17 диалогов, 82% провал.

Корень — МАРШРУТИЗАЦИЯ, не поиск. Каталог такие названия резолвит (проверено
офлайн: `resolve_price_service_name_from_catalog('С-реактивный белок')` →
`CРБ "С" реактивный белок (количественный метод)`), но голое название без
ценового глагола уходит в TEST_ASSIST; `REQUIRED_SLOTS["TEST_ASSIST"]` требует
`test_goal|test_name`, service_name его не закрывает → generic clarify.

Инвариант (класс): если реплика ПРИЗЕМЛИЛАСЬ на реальную услугу каталога
(exact-матч в `_inject_catalog_candidates`), это PRICE. TEST_ASSIST остаётся
только когда услуга НЕ найдена — воронка подбора существует ровно для этого.
Исключения (не относятся к классу): категорийный запрос («чекап» — линейка
пакетов, а не услуга) и вопрос о СРОКАХ ГОТОВНОСТИ результата
(`result_delivery_info`) — цену там показывать нельзя.

Тесты идут по ЖИВОМУ каталогу (region 3), как соседние price-тесты: класс, а не
инстанс — список названий взят из прод-триажа.
"""

from __future__ import annotations

import asyncio

import pytest

from messengers_router import router as router_mod
from messengers_router.memory import MemoryStore
from messengers_router.mess_types import Evidence, RouteDecision, SessionState
from messengers_router.nlu_pipeline import NLUResult
from messengers_router.services import Services


def run(coro):
    return asyncio.run(coro)


def _install_stubs(monkeypatch, *, label: str, flags: set[str], entities: dict | None = None):
    """LLM/rule-слой отдаёт заданное решение; executor не ходит в сеть."""

    async def fake_analyze_with_candidates(_text, _state, runtime_options=None):
        _ = runtime_options
        return NLUResult(
            decision=RouteDecision(
                label=label,
                confidence=0.9,
                entities=dict(entities or {}),
                flags=set(flags),
                needs_handoff=False,
            ),
            candidates=[],
            merged_from="rule",
        )

    def fake_env_flag(name: str, default: bool) -> bool:
        if name == "MR_ROUTER_V2_ENABLE":
            return True
        if name == "MR_ROUTER_V2_SHADOW":
            return False
        return default

    async def fake_execute_plan(_plan, _state, _services):
        return Evidence()

    monkeypatch.setattr(router_mod, "analyze_with_candidates", fake_analyze_with_candidates)
    monkeypatch.setattr(router_mod, "_env_flag", fake_env_flag)
    monkeypatch.setattr(router_mod, "execute_plan", fake_execute_plan)


def _route(text: str, session_id: str):
    state = SessionState(session_id=session_id)
    return run(router_mod.route_patient_message(text, state, Services(), MemoryStore()))


# --- Класс: названный анализ из каталога → PRICE, не воронка --------------

# Список из прод-триажа (кластер №2, 14 дней). Все резолвятся в живом каталоге.
NAMED_TESTS = [
    "С-реактивный белок",
    "спермограмма",
    "Алат",
    "АСАТ",
    "липидограмма",
    "IgG к вирусу кори",
]


@pytest.mark.parametrize("text", NAMED_TESTS)
def test_named_catalog_test_routes_to_price_not_assist_funnel(monkeypatch, text):
    """Голое название анализа, размеченное как TEST_ASSIST, приземляется на
    услугу каталога → метка PRICE, service_name запиннен."""
    _install_stubs(monkeypatch, label="TEST_ASSIST", flags={"rule_test_assist"})

    decision, plan, _ev = _route(text, f"p1-{text}")

    assert decision.label == "PRICE", f"{text!r}: {decision.label} (flags={sorted(decision.flags)})"
    assert plan.label == "PRICE"
    assert (decision.entities or {}).get("service_name"), decision.entities
    assert "test_assist_catalog_hit_is_price" in decision.flags


@pytest.mark.parametrize("text", NAMED_TESTS)
def test_named_catalog_test_does_not_ask_test_goal(monkeypatch, text):
    """Прямое следствие: слот `test_goal` больше не «недостающий» → generic
    clarify воронки не строится."""
    from messengers_router.policies import missing_slots

    _install_stubs(monkeypatch, label="TEST_ASSIST", flags={"rule_test_assist"})

    decision, _plan, _ev = _route(text, f"p1-slots-{text}")

    missing = missing_slots(decision.label, {"city": "Самара", **(decision.entities or {})})
    assert not missing, f"{text!r}: остались слоты {missing}"


# --- Анти-over-trigger: настоящая воронка подбора остаётся TEST_ASSIST -----

@pytest.mark.parametrize(
    "text",
    [
        "проверить щитовидку",
        "какие анализы сдать для профилактики",
        "хочу подобрать анализы",
    ],
    ids=["thyroid_goal", "which_tests", "pick_tests"],
)
def test_goal_query_without_catalog_hit_stays_test_assist(monkeypatch, text):
    """Цель (не услуга) в каталоге не резолвится → воронка подбора сохраняется."""
    _install_stubs(monkeypatch, label="TEST_ASSIST", flags={"rule_test_assist"})

    decision, plan, _ev = _route(text, f"p1-goal-{text}")

    assert decision.label == "TEST_ASSIST", sorted(decision.flags)
    assert plan.label == "TEST_ASSIST"
    assert "test_assist_catalog_hit_is_price" not in decision.flags


def test_goal_phrasing_blocks_relabel_even_on_catalog_hit(monkeypatch):
    """Гард формулировки цели: «какие анализы сдать для профилактики» каталог
    резолвит в «Здоровая молодость: курс для профилактики и восстановления» по
    ОДНОМУ общему токену. Спрашивают цель, а не цену этой услуги → PRICE нельзя."""
    from messengers_router.services._prices_helpers import (
        resolve_price_service_name_from_catalog,
    )

    text = "какие анализы сдать для профилактики"
    # Гард окружения: каталожный матч реально есть (иначе тест ничего не проверяет).
    assert resolve_price_service_name_from_catalog(text), "каталог перестал матчить — кейс устарел"

    _install_stubs(monkeypatch, label="TEST_ASSIST", flags={"rule_test_assist"})

    decision, plan, _ev = _route(text, "p1-goal-guard")

    assert decision.label == "TEST_ASSIST"
    assert plan.label == "TEST_ASSIST"
    assert "catalog_service_exact" in decision.flags, "предусловие: exact-матч состоялся"
    assert "test_assist_goal_phrasing_kept" in decision.flags
    assert "test_assist_catalog_hit_is_price" not in decision.flags


@pytest.mark.parametrize(
    "text, is_goal",
    [
        ("какие анализы сдать для профилактики", True),
        ("хочу подобрать анализы", True),
        ("что сдать из анализов перед операцией", True),
        ("комплекс анализов для мужчин", True),
        ("скрининг", True),
        ("чекап", True),
        # НЕ формулировка цели — это названия услуг:
        ("С-реактивный белок", False),
        ("спермограмма", False),
        ("общий анализ крови", False),
        ("липидограмма", False),
    ],
)
def test_goal_phrasing_detector_separates_goal_from_service_name(text, is_goal):
    """Класс: детектор отделяет ВОПРОС О ЦЕЛИ от НАЗВАНИЯ услуги — в том числе
    для названий, содержащих слово «анализ»."""
    from messengers_router.policies import detect_test_assist_goal_phrasing

    assert detect_test_assist_goal_phrasing(text) is is_goal, text


def test_checkup_category_still_kept_broad(monkeypatch):
    """«Чекап» — КАТЕГОРИЯ: service_name не пиннится (Bug #2), значит и
    переклейка в PRICE не срабатывает — линейка пакетов не схлопывается."""
    _install_stubs(monkeypatch, label="TEST_ASSIST", flags={"rule_test_assist"})

    decision, plan, _ev = _route("чекап", "p1-checkup")

    assert decision.label == "TEST_ASSIST"
    assert plan.label == "TEST_ASSIST"
    assert "test_assist_category_kept_broad" in decision.flags
    assert not (decision.entities or {}).get("service_name")


def test_walkin_where_question_wins_over_price_relabel(monkeypatch):
    """Порядок слоёв: устоявшиеся guardrails сильнее переклейки. «Диабетический
    профиль 1 ГДЕ можно сдать?» — вопрос о МЕСТЕ; walk-in промоушен уводит метку
    в ADDRESS, и переклейка в PRICE уже не применяется."""
    _install_stubs(monkeypatch, label="TEST_ASSIST", flags={"rule_test_assist"})

    state = SessionState(
        session_id="p1-walkin-where",
        last_entities={"test_goal": "Какие анализы сдать на сахарный диабет"},
    )
    decision, plan, _ev = run(
        router_mod.route_patient_message(
            "Диабетический профиль 1 где можно сдать?", state, Services(), MemoryStore()
        )
    )

    assert decision.label == "ADDRESS", sorted(decision.flags)
    assert plan.label == "ADDRESS"
    assert "test_assist_catalog_hit_is_price" not in decision.flags


def test_result_delivery_question_never_becomes_price(monkeypatch):
    """Вопрос о СРОКАХ готовности результата (`result_delivery_info`) — не цена.
    Даже если каталог резолвит упомянутый анализ, цену показывать нельзя."""
    _install_stubs(
        monkeypatch,
        label="TEST_ASSIST",
        flags={"rule_test_assist", "result_delivery_info"},
    )

    decision, plan, _ev = _route("когда будут готовы результаты общего анализа крови", "p1-delivery")

    assert decision.label == "TEST_ASSIST", sorted(decision.flags)
    assert plan.label == "TEST_ASSIST"
    assert "test_assist_catalog_hit_is_price" not in decision.flags
