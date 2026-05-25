from messengers_router.memory import MemoryStore
from messengers_router.mess_types import RouteDecision, SessionState
from messengers_router.planner import build_plan


def _plan(text, entities=None):
    decision = RouteDecision(label="PREPARE", confidence=0.9, entities={}, flags=set())
    state = SessionState(session_id="t")
    state.last_entities = dict(entities or {})
    return build_plan(decision, state, text, MemoryStore())


def test_lab_visit_timing_prepare_schedules_test_prepare():
    """«во сколько/когда прийти сдать кровь» должен дойти до test_prepare
    (а не уйти в clarify-gate с пустыми steps), иначе фикс адресов+графика
    никогда не отрабатывает."""
    for text in ("во сколько прийти сдать кровь", "когда сдавать кровь", "во сколько можно сдать мочу"):
        plan = _plan(text)
        assert any(getattr(s, "tool", "") == "test_prepare" for s in plan.steps), (text, plan.steps)


def test_plain_prepare_without_subject_still_clarifies():
    # Обычный PREPARE без анализа и без «во сколько/когда» — прежнее поведение:
    # пустые steps (clarify-gate спросит название анализа).
    assert _plan("подготовка").steps == []
