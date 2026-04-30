"""Graph/FSM-слой для явного состояния диалога пациента.

Ответственность модуля:
1) Хранить и нормализовать диалоговые состояния (IDLE/FLOW/HITL).
2) Выполнять предсказуемые переходы между состояниями по текущему решению,
   pending-слотам и признакам handoff.
3) Возвращать transition metadata для debug-trace и последующей аналитики.

GraphEngine не меняет внешний API и не заменяет роутер:
это state-machine ядро, которое делает поведение детерминированным.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .mess_types import RouteDecision, SessionState

log = logging.getLogger(__name__)


class DialogState(str, Enum):
    IDLE = "IDLE"
    APPOINTMENT_FLOW = "APPOINTMENT_FLOW"
    TEST_RESULT_FLOW = "TEST_RESULT_FLOW"
    ADDRESS_FLOW = "ADDRESS_FLOW"
    CLARIFY_FLOW = "CLARIFY_FLOW"
    HITL_PENDING = "HITL_PENDING"


@dataclass
class GraphTransition:
    from_state: DialogState
    to_state: DialogState
    reason: str


@dataclass
class GraphOutput:
    state: DialogState
    transition: GraphTransition
    metadata: dict[str, Any]


def get_dialog_state(state: SessionState) -> DialogState:
    raw = str(state.last_entities.get("_dialog_state") or "").strip()
    # Пустое значение — штатная ситуация для свежих сессий, а также для
    # сессий, в которых ``_dialog_state`` ещё не записывали. Раньше пустая
    # строка падала в DialogState('') -> ValueError и заполняла логи
    # WARNING-стек-трейсами на каждом первом ходу.
    if not raw:
        return DialogState.IDLE
    try:
        return DialogState(raw)
    except Exception:
        log.warning("dialog_state_deserialize_failed raw=%r", raw, exc_info=True)
        return DialogState.IDLE


def set_dialog_state(state: SessionState, dialog_state: DialogState) -> None:
    state.last_entities["_dialog_state"] = dialog_state.value


class GraphEngine:
    """
    Явный state-machine слой поверх текущего роутера.
    На текущем этапе не меняет внешний контракт, только нормализует transitions.
    """

    def next(
        self,
        *,
        session: SessionState,
        decision: RouteDecision,
        pending: dict[str, Any] | None,
        handoff_planned: bool = False,
    ) -> GraphOutput:
        cur = get_dialog_state(session)
        reason = "default_idle"
        nxt = DialogState.IDLE

        if handoff_planned or decision.needs_handoff:
            nxt = DialogState.HITL_PENDING
            reason = "handoff"
        elif isinstance(pending, dict):
            nxt = DialogState.CLARIFY_FLOW
            reason = f"pending:{pending.get('label')}"
        elif decision.label == "APPOINTMENT" or bool(session.last_entities.get("appointment_flow_active")):
            nxt = DialogState.APPOINTMENT_FLOW
            reason = "appointment_flow"
        elif decision.label == "TEST_RESULT":
            nxt = DialogState.TEST_RESULT_FLOW
            reason = "test_result_flow"
        elif decision.label == "ADDRESS":
            nxt = DialogState.ADDRESS_FLOW
            reason = "address_flow"

        set_dialog_state(session, nxt)
        return GraphOutput(
            state=nxt,
            transition=GraphTransition(from_state=cur, to_state=nxt, reason=reason),
            metadata={"label": decision.label, "needs_handoff": decision.needs_handoff},
        )
