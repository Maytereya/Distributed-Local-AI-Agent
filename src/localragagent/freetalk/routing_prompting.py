"""Prompt templates for FreeTalk."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_system_prompt(prompt_path: Path) -> str:
    if prompt_path.exists():
        try:
            return prompt_path.read_text(encoding="utf-8").strip()
        except Exception:
            pass
    return (
        "Ты разговорный ассистент клиники. "
        "Для медицинских запросов сначала используй tool_call, "
        "не выдумывай факты клиники."
    )


def _history_lines(turns: list[dict[str, Any]], *, max_turns: int) -> str:
    selected = turns[-max_turns:] if max_turns > 0 else list(turns)
    lines: list[str] = []
    for turn in selected:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("role") or "").strip() or "user"
        content = str(turn.get("content") or "").strip()
        if not content:
            continue
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def build_general_prompt(
    *,
    system_prompt: str,
    summary: str,
    turns: list[dict[str, Any]],
    user_message: str,
) -> str:
    history = _history_lines(turns, max_turns=12)
    summary_text = str(summary or "").strip() or "(нет)"
    return (
        f"{system_prompt}\n\n"
        "Контекст диалога (summary):\n"
        f"{summary_text}\n\n"
        "Последние реплики:\n"
        f"{history or '(нет)'}\n\n"
        "Задача:\n"
        "Ответь кратко и по существу.\n"
        "Не используй служебные фразы вроде 'выполню tool_call' или 'подождите'.\n"
        "Либо дай итоговый ответ сразу, либо честно скажи, что точных данных нет.\n\n"
        f"Пользователь: {user_message}\n"
        "Ассистент:"
    )


def build_tool_result_prompt(
    *,
    system_prompt: str,
    user_message: str,
    tool_name: str,
    tool_payload: dict[str, Any],
) -> str:
    payload_text = json.dumps(tool_payload, ensure_ascii=False, default=str)
    if len(payload_text) > 14000:
        payload_text = payload_text[:14000] + "...(truncated)"
    return (
        f"{system_prompt}\n\n"
        "Ниже результат tool_call. Используй только релевантные факты для ответа.\n"
        "Не показывай JSON целиком. Не добавляй несуществующие факты.\n\n"
        f"Вопрос пользователя: {user_message}\n"
        f"Tool: {tool_name}\n"
        f"Payload: {payload_text}\n\n"
        "Сформируй короткий понятный ответ:"
    )


def build_summary_prompt(*, previous_summary: str, turns: list[dict[str, Any]]) -> str:
    history = _history_lines(turns, max_turns=24)
    return (
        "Сделай компактное summary диалога.\n"
        "Нужно: ключевые факты о запросах пользователя, принятые решения, незакрытые вопросы.\n"
        "Формат: 5-8 коротких предложений.\n\n"
        f"Предыдущее summary:\n{previous_summary or '(нет)'}\n\n"
        f"Новые реплики:\n{history or '(нет)'}\n\n"
        "Обновленное summary:"
    )


def build_clinical_router_prompt(
    *,
    system_prompt: str,
    summary: str,
    turns: list[dict[str, Any]],
    user_message: str,
    pending_intent: str = "",
    pending_slots: list[str] | None = None,
    remembered_doctor: str = "",
    dialog_state: dict[str, Any] | None = None,
) -> str:
    history = _history_lines(turns, max_turns=12)
    summary_text = str(summary or "").strip() or "(нет)"
    pending_intent_text = str(pending_intent or "").strip() or "(нет)"
    pending_slots_text = ", ".join([str(x).strip() for x in (pending_slots or []) if str(x).strip()]) or "(нет)"
    doctor_hint = str(remembered_doctor or "").strip() or "(нет)"
    state_payload = dialog_state if isinstance(dialog_state, dict) else {}
    state_phase = str(state_payload.get("phase") or "").strip() or "(нет)"
    state_clarify_type = str(state_payload.get("clarify_type") or "").strip() or "(нет)"
    state_confirmation_target = str(state_payload.get("confirmation_target") or "").strip() or "(нет)"
    state_open_question = str(state_payload.get("open_question") or "").strip() or "(нет)"
    state_entities = state_payload.get("entities") if isinstance(state_payload.get("entities"), dict) else {}
    state_candidates = (
        state_payload.get("candidate_entities")
        if isinstance(state_payload.get("candidate_entities"), dict)
        else {}
    )
    schema = {
        "intent": "appointment|doctor_schedule|doctor_info|price|prepare|tests|test_result|address|clinic_documents|clinic_news|service_info|unknown",
        "confidence": 0.0,
        "entities": {
            "appointment_action": "",
            "doctor_name": "",
            "specialty": "",
            "service_name": "",
            "service_variant": "",
            "test_name": "",
            "city": "",
            "branch_name": "",
            "date_from": "",
            "date_to": "",
            "time_from": "",
            "time_to": "",
            "date": "",
            "time": "",
            "result_surname": "",
            "result_year_of_birth": "",
            "result_analysis_code": "",
            "result_analysis_number": "",
            "doctor_id": "",
            "patient_name": "",
            "child_age": "",
        },
        "missing_slots": ["doctor_name", "specialty"],
        "clarify_type": "identify|confirm_candidate|narrow_choice|missing_auth_data|other",
        "clarify_question": "",
        "tool_plan": ["doctors_schedule_week", "doctors_info"],
    }
    schema_text = json.dumps(schema, ensure_ascii=False, indent=2)
    return (
        f"{system_prompt}\n\n"
        "Ты роутер клинических интентов. Выбери intent и сущности для tool_call.\n"
        "Если данных недостаточно, заполни missing_slots, clarify_type и короткий clarify_question.\n"
        "Если пользователь пишет \"о нем/его/этот врач\", используй контекст и remembered_doctor.\n"
        "Если это продолжение предыдущего уточнения, дополни уже собранные сущности, а не начинай разбор заново.\n"
        "Если пользователь хочет записаться, перенести или отменить запись, используй intent=appointment.\n"
        "Типы уточнений:\n"
        "- identify: не хватает базовой сущности, кого/что искать.\n"
        "- confirm_candidate: есть один вероятный кандидат, нужен вопрос Да/Нет.\n"
        "- narrow_choice: сущность уже понятна, но нужно сузить по филиалу/дате/времени/варианту.\n"
        "- missing_auth_data: не хватает персональных идентификаторов для patient-specific запроса.\n"
        "- other: только если тип выше не подходит.\n"
        "Если выбран confirm_candidate, вопрос должен быть коротким и бинарным.\n"
        "Не выдумывай конкретные фамилии/услуги, если их нет в сообщении/контексте.\n"
        "Для intent=appointment user-facing слоты: appointment_action, doctor_name/specialty, branch_or_city, date, time, patient_name.\n"
        "В missing_slots используй только: appointment_action, doctor_name, specialty, service_or_analysis_name, "
        "branch_or_city, date, time, patient_name, result_surname, result_year_of_birth, "
        "result_analysis_code, result_analysis_number. Не используй legacy/backend names.\n"
        "Верни ТОЛЬКО JSON без markdown.\n\n"
        "Summary:\n"
        f"{summary_text}\n\n"
        "Последние реплики:\n"
        f"{history or '(нет)'}\n\n"
        f"Pending intent: {pending_intent_text}\n"
        f"Pending slots: {pending_slots_text}\n"
        f"Remembered doctor: {doctor_hint}\n\n"
        f"Current dialog phase: {state_phase}\n"
        f"Current clarify type: {state_clarify_type}\n"
        f"Current confirmation target: {state_confirmation_target}\n"
        f"Current open question: {state_open_question}\n\n"
        "Current dialog entities:\n"
        f"{json.dumps(state_entities, ensure_ascii=False, indent=2) if state_entities else '(нет)'}\n\n"
        "Candidate entities:\n"
        f"{json.dumps(state_candidates, ensure_ascii=False, indent=2) if state_candidates else '(нет)'}\n\n"
        f"Пользователь: {user_message}\n\n"
        "JSON schema example:\n"
        f"{schema_text}\n"
    )


def build_post_tool_verifier_prompt(
    *,
    user_message: str,
    intent: str,
    tool_name: str,
    drafted_answer: str,
    tool_payload: dict[str, Any],
) -> str:
    payload_text = json.dumps(tool_payload, ensure_ascii=False, default=str)
    if len(payload_text) > 8000:
        payload_text = payload_text[:8000] + "...(truncated)"
    answer_text = str(drafted_answer or "").strip()
    if len(answer_text) > 1200:
        answer_text = answer_text[:1200] + "...(truncated)"
    schema = {
        "enough_data": True,
        "should_clarify": False,
        "clarify_type": "identify|confirm_candidate|narrow_choice|missing_auth_data|other",
        "clarify_question": "",
        "answer_policy": "direct|clarify|not_found",
    }
    schema_text = json.dumps(schema, ensure_ascii=False, indent=2)
    return (
        "Ты post-tool verifier для медицинского ассистента клиники.\n"
        "Задача: после tool_call определить, можно ли давать финальный ответ.\n"
        "Не выдумывай факты. Верни ТОЛЬКО JSON без markdown.\n\n"
        "Правила:\n"
        "1) direct: данных достаточно для прямого ответа.\n"
        "2) clarify: данных недостаточно, но можно задать один уточняющий вопрос.\n"
        "3) not_found: данных недостаточно и уточнение не поможет.\n\n"
        "Если answer_policy=clarify, обязательно укажи clarify_type.\n"
        "clarify_type=identify — не хватает базовой сущности или названия.\n"
        "clarify_type=confirm_candidate — есть один вероятный кандидат и нужен Да/Нет.\n"
        "clarify_type=narrow_choice — нужно сузить по филиалу, дате, времени или варианту услуги.\n"
        "clarify_type=missing_auth_data — не хватает персональных идентификаторов для результата/пациента.\n\n"
        f"Intent: {intent}\n"
        f"Tool: {tool_name}\n"
        f"User message: {user_message}\n\n"
        f"Drafted answer: {answer_text or '(empty)'}\n\n"
        f"Tool payload: {payload_text}\n\n"
        "JSON schema example:\n"
        f"{schema_text}\n"
    )


def build_interrupt_arbiter_prompt(
    *,
    user_message: str,
    dialog_state: dict[str, Any] | None,
) -> str:
    state = dialog_state if isinstance(dialog_state, dict) else {}
    entities = state.get("entities") if isinstance(state.get("entities"), dict) else {}
    expected_slots = state.get("expected_slots") if isinstance(state.get("expected_slots"), list) else []
    compact_entities: dict[str, Any] = {}
    for key in (
        "doctor_name",
        "specialty",
        "service_name",
        "test_name",
        "branch_name",
        "city",
        "date",
        "time",
        "patient_name",
        "result_surname",
        "result_year_of_birth",
        "result_analysis_code",
        "result_analysis_number",
    ):
        value = str(entities.get(key) or "").strip()
        if value:
            compact_entities[key] = value
    schema = {
        "decision": "continue|correct|switch|interrupt|hard_reset|unknown",
        "reason": "",
    }
    return (
        "Ты interrupt/topic-switch arbiter для клинического диалога.\n"
        "Твоя задача: классифицировать неоднозначную реплику пользователя при уже активном flow.\n"
        "Нельзя менять state, очищать память или придумывать новый вопрос.\n"
        "Верни только JSON без markdown.\n\n"
        "Классы решения:\n"
        "- continue: это продолжение текущего flow.\n"
        "- correct: это корректировка текущего slot/entity внутри того же flow.\n"
        "- switch: это новый вопрос, нужен confirm на переход к новой теме.\n"
        "- interrupt: пользователь хочет остановить текущий сценарий.\n"
        "- hard_reset: пользователь хочет очистить весь диалог.\n"
        "- unknown: решение неясно.\n\n"
        "Текущий flow:\n"
        f"- flow_kind: {str(state.get('flow_kind') or '').strip() or '(нет)'}\n"
        f"- flow_stage: {str(state.get('flow_stage') or '').strip() or '(нет)'}\n"
        f"- open_question: {str(state.get('open_question') or '').strip() or '(нет)'}\n"
        f"- expected_slots: {json.dumps(expected_slots, ensure_ascii=False)}\n"
        f"- entities: {json.dumps(compact_entities, ensure_ascii=False)}\n\n"
        f"Пользователь: {user_message}\n\n"
        "JSON schema example:\n"
        f"{json.dumps(schema, ensure_ascii=False, indent=2)}\n"
    )
