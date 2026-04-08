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
