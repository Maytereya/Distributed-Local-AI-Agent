"""LLM-валидатор слота услуги: название услуги vs команда/навигация/коррекция.

Аудит 2026-07-22 (класс `slot_accepts_command_as_data`): бот в активной записи
спросил услугу, пациент ответил командой — «Стоп»/«Поменяй»/«Заново»/«Другое» →
форма `_SERVICE_SINGLE_WORD_RE` (любое слово ≥4 букв не из крошечного стоп-листа)
приняла как услугу → «Запись: Иванов Иван, Поменяй, Самара, завтра, 10:00.
Подтверждаете?». Форма семантику не различает, а стоп-лист — бесконечная гонка
(та же, что владелец отверг для ФИО → LLM-first).

Валидатор ТОЛЬКО отвергает уверенное «OTHER» от LLM (команда/навигация). Fail-open
по построению: kill-switch off / таймаут / сбой / мусор → True (принять по форме,
текущее поведение). Запись НИКОГДА не блокируется валидатором. Вызывается лишь на
приёме НОВОГО значения услуги в активной записи (не hot-path общего трафика) и
только для коротких (командо-подобных) реплик — многословные услуги («УЗИ брюшной
полости») в валидатор не идут вовсе.

Kill-switch: [MESSENGER_ROUTER] llm_service_slot_validation = false (config.ini,
правится руками на хосте, рестарт; дефолт ВКЛ; тот же паттерн, что у ФИО-валидатора).
"""

from __future__ import annotations

import logging
from typing import Any

from ..llm_runtime import generate_text
from ..prompt_registry import load_prompt_text
from ..runtime_config import config as _cfg

logger = logging.getLogger(__name__)

_ENABLED: bool = bool(getattr(_cfg, "MR_LLM_SERVICE_SLOT_VALIDATION", True))

_CACHE: dict[str, bool] = {}
_CACHE_CAP = 512

_MIN_LEN = 2
_MAX_LEN = 60

_LLM_TIMEOUT_S = 10
# Короткий queue-timeout: LLM занята другими ходами → не держим пациента,
# принимаем по форме (fail-open) быстрее, чем ждать слот ради валидации.
_LLM_QUEUE_TIMEOUT_MS = 2000


def _norm_key(text: str) -> str:
    return " ".join(str(text or "").lower().split())


async def is_service_name_reply(text: str) -> bool:
    """True, если реплика — название услуги (а не команда/навигация/коррекция).

    Отвергает (False) ТОЛЬКО при уверенном «OTHER» от LLM. Любой иной исход
    (kill-switch off, таймаут, сбой, неоднозначный ответ) → True (fail-open:
    принять по форме, не блокировать запись).

    :param text: реплика пользователя, попавшая в слот услуги (уже прошла форму)
    :return: принять как услугу (True) или отвергнуть и переспросить (False)
    """

    if not _ENABLED:
        return True
    raw = str(text or "").strip()
    if not (_MIN_LEN <= len(raw) <= _MAX_LEN):
        return True  # вне диапазона — не наш случай, принять по форме

    key = _norm_key(raw)
    if key in _CACHE:
        return _CACHE[key]

    verdict = True  # fail-open дефолт
    try:
        template = load_prompt_text("service_slot_validator")
        prompt = template.replace("<<TEXT>>", raw)
        answer = await generate_text(
            prompt,
            timeout_s=_LLM_TIMEOUT_S,
            queue_timeout_ms=_LLM_QUEUE_TIMEOUT_MS,
        )
        # Отвергаем ТОЛЬКО на явном «OTHER». Любой иной ответ (SERVICE/мусор/пусто)
        # → принять (fail-open): валидатор не должен зарезать валидную услугу.
        first = (
            str(answer or "").strip().splitlines()[0].strip().strip('"«»\'`.,:;!').upper()
            if str(answer or "").strip()
            else ""
        )
        if first == "OTHER":
            verdict = False
    except Exception as exc:  # fail-open: любой сбой = принять по форме
        logger.warning("service_slot_validator_failed: %s", type(exc).__name__)
        verdict = True

    if len(_CACHE) >= _CACHE_CAP:
        _CACHE.clear()
    _CACHE[key] = verdict
    return verdict


def debug_state() -> dict[str, Any]:
    return {"enabled": _ENABLED, "cache_size": len(_CACHE)}
