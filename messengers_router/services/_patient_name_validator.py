"""LLM-валидатор слота ФИО: имя пациента vs команда/коррекция.

Прод #668: бот спросил ФИО, пациент написал «Время изменени» (хотел сменить
время) → форма `_looks_like_patient_fio` приняла как ФИО (2+ кириллических
слова, не стоп-лист) → «Запись: Время Изменени…». Форма семантику не различает,
а стоп-лист «время/изменить/…» — бесконечная гонка (решение владельца: LLM-first).

Валидатор ТОЛЬКО отвергает уверенное «OTHER» от LLM. Fail-open по построению:
kill-switch off / таймаут / сбой / мусор → True (принять по форме, текущее
поведение). Запись НИКОГДА не блокируется валидатором. Вызывается лишь на ходе
сбора ФИО (не hot-path общего трафика).

Kill-switch: [MESSENGER_ROUTER] llm_patient_name_validation = false (config.ini,
правится руками на хосте, рестарт; дефолт ВКЛ). Host-конфиг ТОЛЬКО через
runtime_config-порт (arch-гардрейл проекта).
"""

from __future__ import annotations

import logging
from typing import Any

from ..llm_runtime import generate_text
from ..prompt_registry import load_prompt_text
from ..runtime_config import config as _cfg

logger = logging.getLogger(__name__)

_ENABLED: bool = bool(getattr(_cfg, "MR_LLM_PATIENT_NAME_VALIDATION", True))

_CACHE: dict[str, bool] = {}
_CACHE_CAP = 512

_MIN_LEN = 2
_MAX_LEN = 80

_LLM_TIMEOUT_S = 10
# Короткий queue-timeout: LLM занята другими ходами → не держим пациента,
# принимаем по форме (fail-open) быстрее, чем ждать слот ради валидации.
_LLM_QUEUE_TIMEOUT_MS = 2000


def _norm_key(text: str) -> str:
    return " ".join(str(text or "").lower().split())


async def is_patient_name_reply(text: str) -> bool:
    """True, если реплика — ФИО пациента (а не команда/коррекция/вопрос).

    Отвергает (False) ТОЛЬКО при уверенном «OTHER» от LLM. Любой иной исход
    (kill-switch off, таймаут, сбой, неоднозначный ответ) → True (fail-open:
    принять по форме, не блокировать запись).

    :param text: реплика пользователя на шаге сбора ФИО (уже прошла форму)
    :return: принять как ФИО (True) или отвергнуть и переспросить (False)
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
        template = load_prompt_text("patient_name_validator")
        prompt = template.replace("<<TEXT>>", raw)
        answer = await generate_text(
            prompt,
            timeout_s=_LLM_TIMEOUT_S,
            queue_timeout_ms=_LLM_QUEUE_TIMEOUT_MS,
        )
        # Отвергаем ТОЛЬКО на явном «OTHER». Любой иной ответ (FIO/мусор/пусто)
        # → принять (fail-open): валидатор не должен зарезать валидное имя.
        first = str(answer or "").strip().splitlines()[0].strip().strip('"«»\'`.,:;!').upper() if str(answer or "").strip() else ""
        if first == "OTHER":
            verdict = False
    except Exception as exc:  # fail-open: любой сбой = принять по форме
        logger.warning("patient_name_validator_failed: %s", type(exc).__name__)
        verdict = True

    if len(_CACHE) >= _CACHE_CAP:
        _CACHE.clear()
    _CACHE[key] = verdict
    return verdict


def debug_state() -> dict[str, Any]:
    return {"enabled": _ENABLED, "cache_size": len(_CACHE)}
