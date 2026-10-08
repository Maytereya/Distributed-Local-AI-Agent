"""LLM-различитель: «отменить / перенести» без слова «запись» — просьба о своей записи или нет.

Правило записи срабатывает на объекте записи («отменить запись», «перенести приём»). Голый
глагол — только кандидат: «хочу отменить, заболела» и «не смогу прийти, отмените» — отмена
записи, а «нужно ли отменять лекарства перед анализом» и «я перенесла ковид» — вопросы о
подготовке и анализах. По буквам их не различить, а классификатор NLU первые относит к
«прочему» (свип 04.10, BUG-2026-10-04-APPOINTMENT-VERB-LOOKALIKE).

Сбой (kill-switch выключен, тайм-аут, невнятный ответ) → None: вызывающий ведёт себя как
до 04.10 и считает фразу просьбой о записи — отмену увидит оператор, а «не понял» на
настоящую отмену пациент не получит.

Kill-switch: [MESSENGER_ROUTER] llm_appointment_change_validation = false (config.ini,
дефолт ВКЛ).
"""

from __future__ import annotations

import logging

from ..llm_runtime import DECISION_OPTIONS, generate_text
from ..prompt_registry import load_prompt_text
from ..runtime_config import config as _cfg

logger = logging.getLogger(__name__)

_CACHE: dict[str, bool] = {}
_CACHE_CAP = 512

_MAX_LEN = 300

_LLM_TIMEOUT_S = 10
_LLM_QUEUE_TIMEOUT_MS = 2000

_VERDICTS = {"ЗАПИСЬ": True, "ДРУГОЕ": False}


def _norm_key(text: str) -> str:
    return " ".join(str(text or "").lower().split())


async def is_own_appointment_change(text: str) -> bool | None:
    """Просит ли пациент отменить или перенести свою запись.

    :param text: реплика с глаголом отмены/переноса без объекта записи
    :return: True — да, False — пишет о другом, None — LLM не ответила внятно
    """

    if not bool(getattr(_cfg, "MR_LLM_APPOINTMENT_CHANGE_VALIDATION", True)):
        return None
    raw = str(text or "").strip()
    if not raw or len(raw) > _MAX_LEN:
        return None

    key = _norm_key(raw)
    if key in _CACHE:
        return _CACHE[key]

    try:
        prompt = load_prompt_text("appointment_change_validator").replace("<<TEXT>>", raw)
        answer = await generate_text(
            prompt, timeout_s=_LLM_TIMEOUT_S, queue_timeout_ms=_LLM_QUEUE_TIMEOUT_MS, think=False, options=DECISION_OPTIONS
        )
    except Exception as exc:  # сбой LLM — решает вызывающий, как до 04.10
        logger.warning("appointment_change_validator_failed: %s", type(exc).__name__)
        return None

    lines = str(answer or "").strip().splitlines()
    verdict = _VERDICTS.get(lines[0].strip().strip('"«»\'`.,:;!').upper() if lines else "")
    if verdict is None:
        logger.warning("appointment_change_validator_unclear: %r", str(answer or "")[:40])
        return None
    if len(_CACHE) >= _CACHE_CAP:
        _CACHE.clear()
    _CACHE[key] = verdict
    return verdict
