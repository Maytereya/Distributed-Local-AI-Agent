"""LLM-различитель: справочный вопрос о СРОКАХ результата vs запрос своего результата.

Прод #675: «Результат флюорографии отдают сразу?» / «Через сколько готов
результат?» → бот просил фамилию/год/код (result-lookup), хотя пациент спрашивал
про СРОКИ готовности. Наблюдение владельца: 95% «результат» = свои результаты;
справка о сроках редка, её даёт оператор (сроки достоверно не знаем — выдумывать
запрещено, принцип «не дезинформировать»).

Fail-safe В СТОРОНУ LOOKUP: возвращает True (это про сроки → оператор) ТОЛЬКО на
уверенном «TIMING» от LLM. Любой иной исход (kill-switch off / таймаут / сбой /
неоднозначный ответ / «LOOKUP») → False = дефолт lookup (свои результаты, 95%).
Зовётся лишь когда TEST_RESULT без данных пациента (не hot-path).

Kill-switch: [MESSENGER_ROUTER] llm_result_timing_validation = false (config.ini,
дефолт ВКЛ). Host-конфиг ТОЛЬКО через runtime_config-порт (arch-гардрейл).
"""

from __future__ import annotations

import logging
from typing import Any

from ..llm_runtime import generate_text
from ..prompt_registry import load_prompt_text
from ..runtime_config import config as _cfg

logger = logging.getLogger(__name__)

_ENABLED: bool = bool(getattr(_cfg, "MR_LLM_RESULT_TIMING_VALIDATION", True))

_CACHE: dict[str, bool] = {}
_CACHE_CAP = 512

_MIN_LEN = 3
_MAX_LEN = 160

_LLM_TIMEOUT_S = 10
_LLM_QUEUE_TIMEOUT_MS = 2000


def _norm_key(text: str) -> str:
    return " ".join(str(text or "").lower().split())


async def is_result_timing_question(text: str) -> bool:
    """True, если реплика — справка о СРОКАХ готовности результата (не lookup).

    Отклоняет в «сроки» (True) ТОЛЬКО на уверенном «TIMING»; иначе → False
    (fail-safe: дефолт lookup, не штрафуем 95% пациентов, которым нужен свой
    результат).

    :param text: реплика пользователя (TEST_RESULT-путь без данных пациента)
    :return: это справка о сроках (True) или запрос своего результата (False)
    """

    if not _ENABLED:
        return False
    raw = str(text or "").strip()
    if not (_MIN_LEN <= len(raw) <= _MAX_LEN):
        return False

    key = _norm_key(raw)
    if key in _CACHE:
        return _CACHE[key]

    verdict = False  # fail-safe дефолт: lookup
    try:
        template = load_prompt_text("result_timing_validator")
        prompt = template.replace("<<TEXT>>", raw)
        answer = await generate_text(
            prompt,
            timeout_s=_LLM_TIMEOUT_S,
            queue_timeout_ms=_LLM_QUEUE_TIMEOUT_MS,
        )
        first = str(answer or "").strip().splitlines()[0].strip().strip('"«»\'`.,:;!').upper() if str(answer or "").strip() else ""
        if first == "TIMING":
            verdict = True
    except Exception as exc:  # fail-safe: любой сбой = дефолт lookup
        logger.warning("result_timing_validator_failed: %s", type(exc).__name__)
        verdict = False

    if len(_CACHE) >= _CACHE_CAP:
        _CACHE.clear()
    _CACHE[key] = verdict
    return verdict


def debug_state() -> dict[str, Any]:
    return {"enabled": _ENABLED, "cache_size": len(_CACHE)}
