"""LLM-различитель: реплика со словом «запись / приём» — о СВОЕЙ оформленной записи или о новой.

Узкий детектор `detect_existing_appointment_request` пропускал формулировки из реальной
переписки: «Мои записи», «Мои приёмы», «Мне подтверждена запись?», «Я к лору записывался на
сегодня». Такие фразы уходили в оформление НОВОЙ записи, в «Пока не понял, к какому врачу»
или в поиск результата анализов (BUG-2026-10-08-EXISTING-APPT-LOOKUP-MISSED). Кандидатов
отбирает правило (`detect_existing_appointment_candidate`), различает короткий вопрос к LLM.

Решение владельца 08.10: оформленную запись бот не ищет и не показывает (ФИО — не
удостоверение личности); вопрос о своей записи → честный оффер оператора.

Вопрос к LLM — один, «да / нет»: «пациент говорит, что УЖЕ записан, и спрашивает об этой
записи?». Три класса в одном вопросе («своя / новая / другое») модель на живом свипе 08.10
не удерживала. Короткие «Запись на УЗИ», «Приём уролога» сюда не доходят — их отсекает
признак уже оформленной записи в правиле-кандидате.

Сбой (kill-switch выключен, тайм-аут, невнятный ответ) → None: вызывающий ведёт себя как до
08.10.

Kill-switch: [MESSENGER_ROUTER] llm_existing_appointment_validation = false (config.ini,
дефолт ВКЛ).
"""

from __future__ import annotations

import logging

from ..llm_runtime import generate_text
from ..prompt_registry import load_prompt_text
from ..runtime_config import config as _cfg

logger = logging.getLogger(__name__)

_CACHE: dict[str, bool] = {}
_CACHE_CAP = 512

_MAX_LEN = 300

_LLM_TIMEOUT_S = 10
_LLM_QUEUE_TIMEOUT_MS = 2000

_VERDICTS = {"ДА": True, "НЕТ": False}


def _norm_key(text: str) -> str:
    return " ".join(str(text or "").lower().split())


async def is_own_existing_appointment(text: str) -> bool | None:
    """Спрашивает ли пациент о своей уже оформленной записи.

    :param text: реплика-кандидат (объект записи без явной просьбы записаться)
    :return: True — о своей записи, False — нет, None — LLM не ответила внятно
    """

    if not bool(getattr(_cfg, "MR_LLM_EXISTING_APPOINTMENT_VALIDATION", True)):
        return None
    raw = str(text or "").strip()
    if not raw or len(raw) > _MAX_LEN:
        return None

    key = _norm_key(raw)
    if key in _CACHE:
        return _CACHE[key]

    try:
        prompt = load_prompt_text("existing_appointment_validator").replace("<<TEXT>>", raw)
        answer = await generate_text(prompt, timeout_s=_LLM_TIMEOUT_S, queue_timeout_ms=_LLM_QUEUE_TIMEOUT_MS, think=False)
    except Exception as exc:  # сбой LLM — решает вызывающий, как до 08.10
        logger.warning("existing_appointment_validator_failed: %s", type(exc).__name__)
        return None

    lines = str(answer or "").strip().splitlines()
    verdict = _VERDICTS.get(lines[0].strip().strip('"«»\'`.,:;!').upper() if lines else "")
    if verdict is None:
        logger.warning("existing_appointment_validator_unclear: %r", str(answer or "")[:40])
        return None
    if len(_CACHE) >= _CACHE_CAP:
        _CACHE.clear()
    _CACHE[key] = verdict
    return verdict
