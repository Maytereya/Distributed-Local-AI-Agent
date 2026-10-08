"""Ход без данных: LLM выбирает ВИД реплики, а не пишет ответ (L-05, решение владельца 08.10).

Свободный текст LLM на пустых данных — класс 4 журнала («данных нет → ответ не по существу»):
«клиника не занимается больничными» (оператор: «открывает доктор на приёме»), «не предоставляем
выезд» при 38 строках прайса. Запрет в промпте рендера нарушался (06.07 → 23.09). На трафике
08.10 класс 4 — 28% провальных диалогов с 03.09.

Теперь на ходу без данных модель только относит реплику к одному из видов (пункт 2 принципов —
выбор из закрытого списка), а текст печатает код: благодарность, прощание, вопрос о боте,
«ок / понятно» — короткий шаблон; бессмыслица — «уточните»; вопрос о клинике — честный оффер
оператора (пункт 3). Сбой (kill-switch, тайм-аут, невнятный ответ) → None: вызывающий отвечает
нейтральной фразой без утверждений.

Kill-switch: [MESSENGER_ROUTER] llm_empty_evidence_reply = false — вернуть свободный текст
(откат L-05).
"""

from __future__ import annotations

import logging

from ..llm_runtime import generate_text
from ..prompt_registry import load_prompt_text
from ..runtime_config import config as _cfg

logger = logging.getLogger(__name__)

KINDS = ("THANKS", "BYE", "ABOUT_BOT", "ACK", "NOISE", "QUESTION")

_CACHE: dict[str, str] = {}
_CACHE_CAP = 512
_MAX_LEN = 600
_LLM_TIMEOUT_S = 10
_LLM_QUEUE_TIMEOUT_MS = 2000


def enabled() -> bool:
    return bool(getattr(_cfg, "MR_LLM_EMPTY_EVIDENCE_REPLY", True))


def _norm_key(text: str) -> str:
    return " ".join(str(text or "").lower().split())


async def classify_empty_evidence_turn(text: str) -> str | None:
    """Вид реплики, на которую у бота нет данных.

    :param text: реплика пациента
    :return: один из `KINDS` или None, если LLM не ответила внятно
    """

    raw = str(text or "").strip()[:_MAX_LEN]
    if not raw:
        return "NOISE"
    key = _norm_key(raw)
    if key in _CACHE:
        return _CACHE[key]
    try:
        prompt = load_prompt_text("empty_evidence_kind").replace("<<TEXT>>", raw)
        answer = await generate_text(prompt, timeout_s=_LLM_TIMEOUT_S, queue_timeout_ms=_LLM_QUEUE_TIMEOUT_MS, think=False)
    except Exception as exc:  # сбой LLM — нейтральный ответ у вызывающего
        logger.warning("empty_evidence_kind_failed: %s", type(exc).__name__)
        return None
    lines = str(answer or "").strip().splitlines()
    kind = lines[0].strip().strip('"«»\'`.,:;!').upper() if lines else ""
    if kind not in KINDS:
        logger.warning("empty_evidence_kind_unclear: %r", str(answer or "")[:40])
        return None
    if len(_CACHE) >= _CACHE_CAP:
        _CACHE.clear()
    _CACHE[key] = kind
    return kind
