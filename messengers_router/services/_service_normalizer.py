"""П5 (вариант B): LLM-нормализация пациентской формулировки услуги к термину каталога.

Fallback-only слой поверх лексического матчера (`match_catalog_service`):
зовётся ТОЛЬКО когда exact-resolver и difflib-fuzzy уже промахнулись, и только
для service-похожих текстов (см. `should_attempt_llm_normalization`). Выход LLM
НИКОГДА не используется напрямую — вызывающая сторона обязана верифицировать его
через `resolve_price_service_name_from_catalog` (LLM не может выдумать услугу;
непроверяемый выход = прежний честный miss). Любой сбой/таймаут → None (fail-open,
поведение идентично сегодняшнему). Кэш нормализаций — LLM-вызов на уникальную
формулировку платится один раз за процесс.

Дизайн-инварианты (закреплены tests/test_service_llm_normalizer.py):
  1) рабочие exact/fuzzy запросы LLM не трогают (горячий путь не замедляется);
  2) не-услуги (запись/парковка) LLM не трогают — иначе спекулятивный
     catalog-prefetch (router Part IV Stage 15) замедлил бы обычные ходы;
  3) сбой/мусор/NONE → None; 4) эхо входа → None (лексика по нему уже промахнулась).
"""

from __future__ import annotations

import logging
from typing import Any

from ..llm_runtime import generate_text
from ..prompt_registry import load_prompt_text
from ..runtime_config import config as _cfg
from ..russian_nlu import normalize_ru
from ..service_phrase import extract_service_phrase
from ._prices_helpers import _PRICE_REQUEST_RE

logger = logging.getLogger(__name__)

# Kill-switch: [MESSENGER_ROUTER] llm_service_normalization = false в config.ini
# (правится руками на хосте, рестарт контейнера; по умолчанию ВКЛ — слой fail-open
# и срабатывает только на промахах, где сегодня пациент получает бесполезный clarify).
# Host-конфиг ТОЛЬКО через runtime_config-порт (правило arch-гардрейла проекта).
_ENABLED: bool = bool(getattr(_cfg, "MR_LLM_SERVICE_NORMALIZATION", True))

_CACHE: dict[str, str | None] = {}
_CACHE_CAP = 512

_MIN_QUERY_LEN = 3
_MAX_QUERY_LEN = 120
_MAX_ANSWER_LEN = 80

_LLM_TIMEOUT_S = 15
# Короткий queue-timeout: если LLM занята другими ходами — не держим пациента,
# честный miss (как сегодня) быстрее, чем ожидание слота ради nice-to-have.
_LLM_QUEUE_TIMEOUT_MS = 2000


def should_attempt_llm_normalization(raw_text: str) -> bool:
    """Гейт «текст похож на запрос услуги»: фраза услуги ИЛИ price-intent.

    Держит LLM подальше от обычных ходов записи/смолтока, на которых
    match_catalog_service промахивается по определению («запишите на завтра»,
    «у вас есть парковка?»). NB: голое «сколько стоит X» может не давать
    extract_service_phrase (прайс-обёртка) — поэтому ИЛИ по price-regex.
    """

    raw = str(raw_text or "").strip()
    if not (_MIN_QUERY_LEN <= len(raw) <= _MAX_QUERY_LEN):
        return False
    if extract_service_phrase(raw):
        return True
    return bool(_PRICE_REQUEST_RE.search(raw))


def _sanitize_llm_answer(raw_answer: str, query: str) -> str | None:
    """Первая строка, без кавычек/маркеров; отбрасывает NONE/пустое/эхо/оверлонг."""

    line = str(raw_answer or "").strip().splitlines()[0] if str(raw_answer or "").strip() else ""
    line = line.strip().strip('"«»\'`').strip(" .,:;!-–—").strip()
    if not line or len(line) > _MAX_ANSWER_LEN:
        return None
    if line.lower() == "none":
        return None
    if normalize_ru(line) == normalize_ru(query):
        return None
    return line


async def llm_normalize_service_query(query: str) -> str | None:
    """Переписывает формулировку пациента в предполагаемый термин каталога.

    :param query: сырой текст пациента (или его услуга-фрагмент)
    :return: кандидат-термин (ЕЩЁ НЕ верифицированный по каталогу!) или None
    """

    if not _ENABLED:
        return None
    q = str(query or "").strip()
    if not (_MIN_QUERY_LEN <= len(q) <= _MAX_QUERY_LEN):
        return None

    cache_key = normalize_ru(q)
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    try:
        template = load_prompt_text("service_normalizer")
        prompt = template.replace("<<QUERY>>", q)
        raw = await generate_text(
            prompt,
            timeout_s=_LLM_TIMEOUT_S,
            queue_timeout_ms=_LLM_QUEUE_TIMEOUT_MS,
        )
        answer = _sanitize_llm_answer(raw, q)
    except Exception as exc:  # fail-open: любой сбой = прежний честный miss
        logger.warning("service_normalizer_failed: %s", type(exc).__name__)
        answer = None

    if len(_CACHE) >= _CACHE_CAP:
        _CACHE.clear()
    _CACHE[cache_key] = answer
    return answer


def debug_state() -> dict[str, Any]:
    """Для диагностики/тестов: размер кэша и включённость."""

    return {"enabled": _ENABLED, "cache_size": len(_CACHE)}
