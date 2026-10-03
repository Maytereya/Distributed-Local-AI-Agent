"""LLM выбирает памятку подготовки, ПРЕДМЕТ которой — анализ или процедура пациента.

Класс дефектов «упоминание принято за предмет» (журнал, BUG-2026-10-01-PREPARE-MENTION-AS-SUBJECT):
оценка по совпадению слов отдавала «подготовке к анализу на сахар» памятку ФКС со
слабительным (в её списке еды — «Желе, сахар, мед»), колоноскопии — памятку анализа кала
«Колонофлор-16», ФГДС — «профиль анализов перед эндоскопией». Стенд 02.10: из 32 вопросов
верную памятку получали 14.

Источники (решения владельца 02.10):
  - памятки МИС — поле «Подготовка» в карточке услуги, тексты для пациентов;
  - пока клиника не перенесёт их в МИС, — памятки ПАЦИЕНТУ из базы знаний, название
    которых начинается со слова «ПАМЯТКА» (`kb_patient_memos`). Остальные документы базы
    знаний — скрипты администраторов («куда записывать, что говорить»); в подготовку они
    не попадают, пациент не должен их видеть.

LLM получает пронумерованные памятки (название + начало текста) и возвращает номер:
выдумать памятку она не может. Сбой, таймаут, мусор, режим strict, выключенный флаг —
`_choose_without_llm`: без LLM отличить предмет от упоминания нечем, поэтому только
карточки МИС, названные предметом вопроса.
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from dataclasses import dataclass
from typing import Any, Sequence

from agent_logic_1 import meilisearch_client as meilisearch
from converters import html_cleaner

from ..llm_runtime import generate_text
from ..prompt_registry import load_prompt_text
from . import _common as _common_mod
from ._prepare import _prepare_names_subject, _prepare_subject_phrase, _prepare_term_roots

logger = logging.getLogger(__name__)

SOURCE_MIS = "serviceInfoAll"
SOURCE_KB = "main_index"

_EXCERPT_CHARS = 300
_LLM_TIMEOUT_S = 12
_LLM_QUEUE_TIMEOUT_MS = 3000
# Больше двух памяток в одном ответе — уже не выбор, а куча: и LLM (практически
# одинаковые памятки, решение владельца 03.10), и выбор без LLM (ничья) — не больше двух.
_MAX_MEMOS = 2
# Сколько текста каждой памятки показывать LLM при сравнении подготовки.
_SAME_TEXT_CHARS = 2500

# Переходное правило (решение владельца 02.10): памятки пациенту в базе знаний названы
# «ПАМЯТКА …». Когда клиника перенесёт их в МИС, базу знаний из подготовки убрать
# целиком — `kb_patient_memos` возвращает пусто.
_KB_PATIENT_MEMO_TITLE_RE = re.compile(r"^\s*памятка\b", re.I)
_KB_CACHE_TTL_S = 600
_kb_cache: dict[str, Any] = {"at": 0.0, "memos": ()}


@dataclass(frozen=True)
class PrepareMemo:
    """Памятка-кандидат: откуда, к какой услуге (название) и текст для пациента.

    `score` — оценка правил (у памяток базы знаний 0); `named` — название карточки МИС
    называет предмет вопроса: в нём все слова предмета, или на неё указывает синоним
    клиники. Оба поля нужны только выбору без LLM.
    """

    source: str
    title: str
    text: str
    score: float = 0.0
    named: bool = False


def kb_patient_memos() -> tuple[PrepareMemo, ...]:
    """Памятки пациенту из базы знаний (переходный период), с кэшем на 10 минут.

    Сбой базы знаний — прошлый кэш или пусто: подготовка из МИС работает и без неё.
    """

    now = time.monotonic()
    if _kb_cache["memos"] and now - float(_kb_cache["at"]) < _KB_CACHE_TTL_S:
        return _kb_cache["memos"]
    try:
        client = meilisearch.get_meilisearch_client()
        meilisearch.ensure_default_indexes_once()
        docs = client.index("main_index").get_documents({"limit": 500, "fields": ["title", "content"]}).results
    except Exception as exc:  # база знаний недоступна — без её памяток
        logger.warning("prepare_kb_memos_unavailable: %s", type(exc).__name__)
        return _kb_cache["memos"]
    memos = []
    for doc in docs:
        doc = dict(doc) if not isinstance(doc, dict) else doc
        title = str(doc.get("title") or "").strip()
        text = html_cleaner.strip_html(str(doc.get("content") or "")).strip()
        if text and _KB_PATIENT_MEMO_TITLE_RE.search(title):
            memos.append(PrepareMemo(source=SOURCE_KB, title=title, text=text))
    _kb_cache.update(at=now, memos=tuple(memos))
    return _kb_cache["memos"]


def _excerpt(text: str) -> str:
    flat = re.sub(r"\s+", " ", str(text or "")).strip()
    return flat if len(flat) <= _EXCERPT_CHARS else flat[:_EXCERPT_CHARS].rstrip() + "…"


def _build_prompt(question: str, memos: Sequence[PrepareMemo]) -> str:
    rows = "\n".join(f"{i}. {memo.title} — {_excerpt(memo.text)}" for i, memo in enumerate(memos, 1))
    return (
        load_prompt_text("prepare_document_select")
        .replace("<<QUESTION>>", str(question or "").strip())
        .replace("<<DOCUMENTS>>", rows)
    ).strip()


def _parse_choice(raw: str, count: int) -> tuple[int, ...] | None:
    """Номера памяток (1..count): один или два; пусто — LLM сказала «подходящей нет».

    :return: кортеж номеров; None — ответ не годится (не JSON, номер вне списка, больше двух)
    """

    try:
        data = json.loads(str(raw or "").strip())
    except (TypeError, ValueError):
        return None
    if not isinstance(data, dict) or "match" not in data:
        return None
    choice = data.get("match")
    if choice is None:
        return ()
    items = choice if isinstance(choice, list) else [choice]
    numbers: list[int] = []
    for item in items:
        if isinstance(item, bool):
            return None
        try:
            num = int(item)
        except (TypeError, ValueError):
            return None
        if not 1 <= num <= count:
            return None
        if num not in numbers:
            numbers.append(num)
    return tuple(numbers) if 1 <= len(numbers) <= _MAX_MEMOS else None


def _choose_without_llm(memos: Sequence[PrepareMemo]) -> tuple[PrepareMemo, ...]:
    """Какие памятки отдать, когда LLM недоступна или выключена.

    Только карточки МИС, названные предметом вопроса: одна карточка — одна услуга, и
    название ставит клиника. Памятки базы знаний без LLM не берём — их названия
    объединяют процедуры, и на «ФГДС» ушла бы «ФКС + ФГДС» со слабительным.

    Из названных — лучшая по оценке правил. Ничья на первом месте — обе: решение
    владельца 03.10 на тестовый период, администраторы скажут, какую выдавать («ТТГ
    (TSH)» и «Антитела к рецепторам ТТГ» набирают поровну). Больше двух — пусто: честное
    «памятки нет» с оффером оператора лучше кучи чужих памяток.

    :param memos: кандидаты в порядке правил
    :return: ноль, одна или две памятки
    """

    named = [memo for memo in memos if memo.source == SOURCE_MIS and memo.named]
    if not named:
        return ()
    best = max(memo.score for memo in named)
    top = tuple(memo for memo in named if math.isclose(memo.score, best, abs_tol=1e-6))
    return top if len(top) <= _MAX_MEMOS else ()


def _twin_candidate(question: str, chosen: PrepareMemo, memos: Sequence[PrepareMemo]) -> PrepareMemo | None:
    """Ещё одна карточка МИС, в НАЗВАНИИ которой — слова пациента.

    На «подготовка к ттг» LLM выбирает «ТТГ (TSH) тиреотропный гормон»; «Антитела к
    рецепторам ТТГ» — тоже ТТГ, и памятка у неё по сути та же. Синоним клиники сюда не
    засчитывается: «холестерин» у клиники указывает и на «Триглицериды».

    :return: лучшая по оценке правил такая карточка или None
    """

    if chosen.source != SOURCE_MIS:
        return None
    subject_roots = _prepare_term_roots(_prepare_subject_phrase(question))
    twins = [
        memo
        for memo in memos
        if memo is not chosen
        and memo.source == SOURCE_MIS
        and _prepare_names_subject(subject_roots, _prepare_term_roots(memo.title))
    ]
    return max(twins, key=lambda memo: memo.score) if twins else None


async def _same_preparation(first: PrepareMemo, second: PrepareMemo) -> bool:
    """LLM: подготовка в двух памятках практически одинаковая? Сбой — «нет» (одна памятка)."""

    prompt = (
        load_prompt_text("prepare_same_memo")
        .replace("<<TITLE_1>>", first.title)
        .replace("<<TEXT_1>>", first.text[:_SAME_TEXT_CHARS])
        .replace("<<TITLE_2>>", second.title)
        .replace("<<TEXT_2>>", second.text[:_SAME_TEXT_CHARS])
    ).strip()
    try:
        raw = await generate_text(
            prompt,
            timeout_s=_LLM_TIMEOUT_S,
            queue_timeout_ms=_LLM_QUEUE_TIMEOUT_MS,
            fmt="json",
            think=False,
            options={"temperature": 0},
        )
        data = json.loads(str(raw or "").strip())
    except Exception as exc:  # без ответа LLM вторую памятку не добавляем
        logger.warning("prepare_same_memo_failed: %s", type(exc).__name__)
        return False
    return isinstance(data, dict) and data.get("same") is True


async def select_prepare_memos(
    question: str,
    memos: Sequence[PrepareMemo],
    *,
    runtime_llm_mode: str = "",
) -> tuple[PrepareMemo, ...]:
    """Памятки, предмет которых — анализ или процедура из вопроса; пусто — таких нет.

    LLM выбирает одну; если ещё одна карточка МИС названа словами пациента и подготовка в
    ней практически та же (`_same_preparation`) — обе. Без LLM — `_choose_without_llm`:
    одна или две при ничьей.

    :param question: вопрос пациента
    :param memos: кандидаты (`PrepareMemo`)
    :param runtime_llm_mode: strict|hybrid|rich; в strict LLM не зовём
    :return: выбранные памятки
    """

    memos = [memo for memo in memos if isinstance(memo, PrepareMemo) and memo.text.strip()]
    if not memos:
        return ()
    enabled = _common_mod._runtime_bool("MR_PREPARE_RELEVANCE_LLM_ENABLED", True)
    if not enabled or str(runtime_llm_mode or "").strip().lower() == "strict":
        return _choose_without_llm(memos)
    try:
        raw = await generate_text(
            _build_prompt(question, memos),
            timeout_s=_LLM_TIMEOUT_S,
            queue_timeout_ms=_LLM_QUEUE_TIMEOUT_MS,
            fmt="json",
            think=False,
            options={"temperature": 0},
        )
    except Exception as exc:  # LLM недоступна — осторожная политика без неё
        logger.warning("prepare_memo_select_failed: %s", type(exc).__name__)
        return _choose_without_llm(memos)
    choice = _parse_choice(raw, len(memos))
    if choice is None:
        logger.warning("prepare_memo_select_bad_reply: %r", str(raw or "")[:200])
        return _choose_without_llm(memos)
    chosen = tuple(memos[num - 1] for num in choice)
    if len(chosen) == 1:
        # Решение владельца 03.10: практически одинаковые памятки — обе, пациент выберет.
        # Отдельный короткий вопрос: в одном промпте с выбором правило модель не держала.
        twin = _twin_candidate(question, chosen[0], memos)
        if twin is not None and await _same_preparation(chosen[0], twin):
            return (chosen[0], twin)
    return chosen
