"""Словари из МИС: синонимы услуг (`serviceSynonyms`) и биоматериалы (`biomatNames`).

## Зачем модуль существует

Прод 10.08: «Сколько стоит мазок на микрофлору с поверхности задней стенки
глотки» → «Скажите название услуги» дважды, затем на «мазок» — список из шести
нерелевантных мазков. Матч ломает именно УПОМИНАНИЕ БИОМАТЕРИАЛА, а не
отсутствие синонима: «мазок на микрофлору» (биоматериал снят) резолвится в
«Посев на микрофлору» безошибочно.

Хуже того, упоминание биоматериала не просто мешает — оно уводит в ЧУЖУЮ
услугу: «сколько стоит мазок на микрофлору с задней стенки глотки» приземлялся
на «Обработка задней стенки глотки расфокусированным лучом СО2 лазера».

## Два поля МИС, две РАЗНЫЕ роли

- ``serviceSynonyms`` — **высшее доверие**: прямое «слово пациента → услуга».
  Клиника заполняет (на срезе 14.08 — 0 из 1613 строк), механика подключена
  заранее и заработает по мере наполнения.
- ``biomatNames`` — **только СНЯТИЕ и ПРОВЕРКА, не идентификация**. Заполнено у
  98% услуг, но один материал подходит многим услугам: «соскоб с задней стенки
  глотки» — к 44 услугам, «кровь из вены» — к 1103. Определять услугу ТОЛЬКО по
  биоматериалу нельзя.

## Механика

1. Снять из запроса хвост-биоматериал (словарь — из `biomatNames` МИС).
2. Сматчить остаток обычным лексическим путём (делает вызывающий).
3. Верифицировать найденную услугу по `biomatNames`: принимает ли она этот
   материал. Только при ПОДТВЕРЖДЕНИИ результат со снятым материалом
   предпочитается исходному.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from agent_logic_2.nayka_api import api_service_info

from ._common import _normalise_input

log = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-zа-яё0-9]+")

# Предлоги, которые сами по себе не являются биоматериалом.
_PREPOSITIONS = frozenset({"с", "со", "из", "от", "на", "в", "по", "у", "и"})

# Локативная обвязка, которой нет в справочнике МИС, но которой пациент
# описывает то же место: «с ПОВЕРХНОСТИ задней стенки глотки». Список
# анатомически-локативный (где брали), НЕ список услуг и не стоп-лист интентов.
_LOCATIVE_FILLER = frozenset(
    {"поверхности", "поверхность", "области", "область", "участка", "участке", "зоны", "зоне"}
)

# Минимальная длина хвоста-биоматериала в токенах. Одно слово («кал», «моча»,
# «нос») — часто само название услуги или её различающий токен, снимать нельзя.
_MIN_TAIL_TOKENS = 2


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(_normalise_input(text))


class _MisVocabularies:
    """Разобранные словари МИС с кэшем по (путь, mtime) дневного среза."""

    def __init__(self) -> None:
        self.biomat_tokens: frozenset[str] = frozenset()
        self.biomat_phrases: tuple[frozenset[str], ...] = ()
        self.synonym_to_service: dict[str, str] = {}
        self.biomat_by_service: dict[str, tuple[frozenset[str], ...]] = {}


_VOCAB_CACHE: dict[tuple[str, float], _MisVocabularies] = {}


def _cache_key() -> tuple[str, float] | None:
    try:
        path: Path = api_service_info.service_info_path()
        stat = path.stat()
    except Exception:
        return None
    return (str(path), stat.st_mtime)


def _build_vocabularies(rows: list[dict[str, Any]]) -> _MisVocabularies:
    vocab = _MisVocabularies()
    biomat_tokens: set[str] = set()
    phrases: set[frozenset[str]] = set()
    for row in rows:
        service_name = str(row.get("serviceName") or "").strip()
        name_key = _normalise_input(service_name)

        raw_biomat = row.get("biomatNames")
        values = raw_biomat if isinstance(raw_biomat, list) else [raw_biomat]
        service_phrases: list[frozenset[str]] = []
        for value in values:
            for part in str(value or "").split(","):
                part_tokens = frozenset(_tokens(part))
                if not part_tokens:
                    continue
                biomat_tokens |= set(part_tokens)
                phrases.add(part_tokens)
                service_phrases.append(part_tokens)
        if name_key and service_phrases:
            vocab.biomat_by_service.setdefault(name_key, tuple(service_phrases))

        raw_syn = row.get("serviceSynonyms")
        syn_values = raw_syn if isinstance(raw_syn, list) else [raw_syn]
        for value in syn_values:
            for part in re.split(r"[,;]", str(value or "")):
                key = _normalise_input(part)
                if key and service_name:
                    vocab.synonym_to_service.setdefault(key, service_name)

    vocab.biomat_tokens = frozenset(biomat_tokens)
    vocab.biomat_phrases = tuple(phrases)
    return vocab


def _vocabularies() -> _MisVocabularies:
    """Словари МИС; парсинг 45-МБ среза выполняется один раз на дневной файл."""

    key = _cache_key()
    if key is None:
        return _MisVocabularies()
    cached = _VOCAB_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        rows = [row for row in api_service_info.load_service_info() if isinstance(row, dict)]
    except Exception as exc:  # каталог МИС недоступен — слой просто молчит
        log.info("service_info недоступен, словари МИС пропущены: %s", exc)
        return _MisVocabularies()
    vocab = _build_vocabularies(rows)
    _VOCAB_CACHE.clear()
    _VOCAB_CACHE[key] = vocab
    return vocab


def mis_synonym_service(query_text: str) -> str | None:
    """Услуга по ТОЧНОМУ синониму из МИС (`serviceSynonyms`) — высшее доверие.

    Клиника заполняет поле; на текущем срезе оно пустое, поэтому функция
    возвращает None и ничего не меняет. Механика подключена заранее.

    :param query_text: реплика/фраза пациента
    :return: каноническое название услуги либо None
    """

    key = _normalise_input(query_text)
    if not key:
        return None
    return _vocabularies().synonym_to_service.get(key)


def split_biomaterial_tail(text: str) -> tuple[str, str]:
    """Снимает с конца запроса фразу-биоматериал.

    Ищем самый длинный ХВОСТ, все токены которого — слова биоматериалов МИС,
    локативная обвязка или предлоги. Хвост из одного слова не снимаем: «кал»,
    «моча», «нос» сами бывают названием услуги.

    :param text: исходный запрос пациента
    :return: ``(запрос_без_биоматериала, снятый_хвост)``; при отсутствии хвоста —
             ``(исходный_текст, "")``
    """

    vocab = _vocabularies()
    if not vocab.biomat_tokens:
        return text, ""
    tokens = _tokens(text)
    if not tokens:
        return text, ""

    idx = len(tokens)
    while idx > 0 and (
        tokens[idx - 1] in vocab.biomat_tokens or tokens[idx - 1] in _LOCATIVE_FILLER
    ):
        idx -= 1
    tail = tokens[idx:]
    if len(tail) < _MIN_TAIL_TOKENS:
        return text, ""
    # Хвост обязан нести собственно материал, а не только предлоги/локативы.
    if not any(t in vocab.biomat_tokens and t not in _PREPOSITIONS for t in tail):
        return text, ""

    head = tokens[:idx]
    while head and head[-1] in _PREPOSITIONS:
        head.pop()
    if not head:
        return text, ""
    return " ".join(head), " ".join(tail)


def service_accepts_biomaterial(service_name: str, biomaterial: str) -> bool | None:
    """Принимает ли услуга такой биоматериал (по `biomatNames` МИС).

    :param service_name: каноническое название услуги из каталога
    :param biomaterial: снятая фраза-биоматериал
    :return: True/False, либо None — услуги нет в справочнике МИС (судить нечем)
    """

    phrases = _vocabularies().biomat_by_service.get(_normalise_input(service_name))
    if not phrases:
        return None
    tail_tokens = {t for t in _tokens(biomaterial) if t not in _PREPOSITIONS}
    if not tail_tokens:
        return None
    allowed_extra = _LOCATIVE_FILLER | _PREPOSITIONS
    for phrase in phrases:
        if tail_tokens <= (set(phrase) | allowed_extra):
            return True
    return False
