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

# Речевая обвязка вокруг названия услуги. Список ЛИНГВИСТИЧЕСКИЙ (как пациент
# оформляет просьбу), а не медицинский — услуг и болезней тут нет и быть не
# должно. Нужен, чтобы отличить «синоним НАЗЫВАЕТ услугу» («сколько стоит
# тироксин») от «синоним случайно встретился внутри чужого названия» («CA 15 - 3
# (молочная железа)» ← «ca» = кальций). Свип 09.09: без этого 182 названия
# каталога из 1200 уводились в чужую услугу.
_SPEECH_WRAPPER = frozenset({
    "сколько", "стоит", "стоимость", "цена", "цену", "по", "чем",
    "хочу", "хочется", "надо", "нужно", "нужен", "нужна", "можно", "могу",
    "сдать", "сдаю", "сдам", "сдача", "сделать", "пройти", "записаться", "запишите",
    "подскажите", "скажите", "уточните", "пожалуйста", "здравствуйте", "добрый", "день",
    "а", "и", "ли", "это", "мне", "у", "вас", "в", "на", "с", "со", "за", "от", "до",
    "анализ", "анализы", "услуга", "услуги", "тест",
})


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(_normalise_input(text))


class _MisVocabularies:
    """Разобранные словари МИС с кэшем по (путь, mtime) дневного среза."""

    def __init__(self) -> None:
        self.biomat_tokens: frozenset[str] = frozenset()
        self.biomat_phrases: tuple[frozenset[str], ...] = ()
        self.synonym_to_service: dict[str, str] = {}
        # Все услуги на синоним. 9% живого словаря клиники неоднозначны («вэб» →
        # 9 услуг, «рак» → 4 онкомаркера), поэтому кандидаты храним полностью, а
        # `synonym_to_service` оставляем ТОЛЬКО для однозначных — молча выбрать
        # одну из четырёх значило бы вернуть класс «дезинформация ценой».
        self.synonym_candidates: dict[str, tuple[str, ...]] = {}
        self.biomat_by_service: dict[str, tuple[frozenset[str], ...]] = {}


_VOCAB_CACHE: dict[tuple[str, float], _MisVocabularies] = {}


def _cache_key() -> tuple[str, float] | None:
    try:
        path: Path = api_service_info.service_info_path()
        stat = path.stat()
    except Exception:
        return None
    return (str(path), stat.st_mtime)


def _can_be_a_synonym(key: str) -> bool:
    """Мог ли человек написать это как НАЗВАНИЕ услуги.

    Разделители в `serviceSynonyms` задаёт клиника руками, и запятые встречаются
    ВНУТРИ химических названий: «11, 13-диметил-7-(1,5-диметилгексил)…». Разрез
    по запятой превращает одно название в обрывки, и каждый обрывок становился
    полноправным ключом словаря. На срезе 17.09 таких ключей шесть — «1», «2»,
    «6», «9», «11», «18», — и через них бот отвечал на «сколько стоит 2» ценой
    мочевой кислоты. Опаснее, чем кажется: бот сам печатает нумерованные списки
    и предлагает выбрать, так что «2» — естественный ход диалога.

    Правило минимальное: в названии услуги обязана быть хоть одна БУКВА. Порог
    по длине не годится — «ca», «fe», «lh», «p4», «т3», «rw» и ещё два десятка
    коротких ключей законны, их клиника завела осознанно.

    См. BUG-2026-09-16-NUMERIC-SYNONYM-SHARD.

    :param key: нормализованный ключ-кандидат
    :return: True, если ключ может быть названием услуги
    """

    return any(char.isalpha() for char in key)


def _build_vocabularies(rows: list[dict[str, Any]]) -> _MisVocabularies:
    vocab = _MisVocabularies()
    candidates: dict[str, list[str]] = {}
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
            # Разделители задаёт клиника руками: в выгрузке встречаются и запятые,
            # и точки с запятой, и «|| Хламидия трахоматис ||».
            for part in re.split(r"[,;]|\|\|", str(value or "")):
                key = _normalise_input(part)
                if not key or not service_name or not _can_be_a_synonym(key):
                    continue
                seen = candidates.setdefault(key, [])
                if service_name not in seen:
                    seen.append(service_name)

    vocab.biomat_tokens = frozenset(biomat_tokens)
    vocab.biomat_phrases = tuple(phrases)
    vocab.synonym_candidates = {k: tuple(v) for k, v in candidates.items()}
    vocab.synonym_to_service = {k: v[0] for k, v in candidates.items() if len(v) == 1}
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


def _significant_tokens(text: str) -> list[str]:
    """Токены без предлогов.

    Предлоги не значащая часть названия услуги ни в словаре клиники, ни в
    реплике пациента: «кровь С лейкоформулой» и «кровь лейкоформулой» — одно и
    то же. Прод 15.09: предлог срезался выше по пути, синоним переставал
    находиться, и пациенту предлагали ЧУЖУЮ услугу («Лейкоцитарная формула»
    210 руб. вместо ОАК 550 руб.).
    """

    return [tok for tok in _tokens(text) if tok not in _PREPOSITIONS]


def _synonym_index(vocab: _MisVocabularies) -> dict[str, tuple[tuple[str, ...], str]]:
    """Индекс «первый значащий токен синонима → (токены, ключ)» для вхождения."""

    source = vocab.synonym_candidates or {k: (v,) for k, v in vocab.synonym_to_service.items()}
    index: dict[str, list[tuple[tuple[str, ...], str]]] = {}
    for key in source:
        toks = tuple(_significant_tokens(key))
        if toks:
            index.setdefault(toks[0], []).append((toks, key))
    return index


def _match_synonym_key(query_text: str, vocab: _MisVocabularies) -> str | None:
    """Самый длинный синоним, входящий в реплику непрерывной цепочкой токенов.

    Сверка реплики ЦЕЛИКОМ обесценивает словарь: живая формулировка почти всегда
    несёт вокруг названия глаголы и вежливость («кровь с лейкоформулой сдать
    хочу»). Ищем вхождение, но ТОЛЬКО по границам токенов — иначе «оак» нашёлся
    бы внутри «трОАКарная». Из нескольких подошедших берём длиннейший: он точнее
    («глюкоза в моче» важнее, чем просто «моча»).

    :param query_text: реплика пациента
    :param vocab: разобранные словари МИС
    :return: ключ словаря либо None
    """

    q = _significant_tokens(query_text)
    if not q:
        return None
    index = _synonym_index(vocab)
    best: str | None = None
    best_len = 0
    for pos, token in enumerate(q):
        for toks, key in index.get(token, ()):
            n = len(toks)
            if n <= best_len or pos + n > len(q):
                continue
            if tuple(q[pos : pos + n]) != toks:
                continue
            # Синоним обязан НАЗЫВАТЬ услугу: всё, что осталось сверх него, —
            # только речевая обвязка. Содержательный остаток означает, что
            # синоним встретился внутри чужого названия.
            rest = q[:pos] + q[pos + n :]
            if any(tok not in _SPEECH_WRAPPER for tok in rest):
                continue
            best, best_len = key, n
    return best


def mis_synonym_candidates(
    query_text: str, *, vocab: _MisVocabularies | None = None
) -> tuple[str, ...]:
    """Все услуги, на которые указывает синоним из реплики.

    9% живого словаря клиники неоднозначны: «вэб» → 9 услуг, «рак» → 4 разных
    онкомаркера, «холестерин» → 4 анализа. Вызывающая сторона обязана развести
    их развилкой, а не выбирать молча.

    :param query_text: реплика пациента
    :param vocab: словари МИС (по умолчанию — дневной срез)
    :return: кортеж канонических названий услуг (пустой, если синонима нет)
    """

    vocab = vocab if vocab is not None else _vocabularies()
    key = _match_synonym_key(query_text, vocab)
    if key is None:
        return ()
    if vocab.synonym_candidates:
        return vocab.synonym_candidates.get(key, ())
    single = vocab.synonym_to_service.get(key)
    return (single,) if single else ()


def mis_synonym_service(
    query_text: str, *, vocab: _MisVocabularies | None = None
) -> str | None:
    """Услуга по синониму МИС — ТОЛЬКО когда синоним однозначен.

    Однозначный синоним — высшее доверие: клиника прямо сказала «это слово
    означает вот эту услугу». Неоднозначный отдаётся как None: выбрать одну из
    нескольких молча — это класс «дезинформация ценой», см.
    `mis_synonym_candidates`.

    :param query_text: реплика пациента
    :param vocab: словари МИС (по умолчанию — дневной срез)
    :return: каноническое название услуги либо None
    """

    found = mis_synonym_candidates(query_text, vocab=vocab)
    return found[0] if len(found) == 1 else None


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


def is_biomaterial_phrase(text: str) -> bool:
    """Состоит ли фраза ТОЛЬКО из слов-биоматериалов МИС.

    Нужна, чтобы отличить ЗАГОЛОВОК списка от его пункта: пациент пишет
    «Кровь: АЛТ, АСТ, ГГТП», и «Кровь» здесь — не услуга, а указание на
    материал. Без этого разрез по двоеточию делал «Кровь» отдельной позицией,
    и она находила «Кровь на стерильность» — услугу, которой пациент не
    называл (BUG-2026-09-16-CART-DROPS-SHORT-CODE).

    Судим по словарю МИС, а не по списку слов в коде: `biomatNames` заполнен у
    98% услуг и обновляется вместе со справочником.

    :param text: фрагмент реплики
    :return: True, если все значащие слова фрагмента — названия биоматериалов
    """

    vocab = _vocabularies()
    if not vocab.biomat_tokens:
        return False
    tokens = _significant_tokens(text)
    if not tokens:
        return False
    return all(token in vocab.biomat_tokens for token in tokens)
