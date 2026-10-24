"""Врачи по услуге без списков слов: предмет — фразы разбора хода, врачи — данные МИС (DESIGN-2026-10-23-NO-STOPWORDS, этап 2).

Решения владельца 22.10: 28 списков стоп-слов удалить; без LLM понять, какую услугу назвал пациент, бот не пытается — оффер
оператора или «уточните». Прежний путь (`doctors.doctors_info`) искал услугу и её врачей буквами: токены запроса без
`_SERVICE_FILTER_STOPWORDS` против описаний врачей, «различающий» токен — вне `_PROCEDURE_GENERIC_TOKENS`, связь «врач ↔ услуга» —
по словам названия без `_SERVICE_CODE_STOPWORDS`. (`_CATALOG_SERVICE_STOPWORDS` и `_CATALOG_SERVICE_TRAILING_TIME_RE` читает не
поиск врачей, а подстановка каталога `match_catalog_service` — запись, цена, адрес, walk-in; без них она теряет «а на ферритин
нужна запись?», пока разбора хода нет до её префетча, — STEP-2026-10-24-DOCTORS-BY-PARSE, «Не сделано».)

Здесь (`[MESSENGER_ROUTER] doctors_by_parse = on`):
- что в реплике услуга — фразы разбора (`_turn_spans`: LLM размечает номера слов): разбор хода этой реплики, если он уже есть,
  иначе его строит производитель (`turn_understanding.parse_service_reply`) — один раз за ход, только на вопросе о враче по услуге;
- какая это услуга — данные МИС: строки прайса фразы (`phrase_rows`: название, синоним клиники, код; то же название в другой
  словоформе); фразы нет в данных дословно — услуги с врачами, в названии которых есть все слова фразы (основы), и из них
  выбирает LLM (`choose(doctors_service)`, принцип 2): «колоноскопию» — «Колоноскопия под седацией» и другие её варианты,
  «эндоскопию» — не «Эндоскопия полости носа»;
- врачи услуги — связь «врач ↔ код услуги» в МИС (`doctor_prices`, ход (г) L-06); роль приёма, названная словами МИС (каталог
  ролей L-17), сужает; описание врача — блок роли, под которой он делает услугу по данным;
- сбой разбора или выбора, LLM не узнала услугу, слишком общее слово — «уточните услугу или специальность» с оффером
  оператора (решение (а)); назвал пациент роль словами МИС («кто из урологов…») — врачи этой роли (услугу не угадываем);
  услуга есть, а врачей по ней в данных нет (ЭКГ, анализы) или срез недоступен — оффер оператора:
  «врачей нет» по пустым данным не утверждаем; в реплике услуги нет — вопрос о враче без услуги (роль, филиал), без них — «уточните».
Модуль не читает пяти списков и функций, которые их читают (`tests/test_doctors_by_parse_class.py`, AST и ловушки).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from .. import turn_telemetry, turn_understanding
from ..choice import Option, Status, Task, choose
from ..doctor_name_port import extract_doctor_name_candidate, resolve_schedule_surname
from ..policies import handoff_message
from ..runtime_config import config as _cfg
from ._common import (
    DOCTORS_TOP_N,
    _as_int,
    _coerce_top_n,
    _get_first_present,
    _normalise_input,
    _service_fallback,
)
from ._doctors_helpers import (
    _SERVICE_QUERY_SIGNAL_RE,
    _compact_specialization,
    _dedupe_doctors_by_fio,
    _doctor_main_payload,
    _doctor_matches_fio,
    _doctor_matches_specialty,
    _doctor_role_specialty_match_level,
    _doctor_sort_key,
    _extract_specialty_from_text,
    _is_role_specialty_query,
    _specialization_matches_specialty,
    _specialty_priority_rank,
)
from ._prices_helpers import _catalog_fingerprint, _catalog_rows
from ._role_catalog import _hit as _role_hit, _stem as _role_stem, _tokens as _role_tokens, build_role_catalog
from ._service_doctors import ServiceDoctorIndex, ServiceDoctors
from ._turn_phrases import phrase_rows

if TYPE_CHECKING:
    from .core import Services

logger = logging.getLogger(__name__)

# Что пациент назвал услугой (`doc_parse_outcome`).
PARSE_FAILED = "parse_failed"  # разбора нет: сбой LLM, ответ вне контракта, слишком длинная реплика
WEAK = "weak"  # фразы нет в данных дословно — услугу выбирает LLM из услуг с теми же словами
NO_SERVICE = "no_service"  # в реплике услуги нет, в контексте хода — тоже
ROWS = "rows"  # строки прайса по данным: фраза реплики или услуга из контекста хода

GENERAL = "general"  # исход LLM: пациент назвал услугу общо, а варианты — частные случаи другого органа или разные вмешательства
TASK = Task(name="doctors_service", outcomes=(GENERAL,), num_predict=160)
# Услуг с теми же словами больше — слово общее («узи» — 71 услуга с врачами): «уточните», а не выбор из десятков; у «приём
# гинеколога» вариантов 15 (первичный, повторный, к.м.н., …) — это ещё выбор (свип 23.10, мир 20261020).
MAX_OPTIONS = 24


def enabled() -> bool:
    """`doctors_by_parse = on`: врачи по услуге — фразы разбора и данные МИС вместо пяти списков слов."""

    return str(getattr(_cfg, "MR_DOCTORS_BY_PARSE", "off") or "off").strip().lower() == "on"


@dataclass(frozen=True)
class Subject:
    """Что пациент назвал услугой: исход, слова пациента (текст оффера), строки прайса по данным или фразы для выбора LLM."""

    outcome: str
    label: str = ""
    rows: tuple[dict[str, Any], ...] = ()
    phrases: tuple[str, ...] = ()  # у WEAK — фразы разбора, которых нет в данных дословно
    source: str = ""  # reply — фраза реплики; context — услуга хода (прошлый ход, «по этой услуге»)
    status: str = ""  # статус разбора реплики


@dataclass(frozen=True)
class Found:
    """Какие услуги (коды) назвал пациент, врачи региона, которых МИС с ними связывает, и роли приёма, под которыми они их делают.

    `answer` — исход для ответа: `data` — врачи есть; `no_doctors` — услуга есть, врачей в данных нет; `no_slice` — срез
    недоступен; `clarify` — какую услугу назвал пациент, не понятно (LLM не узнала, сбой выбора, слишком общее слово)."""

    answer: str
    codes: tuple[str, ...] = ()
    doctor_ids: tuple[int, ...] = ()
    unit_ids: frozenset[int] = frozenset()
    options: int = 0


_FORMS: list[tuple[list[dict[str, Any]], tuple[int, str, str], dict[int, list[tuple[tuple[str, ...], int]]]]] = []


def _forms_index(catalog: list[dict[str, Any]]) -> dict[int, list[tuple[tuple[str, ...], int]]]:
    """Основы слов названий прайса по числу слов (срез меняется целиком — индекс по отпечатку, как у `strong_rows`)."""

    stamp = _catalog_fingerprint(catalog)
    if _FORMS and _FORMS[0][1] == stamp:  # срез загружается заново на каждый вопрос — сверяем отпечаток, а не сам список
        return _FORMS[0][2]
    index: dict[int, list[tuple[tuple[str, ...], int]]] = {}
    for pos, row in enumerate(catalog):
        words = _role_tokens(str(row.get("serviceName") or row.get("name") or ""))
        if words:
            index.setdefault(len(words), []).append((tuple(_role_stem(word) for word in words), pos))
    _FORMS[:] = [(catalog, stamp, index)]
    return index


def same_name_rows(phrase: str, catalog: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Строки, чьё название МИС — фраза пациента в другой словоформе: столько же слов, по порядку, каждое — та же основа
    («цистоскопию» — «Цистоскопия»). Механика основ — та же, что у каталога ролей (L-17); часть названия — не название."""

    words = _role_tokens(phrase)
    if not words:
        return []
    index = _forms_index(catalog)
    rows = _FORMS[0][0]  # позиции — в срезе, по которому строился индекс (тот же отпечаток)
    return [
        rows[pos]
        for stems, pos in index.get(len(words), ())
        if all(_role_hit(stem, (word,)) for stem, word in zip(stems, words))
    ]


def _rows_of(phrase: str, rows: tuple[dict[str, Any], ...], catalog: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Строки прайса фразы по данным: сильные разбора (название, синоним клиники, код) или то же название в другой словоформе."""

    return list(rows) or same_name_rows(phrase, catalog)


async def subject_of(reply: str, context_service: str) -> Subject:
    """Что пациент назвал услугой — фразы разбора реплики; в реплике услуги нет — услуга хода из контекста (`service_name`
    прошлого хода: «врачи по этой услуге»), если она — строка данных МИС. Слова реплики код не выкидывает и не угадывает."""

    parse, status = await turn_understanding.parse_service_reply(reply)
    catalog = await asyncio.to_thread(_catalog_rows, None)
    if parse is None:
        return Subject(PARSE_FAILED, status=status)
    if parse.items:
        label = ", ".join(item.text for item in parse.items)
        found = await asyncio.to_thread(lambda: [(item.text, _rows_of(item.text, item.rows, catalog)) for item in parse.items])
        rows = [row for _text, item_rows in found for row in item_rows]
        if rows:  # фразы без строк рядом с узнанной — уточнения при ней, а не вторая услуга
            return Subject(ROWS, label, tuple(_unique(rows)), source="reply", status=status)
        return Subject(WEAK, label, phrases=tuple(text for text, _rows in found), source="reply", status=status)
    context_service = str(context_service or "").strip()
    if context_service:
        rows = await asyncio.to_thread(lambda: _rows_of(context_service, tuple(phrase_rows(context_service, catalog)), catalog))
        if rows:
            return Subject(ROWS, context_service, tuple(_unique(rows)), source="context", status=status)
    return Subject(NO_SERVICE, status=status)


def _unique(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if row not in out:
            out.append(row)
    return out


_SHORT_WORD = 4  # как у каталога ролей: короткое слово («узи», «лор») — только целым словом
_INNER_CORE = 6  # основа короче — только с начала слова: «масс» иначе внутри «пластмассовая», «пневмомассаж» (живой свип 24.10)


def _inside(word: str, name_words: tuple[str, ...]) -> bool:
    """Слово пациента без окончания (до двух последних букв, не короче 4) — в слове названия: длинная основа — где угодно
    («колоноскопию» — «Фиброколоноскопия»), короткая — с начала слова («серную» — «серной»); короткое слово — только целым."""

    if len(word) <= _SHORT_WORD:
        return word in name_words
    core = word[: max(_SHORT_WORD, len(word) - 2)]
    if len(core) < _INNER_CORE:
        return any(name_word.startswith(core) for name_word in name_words)
    return any(core in name_word for name_word in name_words)


def candidate_services(phrases: tuple[str, ...], index: ServiceDoctorIndex) -> list[ServiceDoctors]:
    """Услуги с врачами, в названии которых есть слова фразы (основы, любой порядок; во фразе из трёх слов и длиннее одно может не
    совпасть — глагол «удаляет серную пробку», сокращение «узи малого таза»), — варианты выбора LLM, а не ответ: «слова есть в
    названии» — буквы, не опознание (свип L-06 21.10: «эндоскопия» — только полость носа). Полные совпадения — первыми."""

    wanted = [words for words in (_role_tokens(phrase) for phrase in phrases) if words]
    scored: list[tuple[int, ServiceDoctors]] = []
    for item in index.services.values():
        name_words = _role_tokens(item.name)
        missing = [sum(not _inside(word, name_words) for word in words) for words in wanted]
        best = min((miss for miss, words in zip(missing, wanted) if miss == 0 or (len(words) >= 3 and miss == 1)), default=None)
        if best is not None:
            scored.append((best, item))
    return [item for _miss, item in sorted(scored, key=lambda pair: (pair[0], pair[1].name, pair[1].code))]


def _found(answer: str, codes: tuple[str, ...], index: ServiceDoctorIndex, options: int = 0) -> Found:
    units = {unit for code in codes if (item := index.by_code(code)) is not None for unit in item.unit_ids}
    ids = index.doctors_for(codes)
    return Found(answer if ids or answer != "data" else "no_doctors", codes, ids, frozenset(units), options)


ServiceIndex = Callable[[set[int], Callable[[str], bool]], Awaitable[ServiceDoctorIndex | None]]


async def service_doctors(subject: Subject, question: str, index: ServiceDoctorIndex | None) -> Found:
    """Коды услуги и её врачи: строки по данным — соединением по коду (`serviceHomecode`); фраза не из данных — выбором LLM из
    услуг с теми же словами. Решение «какая услуга» принимает либо данные дословно, либо LLM — не сопоставление слов."""

    if index is None or not index.services:  # срез недоступен или пуст — «нет» по нему не решаем
        return Found("no_slice")
    if subject.outcome == ROWS:
        codes = tuple(code for code in dict.fromkeys(str(row.get("serviceHomecode") or "").strip() for row in subject.rows) if code)
        return _found("data", codes, index)
    pool = await asyncio.to_thread(candidate_services, subject.phrases, index)
    if not pool:
        return Found("no_doctors")
    if len(pool) > MAX_OPTIONS:
        return Found("clarify", options=len(pool))
    choice = await choose(TASK, question, [Option(id=item.code, text=item.name) for item in pool])
    if choice.status is not Status.PICKED or not choice.ids:
        return Found("clarify", options=len(pool))  # не узнала, «названо общо», сбой, бюджет хода — без LLM не угадываем
    return _found("data", tuple(choice.ids), index, len(pool))


def _display_specialization(doc: dict[str, Any], *, unit_ids: frozenset[int], specialty: str) -> str:
    """Описание врача в выдаче: блок роли, под которой он делает услугу по данным МИС; иначе — блок названной специальности;
    иначе — основное описание (порядок прежнего `_pick_display_specialization` без сопоставления слов услуги)."""

    links = [link for link in doc.get("unit_links") or [] if isinstance(link, dict) and str(link.get("specialization") or "").strip()]
    links.sort(key=lambda link: not link.get("main"))  # основные связи — первыми
    for link in links:
        if link.get("company_unit_id") in unit_ids:
            return str(link["specialization"]).strip()
    if specialty:
        for link in links:
            if _specialization_matches_specialty(str(link.get("company_unit_name") or ""), str(link["specialization"]), specialty):
                return str(link["specialization"]).strip()
    _, main_specs, _ = _doctor_main_payload(doc)
    if main_specs:
        return main_specs[0]
    if links:
        return str(links[0]["specialization"]).strip()
    return str(doc.get("specialization") or "")


async def doctors_info(
    self: "Services",
    query: str,
    entities: dict[str, Any],
    output_max: int | None,
    *,
    in_region: Callable[[dict[str, Any]], bool],
    service_index: ServiceIndex,
) -> dict[str, Any]:
    """`doctors_info` под `doctors_by_parse = on`: врачи по имени, роли и филиалу — как прежний путь; по услуге — фразы разбора
    и данные МИС (см. модуль). Контракт payload прежний; `service_clarify` — «уточните услугу или специальность».

    :param in_region: врач принимает в регионе бота (предикат прежнего пути)
    :param service_index: индекс «код услуги → врачи региона» по срезу `doctor_prices` (L-06)
    """

    doctors = await self._ensure_doctors_cache_loaded()
    if not doctors:
        return _service_fallback(
            note="doctors_info source unavailable",
            handoff_message=handoff_message("service_error_doctors_list"),
            entities=entities,
            extra={"doctors": []},
        )

    q = _normalise_input(query)
    doctor_raw = _get_first_present(entities, ["doctor", "doctor_name", "fio", "last_name", "doctor_last_name"]) or ""
    fio_q = _normalise_input(doctor_raw)
    spec_q = _normalise_input(_get_first_present(entities, ["specialty", "specialization", "spec"]) or "")
    if not spec_q:
        spec_q = _extract_specialty_from_text(query)
    context_service = str(_get_first_present(entities, ["service_name", "test_name"]) or "")
    # Вопрос о враче по услуге — тот же круг ходов, что у прежнего пути (сущность услуги или слово процедуры в реплике);
    # что в реплике услуга, решает разбор, а не эти слова.
    service_asked = bool(context_service.strip()) or bool(_SERVICE_QUERY_SIGNAL_RE.search(q))
    region_q = _normalise_input(_get_first_present(entities, ["region", "branch", " филиал", "company_unit"]) or "")
    resolved_surname = resolve_schedule_surname(doctor_raw, doctors) if doctor_raw else None
    query_candidate = extract_doctor_name_candidate(query, prefer_schedule=True) if query else None
    query_resolved = resolve_schedule_surname(query_candidate, doctors) if query_candidate else None
    if not resolved_surname:
        resolved_surname = query_resolved
    if spec_q and not query_resolved:
        fio_q = ""
        resolved_surname = None

    role_query = bool(spec_q and _is_role_specialty_query(query, spec_q))
    role_levels = {id(d): _doctor_role_specialty_match_level(d, spec_q) for d in doctors} if role_query else {}

    def hay(doc: dict[str, Any]) -> str:
        regions = " ".join(_normalise_input(str(x)) for x in (doc.get("regions") or []) if str(x).strip())
        units = " ".join(_normalise_input(str(x)) for x in (doc.get("units") or []))
        return " | ".join([_normalise_input(str(doc.get("fio", ""))), _normalise_input(str(doc.get("specialization", ""))), regions, units])

    def match_doc(doc: dict[str, Any]) -> bool:
        if not in_region(doc):
            return False
        if fio_q and not _doctor_matches_fio(_normalise_input(str(doc.get("fio", ""))), fio_q, resolved_surname):
            return False
        if spec_q:
            if role_query:
                if role_levels.get(id(doc), 0) <= 0:
                    return False
            elif not _doctor_matches_specialty(doc, spec_q, query):
                return False
        if region_q and region_q not in hay(doc):
            return False
        if not (fio_q or spec_q or region_q):
            return len(q) >= 3 and q in hay(doc)
        return True

    def named_role(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Врачи ролей приёма, которые пациент назвал словами МИС (точный слой каталога ролей L-17), — не роль, выведенная из
        процедуры правилом («колоноскопия → эндоскопист» — буквы, их L-06 не допускает)."""

        roles = [role for role in build_role_catalog(doctors, in_region=in_region).find(query).exact if role.reception]
        units = {unit for role in roles for unit in role.unit_ids}
        return [d for d in docs if any((link or {}).get("company_unit_id") in units for link in d.get("unit_links") or [])]

    subject: Subject | None = None
    found: Found | None = None
    filtered: list[dict[str, Any]] = []
    source = "rules"
    if service_asked and not fio_q:  # назван врач — выдачу решает имя, услугу не разбираем
        subject = await subject_of(query, context_service)
    if subject is None or (subject.outcome == NO_SERVICE and (spec_q or region_q)):
        filtered = [d for d in doctors if match_doc(d)]
    elif subject.outcome in (PARSE_FAILED, NO_SERVICE):
        source = "parse_clarify"
    else:
        region_ids = {i for i in (_as_int(d.get("id")) for d in doctors if in_region(d)) if i is not None}
        index = await service_index(region_ids, lambda branch: in_region({"regions": [branch]}))
        found = await service_doctors(subject, query, index)
        by_data = [d for d in doctors if _as_int(d.get("id")) in found.doctor_ids and in_region(d)]
        if region_q:
            by_data = [d for d in by_data if region_q in hay(d)]
        if found.answer == "clarify":
            source = "parse_clarify"
        elif not by_data:
            source = "parse_no_slice" if found.answer == "no_slice" else "parse_no_doctors"
        else:
            narrowed = named_role(by_data)
            filtered = narrowed or by_data
            source = "parse_data_role" if narrowed else "parse_data"
    if source == "parse_clarify":
        # Услугу понять не удалось, но пациент назвал роль («кто из урологов делает …») — врачи этой роли, как на вопрос о роли:
        # услугу не угадываем, роль назвал он сам. Роли словами МИС нет — «уточните».
        filtered = named_role([d for d in doctors if in_region(d)])
        source = "parse_role" if filtered else source
    if subject is not None:
        turn_telemetry.note(
            doc_parse_outcome=subject.outcome,
            doc_parse_status=subject.status,
            doc_parse_src=subject.source or None,
            doc_parse_options=found.options if found is not None else None,
            doc_parse_codes=len(found.codes) if found is not None else None,
            doc_parse_data=len(found.doctor_ids) if found is not None else None,
            doc_parse_answer=source,
        )

    filtered = _dedupe_doctors_by_fio(filtered)
    if role_query:
        filtered = sorted(filtered, key=lambda d: (-role_levels.get(id(d), 0), _specialty_priority_rank(d, spec_q), *_doctor_sort_key(d)))
    else:
        filtered = sorted(filtered, key=_doctor_sort_key)
    limit = _coerce_top_n(output_max, default=DOCTORS_TOP_N)
    if resolved_surname:
        limit = min(limit, 3)
    filtered = filtered[:limit]
    # Полное описание — только когда спрошен конкретный врач; список — выжимка (кейс 08.07: ~12К символов на 4 врачей).
    is_specific_doctor = bool(fio_q or resolved_surname) or len(filtered) == 1
    unit_ids = found.unit_ids if found is not None else frozenset()
    compact: list[dict[str, Any]] = []
    for d in filtered:
        row = dict(d)
        display_spec = _display_specialization(row, unit_ids=unit_ids, specialty=spec_q)
        row["specialization"] = (
            _compact_specialization(display_spec) if is_specific_doctor else _compact_specialization(display_spec, max_lines=4, max_chars=260)
        )
        compact.append(row)

    return {
        "doctors": compact,
        "note": "doctors_info: from cached registry (jsonl), service by turn parse",
        "service_filter_emptied": source in ("parse_no_slice", "parse_no_doctors"),
        "service_clarify": source == "parse_clarify",
        "doctors_source": source,
        "cache_file": self._doctors_cache_path,
        "entities_used": {
            "doctor_query": fio_q,
            "doctor_resolved": resolved_surname,
            "specialty_query": spec_q,
            "service_query": subject.label if subject is not None else "",
            "region_query": region_q,
            "output_limit": limit,
        },
    }
