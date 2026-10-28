"""Уточнения позиции, этап 3 — пул выбора строки по материалу (DESIGN-2026-10-28-POSITION-QUALIFIERS; `qualifier_pool = on`).

Решение владельца 27.10, п. 7 (g0372 «посев кала на микрофлору» → общий посев): материал в вопросе теряется при сборе кандидатов по
буквам названия, а выбор не может взять строку, которой нет в пуле. Материал позиции отмечает LLM разметки хода (вид `material`,
этап 2); есть ли он в строке, решают **данные МИС**: у услуги справочника — поле `biomatNames` («Дисбактериоз кишечника» — «кал»;
общий «Посев на микрофлору» — 37 материалов, кала среди них нет), у строки без материалов в справочнике (исследования, приёмы:
«Дуплексное сканирование вен …») — слова её названия. Сверка — по основам слов (морфология, не список слов): «кала» ~ «кал».

Пул позиции с материалом: кандидаты как были плюс строки с этим материалом, ближайшие по названию к фразе пациента без материала.
Кандидатов пул не исключает (карточка 28.10: исключение по тексту `biomatNames` теряло верные строки); общий посев на «посев кала»
честно помечает отметка выбора «материала в строке нет» (этап 4). Какая строка — та услуга, решает выбор LLM по смыслу.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .. import turn_parse, turn_telemetry
from ..choice import Option, Task, choose
from ..runtime_config import config as _cfg
from . import _biomaterial
from ._common import _normalise_catalog_text

_ADDED_LIMIT = 12  # строк с материалом сверх кандидатов: пул выбора — до 24 строк, половина — на материал
_MIN_STEM = 3  # «кал», «моч», «вен»: короче — не основа
# Этап 4 — проверка выбора отдельным вопросом (карточка кнопок цены 28.10, `c3kb*`): отметка «есть в строках» тем же вызовом, что
# выбор строки, защищала выбор — «суммарные» «есть» в IgG и IgM, «ПЦР» — в «Аскарида (p1)», спермограмма — на «пероксидазный тест».
# Данные справочника под строкой (другое название, материал, выдержка описания) удлиняют запрос — тайм-аут на секунду больше.
CHECK_TASK = Task(name="qualifier_in_rows", num_predict=48, timeout_s=5.0, queue_timeout_ms=1500, verdict=True)
# Данные справочника МИС под строкой проверки: названий каждого вида — не больше двух, из описания — не больше двух выдержек по
# `_EXCERPT_WORDS` слов вокруг слов пациента (окно начинается за `_EXCERPT_BEFORE` слов до совпадения); строк в запросе — не больше
# `_CHECK_ROWS` (разных названий).
_MIS_NAMES = 2
_EXCERPTS = 2
_EXCERPT_WORDS = 32
_EXCERPT_BEFORE = 10
_CHECK_ROWS = 6


def enabled() -> bool:
    """`[MESSENGER_ROUTER] qualifier_pool = on` (нужны `position_qualifiers = on` и разметка `turn_spans`)."""

    from . import _turn_spans  # разметка импортирует сервисы цены — импорт на вызове

    return str(getattr(_cfg, "MR_QUALIFIER_POOL", "off") or "off").strip().lower() == "on" and _turn_spans.qualifiers_enabled()


def material(question: str) -> tuple[str, str]:
    """Материал единственной позиции разбора этой реплики и её фраза без слов материала: («кала», «посев на микрофлору»);
    материала нет, позиция не одна, разбора нет или выключатель off — («», «»)."""

    if not enabled():
        return "", ""
    from .. import turn_understanding  # понимание хода импортирует сервисы цены — импорт на вызове

    parse = turn_understanding.parse_for(question)
    if parse is None or not parse.decided or len(parse.items) != 1:
        return "", ""
    return of_phrase(parse.items[0])


def of_phrase(item: Any) -> tuple[str, str]:
    """Материал фразы разбора (вид `material`) и её слова без него: («кала», «посев на микрофлору»); материала нет — («», «»)."""

    words = [q for q in getattr(item, "qualifiers", ()) if q.kind == turn_parse.QualifierKind.MATERIAL]
    if not words:
        return "", ""
    text = str(item.text or "")
    for qualifier in words:
        if qualifier.inside:
            local = qualifier.start - item.start
            text = text[:local] + " " * (qualifier.end - qualifier.start) + text[local + qualifier.end - qualifier.start :]
    return " ".join(q.text for q in words), " ".join(text.split())


def marks_enabled() -> bool:
    """`[MESSENGER_ROUTER] qualifier_marks = on` (этап 4; нужны `position_qualifiers = on` и разметка `turn_spans`)."""

    from . import _turn_spans  # разметка импортирует сервисы цены — импорт на вызове

    return str(getattr(_cfg, "MR_QUALIFIER_MARKS", "off") or "off").strip().lower() == "on" and _turn_spans.qualifiers_enabled()


def to_mark(question: str) -> tuple[str, ...]:
    """Уточнения единственной позиции разбора этой реплики, которые выбор строки отмечает «есть в строке» (этап 4): все виды,
    кроме оплаты (её решает этап 2 — без розничной цены). Выключатель off, позиция не одна, уточнений нет — пусто."""

    if not marks_enabled():
        return ()
    from .. import turn_understanding  # понимание хода импортирует сервисы цены — импорт на вызове

    parse = turn_understanding.parse_for(question)
    if parse is None or not parse.decided or len(parse.items) != 1:
        return ()
    return tuple(dict.fromkeys(q.text for q in parse.items[0].qualifiers if q.kind != turn_parse.QualifierKind.INSURANCE))


def addons(question: str) -> frozenset[str]:
    """Сопутствующие услуги (вид `addon`: «под наркозом», «с седацией») единственной позиции разбора этой реплики — их ненайденная
    строка не «другая услуга», а отдельная цена: «стоимость уточнит оператор» (решение 20.10 `4cec16e`, 27.10 п. 12)."""

    if not marks_enabled():
        return frozenset()
    from .. import turn_understanding  # понимание хода импортирует сервисы цены — импорт на вызове

    parse = turn_understanding.parse_for(question)
    if parse is None or not parse.decided or len(parse.items) != 1:
        return frozenset()
    return frozenset(q.text for q in parse.items[0].qualifiers if q.kind == turn_parse.QualifierKind.ADDON)


@dataclass(frozen=True)
class RowsCheck:
    """Что из названного пациентом есть в выбранных строках: сама услуга (`service`; не спрашивали — True) и уточнения, которых нет."""

    service: bool
    missing: tuple[str, ...] = ()


def named(question: str) -> str:
    """Название услуги единственной позиции разбора без её уточнений внутри фразы («антитела к аннексину» из «антитела к аннексину
    суммарные»); позиция не одна или разбора нет — пусто."""

    from .. import turn_understanding  # понимание хода импортирует сервисы цены — импорт на вызове

    parse = turn_understanding.parse_for(question)
    if parse is None or not parse.decided or len(parse.items) != 1:
        return ""
    item = parse.items[0]
    text = str(item.text or "")
    for qualifier in sorted((q for q in item.qualifiers if q.inside), key=lambda q: q.start, reverse=True):
        local = qualifier.start - item.start
        text = text[:local] + " " + text[local + qualifier.end - qualifier.start :]
    return " ".join(text.split()).strip(" ,.;:()")


def _row_name(row: dict[str, Any]) -> str:
    return " ".join(str(row.get("serviceName") or row.get("name") or "").split())


def _matching(texts: Sequence[str], wanted: Sequence[str], limit: int) -> list[str]:
    """Тексты справочника (названия или предложения описания), в которых больше всего слов пациента (по основам, `_same_word`),
    лучшие первыми, не больше `limit`; ни одного слова — пусто. Сужение кандидатов для LLM, не вердикт: судит проверка."""

    scored = []
    for k, text in enumerate(texts):
        have = _stems(text)
        score = sum(1 for want in wanted if any(_same_word(want, word) for word in have))
        if score:
            scored.append((-score, k, text))
    return [text for _score, _k, text in sorted(scored)[:limit]]


def _excerpts(description: str, wanted: Sequence[str]) -> list[str]:
    """Выдержки описания МИС вокруг слов пациента: окна по `_EXCERPT_WORDS` слов, в которых больше всего разных слов пациента
    (по основам), не больше `_EXCERPTS`, без пересечений, в порядке текста; ни одного слова — пусто. Сужение для LLM, не вердикт."""

    words = description.split()
    stems = [_stems(word) for word in words]
    hits = [k for k, found in enumerate(stems) if any(_same_word(want, have) for want in wanted for have in found)]
    windows = []
    for k in hits:
        start = max(0, k - _EXCERPT_BEFORE)
        covered = {want for found in stems[start : start + _EXCERPT_WORDS] for have in found for want in wanted if _same_word(want, have)}
        windows.append((-len(covered), start))
    chosen: list[int] = []
    for _score, start in sorted(windows):
        if len(chosen) < _EXCERPTS and all(abs(start - other) >= _EXCERPT_WORDS for other in chosen):
            chosen.append(start)
    return [
        ("… " if start else "") + " ".join(words[start : start + _EXCERPT_WORDS]) + (" …" if start + _EXCERPT_WORDS < len(words) else "")
        for start in sorted(chosen)
    ]


def rows_context(rows: Sequence[dict[str, Any]], words: Sequence[str]) -> str:
    """Строки для проверки выбора: «номер. название» и под ним — что справочник МИС знает о строке и назвал пациент (другие
    названия `serviceSynonyms`, материалы `biomatNames`, выдержки описания `description` вокруг слов пациента). Повтор названия
    (одна услуга разных филиалов и цен) — одной строкой. Данных нет — только название, как в прайсе."""

    wanted = [stem for text in words for stem in _stems(text)]
    vocab = _biomaterial._vocabularies() if wanted else None  # один раз на запрос: у словаря — проверка среза на диске
    names = list(dict.fromkeys(name for name in (_row_name(row) for row in rows) if name))[:_CHECK_ROWS]
    lines: list[str] = []
    for number, name in enumerate(names, 1):
        lines.append(f"{number}. {name}")
        if not wanted:
            continue
        materials, synonyms = _biomaterial.service_mis_names(name, vocab=vocab)
        for label, found in (
            ("другое название в справочнике клиники", _matching(synonyms, wanted, _MIS_NAMES)),
            ("материал по справочнику клиники", _matching(materials, wanted, _MIS_NAMES)),
            ("из описания исследования в справочнике клиники", _excerpts(_biomaterial.service_mis_description(name, vocab=vocab), wanted)),
        ):
            if found:
                lines.append(f"- {label}: {' / '.join(found)}")  # чистка запроса снимает отступы — данные строки под ней с дефисом
    return "\n".join(lines)


async def check_rows(question: str, rows: Sequence[dict[str, Any]]) -> RowsCheck | None:
    """Проверка выбора строки (`qualifier_marks = on`): отдельный вопрос LLM — какие из слов пациента (название услуги и каждое
    уточнение позиции, кроме оплаты) есть в выбранных строках по смыслу. Строки с данными справочника МИС (`rows_context`) — данные
    хода (`context`), варианты — слова пациента. Один механизм для цены (`_price_select_llm._checked`) и подбора анализов
    (`test_assist_answer`): «в таком виде нет» пишет код по этому ответу, а не вызов, который выбирал строки.
    Решение хода вопрос не тратит и отказа лимита числа решений не получает (`Task.verdict`; прецедент — решение владельца Q5 18.10):
    с возвратом решения после вызова проверка на ходе подбора анализов упиралась в исчерпанный лимит и молча не шла. None —
    уточнений нет, строк нет или вопрос не ответил (выбор как есть, заголовок прежний)."""

    marks = to_mark(question)
    if not marks or not rows:
        return None
    service = named(question)
    words = ([("service", service)] if service else []) + [(f"q{k}", text) for k, text in enumerate(marks)]
    choice = await choose(
        CHECK_TASK, question, [Option(id=key, text=text) for key, text in words], context=rows_context(rows, [text for _key, text in words])
    )
    if choice.failed:
        turn_telemetry.note(price_qual_check="unavailable")
        return None
    found = set(choice.ids)
    check = RowsCheck(service=not service or "service" in found, missing=tuple(text for key, text in words if key != "service" and key not in found))
    turn_telemetry.note(price_qual_check="service_absent" if not check.service else "missing" if check.missing else "found")
    return check


def note_missing(missing: Sequence[str]) -> tuple[str, ...]:
    """Ненайденные уточнения выбора — в телеметрию хода (`price_qual_missing`, число) и вызывающему."""

    if missing:
        turn_telemetry.note(price_qual_missing=len(missing))
    return tuple(missing)


def _stems(text: str) -> list[str]:
    return [word for word in _normalise_catalog_text(text).split() if len(word) >= _MIN_STEM]


def _same_word(a: str, b: str) -> bool:
    """Одно слово в разных формах: общее начало длиной в короткое слово без последней буквы (не меньше трёх), после него у
    длинного — не больше двух букв окончания: «кала» ~ «кал», «мочи» ~ «моча», «крови» ~ «кровь»; «кала» ≁ «кальций»."""

    short, long_ = sorted((a, b), key=len)
    size = max(_MIN_STEM, len(short) - 1)
    return len(short) >= _MIN_STEM and long_[:size] == short[:size] and len(long_) - size <= 2


def _covers(material_words: Sequence[str], text: str) -> bool:
    words = _stems(text)
    return all(any(_same_word(want, have) for have in words) for want in material_words)


def accepts(row: dict[str, Any], material_text: str) -> bool | None:
    """Есть ли материал в строке: по `biomatNames` справочника МИС, у строки без материалов в справочнике — по словам названия.
    None — судить нечем (у материала нет слов)."""

    want = _stems(material_text)
    if not want:
        return None
    name = str(row.get("serviceName") or row.get("name") or "")
    biomats, _synonyms = _biomaterial.service_mis_names(name)
    if biomats:
        return any(_covers(want, biomat) for biomat in biomats)
    return _covers(want, name)


def pool(pool_rows: Sequence[dict[str, Any]], material_text: str, phrase: str, retail_rows: Sequence[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Пул выбора строки с материалом: кандидаты как были, плюс строки прайса с этим материалом (по данным МИС), ближайшие по
    названию к фразе без материала. Кандидатов не исключает: карточка 28.10 (`c3ton_*`) — исключение по тексту `biomatNames`
    выкидывало верные строки («с поверхности задней стенки глотки» ≠ «соскоб с задней стенки глотки», «глюкоза в плазме»), а
    «материалом» LLM разметки называла и возбудитель, и аллерген. Есть ли материал в выбранной строке — отметка выбора (этап 4)."""

    from ._price_select_llm import _row_key  # выбор строки импортирует этот модуль — импорт на вызове

    kept = list(pool_rows)
    seen = {_row_key(row) for row in kept}
    words = _stems(phrase)

    def closeness(row: dict[str, Any]) -> int:
        name = _stems(str(row.get("serviceName") or row.get("name") or ""))
        return sum(1 for word in words if any(_same_word(word, have) for have in name))

    found = [row for row in retail_rows if isinstance(row, dict) and _row_key(row) not in seen and accepts(row, material_text)]
    found.sort(key=closeness, reverse=True)
    added = found[:_ADDED_LIMIT]
    turn_telemetry.note(pool_material=f"{len(kept)}/0/{len(added)}")
    # Добавленные — в пределах пула: строки с материалом важнее хвоста кандидатов по буквам.
    room = max(0, limit - len(added))
    return [*kept[:room], *added][:limit]
