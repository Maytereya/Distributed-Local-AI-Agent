"""Уточнения позиции, этап 3 — пул выбора строки по материалу (DESIGN-2026-10-28-POSITION-QUALIFIERS; `qualifier_pool = on`).

Решение владельца 27.10, п. 7 (g0372 «посев кала на микрофлору» → общий посев): материал в вопросе теряется при сборе кандидатов по
буквам названия, а выбор не может взять строку, которой нет в пуле. Материал позиции отмечает LLM разметки хода (вид `material`,
этап 2); есть ли он в строке, решают **данные МИС**: у услуги справочника — поле `biomatNames` («Дисбактериоз кишечника» — «кал»;
общий «Посев на микрофлору» — 37 материалов, кала среди них нет), у строки без материалов в справочнике (исследования, приёмы:
«Дуплексное сканирование вен …») — слова её названия. Сверка — по основам слов (морфология, не список слов): «кала» ~ «кал».

Пул позиции с материалом: кандидаты, в которых материал есть или о которых данных нет, плюс строки с этим материалом, ближайшие по
названию к фразе пациента без материала. Строки, которые по данным материала не принимают, в пул не идут: общий посев — не ответ на
посев кала. Какая из строк с материалом — та услуга, решает выбор LLM по смыслу (приравнивать кодом нельзя: синонимы — клиники).
"""

from __future__ import annotations

from typing import Any, Sequence

from .. import turn_parse, turn_telemetry
from ..runtime_config import config as _cfg
from . import _biomaterial
from ._common import _normalise_catalog_text

_ADDED_LIMIT = 12  # строк с материалом сверх кандидатов: пул выбора — до 24 строк, половина — на материал
_MIN_STEM = 3  # «кал», «моч», «вен»: короче — не основа


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
    item = parse.items[0]
    words = [q for q in item.qualifiers if q.kind == turn_parse.QualifierKind.MATERIAL]
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
    return tuple(dict.fromkeys(q.text for q in parse.items[0].qualifiers if q.kind != turn_parse.QualifierKind.PAYMENT))


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
    """Пул выбора строки с материалом: кандидаты, в которых он есть (по данным), затем строки прайса с этим материалом, ближайшие
    по названию к фразе без материала; кандидаты, которые по данным материал не принимают, — вон."""

    from ._price_select_llm import _row_key  # выбор строки импортирует этот модуль — импорт на вызове

    kept = [row for row in pool_rows if accepts(row, material_text) is not False]
    seen = {_row_key(row) for row in kept}
    words = _stems(phrase)

    def closeness(row: dict[str, Any]) -> int:
        name = _stems(str(row.get("serviceName") or row.get("name") or ""))
        return sum(1 for word in words if any(_same_word(word, have) for have in name))

    found = [row for row in retail_rows if isinstance(row, dict) and _row_key(row) not in seen and accepts(row, material_text)]
    found.sort(key=closeness, reverse=True)
    added = found[:_ADDED_LIMIT]
    turn_telemetry.note(pool_material=f"{len(kept)}/{len(pool_rows) - len(kept)}/{len(added)}")
    return [*kept, *added][:limit]
