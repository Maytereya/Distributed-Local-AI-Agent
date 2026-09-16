"""П6 «корзина анализов» (BUG-B): мульти-список анализов распознаётся целиком.

Класс: пациент шлёт СПИСОК анализов (переносы строк, маркеры, или сплошная
строка CAPS без разделителей) → раньше сплиттер (только `, ; + / и`) видел одну
строку → single-путь → ответ про ПЕРВЫЙ анализ. Теперь:
  S1: `\\n` и `•`-маркеры — разделители; шумовые хвосты («Сколько будет стоить?»)
      отбрасываются молча (не попадают ни в услуги, ни в «не распознал»);
  S2: сплошной токен-ран сегментируется ПО КАТАЛОГУ (жадный longest-match:
      лексикон = сам прайс; «ОБЩИЙ БЕЛОК» — 2 токена, «ОАК» — 1);
  S3: нераспознанные позиции честно перечисляются пациенту («Не распознал: …»),
      а не замалчиваются (не дезинформируем полнотой).
Все проверки — на РЕАЛЬНОМ каталоге (region 3), как соседние price-тесты.
"""

from __future__ import annotations

from agent_logic_2.nayka_api import api_price
from messengers_router.renderer import format_price_for_patient
from messengers_router.services._prices_helpers import (
    _build_multi_price_payload,
    _resolve_multi_price_items,
    _split_price_query_items,
)


def _rows():
    return [r for r in api_price.load_price_by_region(3) if isinstance(r, dict)]


# --- S1: переносы строк и маркеры ------------------------------------------

def test_split_newline_list_yields_items_and_drops_price_noise():
    frags = _split_price_query_items("ОАК\nферритин\nвитамин д\nСколько будет стоить?")
    lowered = [f.lower() for f in frags]
    assert "оак" in lowered and "ферритин" in lowered, frags
    assert not any("сколько" in f for f in lowered), f"шумовой хвост попал в услуги: {frags}"


def test_split_bullet_list():
    frags = _split_price_query_items("стоимость • ОАМ • глюкоза • ферритин")
    lowered = [f.lower() for f in frags]
    assert {"оам", "глюкоза", "ферритин"} <= set(lowered), frags


def test_resolve_newline_list_end_to_end():
    items, unrecognized = _resolve_multi_price_items(
        "ОАК\nферритин\nвитамин д\nСколько будет стоить?", _rows()
    )
    names = " | ".join(i["service_name"].lower() for i in items)
    assert len(items) >= 3, names
    assert "общий анализ крови" in names and "ферритин" in names and "витамин d" in names
    assert not unrecognized, unrecognized


# --- S2: сплошная строка без разделителей (сегментация по каталогу) ---------

def test_segment_caps_run_bug_b_replica():
    # Реплика класса BUG-B: CAPS-список одной строкой, мульти-словная позиция
    # («ОБЩИЙ БЕЛОК»), опечатка («ЛАКТАТІ» → Лактат через гомоглиф-фолдинг),
    # шумовой хвост-вопрос.
    q = "ОАК ОАМ ОБЩИЙ БЕЛОК ГЛЮКОЗА ФЕРРИТИН ЛАКТАТІ Сколько будет стоить?"
    items, unrecognized = _resolve_multi_price_items(q, _rows())
    names = " | ".join(i["service_name"].lower() for i in items)
    assert len(items) >= 5, f"распозналось {len(items)}: {names}"
    for need in ("общий анализ крови", "общий анализ мочи", "общий белок", "глюкоза", "ферритин"):
        assert need in names, (need, names)
    assert not any("сколько" in str(u).lower() for u in unrecognized), unrecognized


def test_single_service_query_not_segmented():
    # Регресс: одиночный запрос НЕ должен уходить в multi (гейт ≥2 распознанных).
    items, _ = _resolve_multi_price_items("сколько стоит ттг", _rows())
    assert items == []


def test_non_service_sentence_not_segmented():
    # Регресс: обычная фраза записи не сегментируется в «услуги» по кускам.
    items, _ = _resolve_multi_price_items(
        "запишите меня к кардиологу на завтра после обеда", _rows()
    )
    assert items == []


# --- S3: честный «не распознал» ---------------------------------------------

def test_unrecognized_items_surface_honestly():
    q = "ОАК\nферритин\nКВАНТОВЫЙ АНАЛИЗ АУРЫ\nглюкоза"
    items, unrecognized = _resolve_multi_price_items(q, _rows())
    assert len(items) >= 3
    assert any("аур" in str(u).lower() for u in unrecognized), unrecognized

    payload = _build_multi_price_payload(q, _rows())
    assert payload is not None
    note = str(payload.get("unrecognized_note") or "")
    assert "не распознал" in note.lower() and "аур" in note.lower(), payload.get("unrecognized_note")

    rendered = format_price_for_patient(payload, {})
    assert "не распознал" in rendered.lower(), rendered[-300:]


def test_multi_payload_without_unrecognized_has_no_note():
    payload = _build_multi_price_payload("ОАК, ферритин", _rows())
    assert payload is not None
    assert not payload.get("unrecognized_note")
    rendered = format_price_for_patient(payload, {})
    assert "не распознал" not in rendered.lower()


# --- инвариант цикла не пересчитывается по строкам --------------------------

def test_cart_assembly_does_not_reparse_query_per_price_row():
    """Класс `loop_invariant_recomputed_per_row`.

    16.09 `_price_row_score` звал `extract_specialty_from_text(query)` ДЛЯ
    КАЖДОЙ строки прайса: на живом каталоге это 21 979 разборов ОДНОГО И ТОГО
    ЖЕ запроса на одну корзину — 76% времени сборки. Пациент этого не видел
    только потому, что маршрут до корзины не доходил.

    Проверяем не секунды, а СТРУКТУРУ: сколько раз реально выполнился разбор.
    Порог по времени пришлось бы ставить между 3 и 1 секундой, и на другой
    машине он ловил бы её скорость, а не дефект. Число разборов от машины не
    зависит. См. BUG-2026-09-16-SPECIALTY-PARSE-PER-ROW.
    """

    from messengers_router.specialty_parser import extract_specialty_from_text

    # Именно assert, а не skip. Пропуск означал бы, что снятие мемоизации делает
    # тест молчаливым — ровно тот тип молчаливого ущерба, против которого он и
    # поставлен. Если инвариант станут держать иначе (вынесут вызов из цикла),
    # тест надо переписать под новый механизм, а не дать ему тихо исчезнуть.
    assert hasattr(extract_specialty_from_text, "cache_info"), (
        "разбор специальности больше не мемоизирован; если вызов вынесен из "
        "построчного цикла — перепишите тест под подсчёт вынесенных вызовов"
    )

    rows = _rows()
    assert len(rows) > 1000, "нужен живой каталог, иначе порог ничего не значит"

    extract_specialty_from_text.cache_clear()
    payload = _build_multi_price_payload("ферритин, глюкоза, холестерин", rows)
    misses = extract_specialty_from_text.cache_info().misses

    assert payload is not None
    # Разборов должно быть порядка числа РАЗНЫХ текстов (запрос и его фрагменты),
    # а не числа строк прайса. Запас десятикратный: ловим класс, а не единицы.
    assert misses < len(rows) // 10, (
        f"разбор специальности выполнился {misses} раз при {len(rows)} строках прайса — "
        "похоже, он снова считается внутри построчного цикла"
    )


# --- позиция не исчезает молча в списке -------------------------------------

def test_short_designations_survive_the_list_splitter():
    """Класс `list_drops_resolvable_item`: два слоя не вправе расходиться в
    одном и том же словаре коротких обозначений.

    `_MULTI_PRICE_SERVICE_HINT_RE` требовал три подряд БУКВЫ, поэтому «т3»,
    «т4» и «rw» выбрасывались сплиттером ДО резолва — «т3, т4, ттг» отдавало
    пусто, и пациент не видел ни цены, ни отказа. При этом все три уже лежали в
    `_PRICE_SHORT_TOKEN_WHITELIST`, который уважает токенайзер резолвера.

    Судим ПРЯМО по словарю, а не через резолв живого каталога: в гейте синонимы
    МИС герметично пусты (`_hermetic_mis_synonyms`), и «т3» там не резолвится
    вовсе — проверка «резолвится ли по отдельности» молча пропускала бы ровно
    те случаи, ради которых тест написан. Словарь же — контракт между слоями,
    и он от дневного среза не зависит.

    См. BUG-2026-09-16-CART-DROPS-SHORT-CODE.
    """

    from messengers_router.services._prices_helpers import (
        _PRICE_SHORT_TOKEN_WHITELIST,
        _split_price_query_items,
    )

    lost = [
        short
        for short in sorted(_PRICE_SHORT_TOKEN_WHITELIST)
        if short not in _split_price_query_items(f"{short}, ферритин")
    ]
    assert not lost, (
        f"сплиттер выбрасывает обозначения из словаря резолвера: {lost}"
    )


def test_biomaterial_header_is_not_treated_as_a_service():
    """Заголовок списка — не его пункт.

    Пациент пишет «Кровь: АЛТ, АСТ, ГГТП, цистатин С». После того как двоеточие
    стало разделителем, «Кровь» едва не превратилась в позицию и находила
    «Кровь на стерильность» — услугу, которой пациент не называл. Отличаем по
    словарю биоматериалов МИС, не по списку слов в коде.
    """

    from messengers_router.services._prices_helpers import (
        _resolve_multi_price_items,
        _split_price_query_items,
    )

    rows = _rows()
    fragments = _split_price_query_items("Кровь: АЛТ, АСТ, ГГТП, цистатин С")
    assert "Кровь" not in fragments and "кровь" not in fragments, fragments

    items, _unrecognized = _resolve_multi_price_items("Кровь: АЛТ, АСТ, ГГТП, цистатин С", rows)
    names = " | ".join(str(i.get("service_name") or "") for i in items)
    assert "стерильность" not in names.lower(), f"заголовок стал услугой: {names}"
    # Позиции, которые пациент действительно назвал, должны найтись.
    assert "АлАТ" in names and "АсАТ" in names, names


def test_numeric_fragment_cannot_name_a_service():
    """Чисто числовой фрагмент услугой не становится.

    В словаре синонимов МИС есть ключи-обрывки от разреза химических названий
    по запятой («2» → Мочевая кислота, «11» → Витамин А). Сплиттер обязан
    отбрасывать такие фрагменты, иначе «сколько стоит 2, глюкоза» назовёт
    пациенту цену мочевой кислоты. См. BUG-2026-09-16-NUMERIC-SYNONYM-SHARD.
    """

    from messengers_router.services._prices_helpers import _split_price_query_items

    fragments = _split_price_query_items("сколько стоит 2, глюкоза")
    assert [f for f in fragments if f.strip().isdigit()] == [], fragments
