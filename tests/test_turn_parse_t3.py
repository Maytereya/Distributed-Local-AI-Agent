"""Правки по разбору 95 изменившихся ответов `turn_parse` off → on (L-15b, шаг 6, T3) — через обработчик
эндпоинта, мини-прод. Каждая правка действует только под `turn_parse = on`; под `off` и `shadow` поведение
прежнее (эталоны `test_price_turns_golden.py`, `test_turn_parse_shadow.py`).

Классы (журнал STEP-2026-10-14-L15B-6-T3):
- п. 1 — гард П2 в корзине из фраз судит остаток после резолвера, а не всю слабую фразу: фраза, которую
  резолвер разрешает («лактозная непереносимость», «сывороточное железо»), получает кандидатов; фраза с
  уточнением, которого в прайсе нет («приём терапевта по ОМС»), — по-прежнему «Не распознал»;
- п. 2 — подбор анализов не о цене («есть ли полный чекап», «что входит в комплекс»): разбор «одна услуга»
  не выключает сегментацию правил — выдача подбора та же, что под off; при вопросе о цене разбор читается;
- п. 4 — путь одной услуги по фразе разбора: сделан и отменён (короткая фраза LLM уводила к чужим строкам);
  держится граница — фраза не заменяет то, что находит реплика; класс g0609 — `xfail(strict=True)`;
- п. 7 — корзина из фраз: бюджет вариантов слабых позиций — поровну с передачей недобора (позиции без
  кандидатов не съедают долю); строка прайса — один раз в ответе (повтор у второй позиции сливается);
- п. 8 — «Не распознал» корзины из фраз не печатает фрагмент только из обвязки и родовых слов («анализов»).

Две LLM, как в классе «одна или несколько»: «недоступна» (гейт) и оракул — на выборе фраз отмечает ровно
фразы разреза случая, на выборе строк — строки с заданным началом названия, прочие места — «недоступна».
Тест на два хода (правило «Слепое пятно»): после ответа короткий вопрос («а сколько?», «да», цифра) не
получает цену услуги, которой пациент в первом ответе не видел.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from ollama import GenerateResponse

from evalbench import world as ev_world
from messengers_router import llm_runtime
from messengers_router.services._common import _normalise_catalog_text
from tests.test_price_one_or_many_class import _priced_sections
from tests.test_price_turns_golden import mini_prod_world, run_dialog  # noqa: F401 — фикстура
from tests.test_turn_parse_shadow import _deploy

_PHRASES_MARK = "понять, какие услуги назвал пациент"
_ROWS_MARKS = ("ответить на вопрос о ценах нескольких услуг", "ответить пациенту на вопрос о цене")
_FOLLOW_UPS = ("а сколько?", "да", "2")


def _parse_on(monkeypatch, mode: str = "on") -> None:
    from messengers_router.runtime_config import config as cfg

    monkeypatch.setattr(cfg, "MR_TURN_PARSE", mode, raising=False)


class _Oracle:
    """Выбор фраз — ровно `phrases`; выбор строк (корзина, цена) — строки, чьё название начинается с одного из
    `rows`; прочие места — «недоступна»."""

    def __init__(self, phrases: tuple[str, ...], rows: tuple[str, ...] = ()) -> None:
        self.phrases = {_normalise_catalog_text(text) for text in phrases}
        self.rows = rows
        self.asked: list[str] = []
        self.rows_prompts: list[str] = []

    async def generate(self, *, prompt: str = "", system: str | None = None, **_kwargs):
        seen = f"{system}\n\n{prompt}" if system else prompt
        options = [(int(head), rest) for head, dot, rest in (line.partition(". ") for line in prompt.splitlines()) if dot and head.isdigit()]
        if _PHRASES_MARK in seen:
            self.asked.append("phrases")
            picks = [n for n, text in options if _normalise_catalog_text(text.strip("«»")) in self.phrases]
            return GenerateResponse(model="m", response=json.dumps({"picks": picks}), done=True)
        if self.rows and any(mark in seen for mark in _ROWS_MARKS):
            self.asked.append("rows")
            self.rows_prompts.append(prompt)
            picks = [n for n, text in options if text.split(" → ", 1)[-1].startswith(self.rows)]
            answer = {"match": picks, "search_terms": []} if system is None else {"picks": picks}
            return GenerateResponse(model="m", response=json.dumps(answer, ensure_ascii=False), done=True)
        raise ConnectionError("тест: LLM недоступна для этого места")


def _dialog(turns: list[str], world_dir, monkeypatch, session: str, llm: _Oracle | None = None):
    _deploy(monkeypatch)
    _parse_on(monkeypatch)
    if llm is not None:
        monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", llm)
    with ev_world.FrozenWorld(world_dir, "day") as world:
        record = run_dialog(turns, session)
        rows = list(world.telemetry)
        attempts = list(world.net.attempts)
    assert attempts == []
    return record, rows


def _priced(text: str) -> set[str]:
    return {name for section in _priced_sections(text) for name in section}


def _assert_follow_up_stays_on_what_was_shown(first: dict, second: dict) -> None:
    """Короткий вопрос после ответа: цены — только строк, которые пациент уже видел (или без цены вовсе),
    услуга сессии — не услуга, которой в ответе не было."""

    shown = _priced(first["text"])
    assert _priced(second["text"]) <= shown, (second["text"], shown)
    remembered = second.get("service_name")
    if remembered:
        assert any(remembered in name or name in remembered for name in shown | {first.get("service_name_effective") or ""}), remembered


# --------------------------------------------------------------------------- п. 1: гард П2 — остаток после резолвера

_RESOLVED = "сколько стоит ферритин, лактозная непереносимость и сывороточное железо"
_RESOLVED_ORACLE = (("ферритин", "лактозная непереносимость", "сывороточное железо"), ("Непереносимость лактозы", "Железо в сыворотке"))


@pytest.mark.parametrize("llm", ["down", "oracle"])
def test_weak_phrase_the_resolver_knows_gets_candidates(llm, mini_prod_world, monkeypatch):  # noqa: F811
    """Слово пациента, которого нет в названиях прайса («лактозная», «сывороточное»), — не уточнение, если
    резолвер фразу разрешил: позиция получает кандидатов (выбор LLM или варианты кода при сбое), а не «Не
    распознал» (G3: g0074, g0144, g0187, g0281, g0363, g0390, g0425, g0465, g0533, r0141)."""

    oracle = _Oracle(*_RESOLVED_ORACLE) if llm == "oracle" else None
    [turn], [row] = _dialog([_RESOLVED], mini_prod_world, monkeypatch, f"t3-p2-resolved-{llm}", oracle)
    assert row["tu_parse_status"] == ("picked" if llm == "oracle" else "unavailable")
    assert "Не распознал" not in turn["text"], turn["text"]
    assert "Ферритин — 530" in turn["text"] and "Железо в сыворотке — 240" in turn["text"]
    assert "Непереносимость лактозы" in turn["text"]
    if llm == "oracle":
        assert "Непереносимость лактозы — 1 100" in turn["text"] and oracle.asked == ["phrases", "rows"]


@pytest.mark.parametrize("llm", ["down", "oracle"])
def test_phrase_with_a_qualifier_the_price_lacks_stays_unrecognized(llm, mini_prod_world, monkeypatch):  # noqa: F811
    """Граница класса: «приём терапевта по ОМС» резолвер не разрешает (уточнения нет в прайсе) — «Не
    распознал», не платные приёмы (решение владельца П2)."""

    oracle = _Oracle(("ферритин", "приём терапевта по ОМС"), ("Прием",)) if llm == "oracle" else None
    [turn], [row] = _dialog(["сколько стоит ферритин и приём терапевта по ОМС"], mini_prod_world, monkeypatch, f"t3-p2-oms-{llm}", oracle)
    assert "Ферритин — 530" in turn["text"] and "Не распознал: приём терапевта по ОМС" in turn["text"]
    assert "врача-терапевта —" not in turn["text"]


@pytest.mark.parametrize("follow_up", _FOLLOW_UPS)
@pytest.mark.parametrize("llm", ["down", "oracle"])
def test_follow_up_after_the_resolved_basket_stays_on_what_was_shown(llm, follow_up, mini_prod_world, monkeypatch):  # noqa: F811
    oracle = _Oracle(*_RESOLVED_ORACLE) if llm == "oracle" else None
    first, second = _dialog([_RESOLVED, follow_up], mini_prod_world, monkeypatch, f"t3-p2-two-{llm}-{follow_up}", oracle)[0]
    _assert_follow_up_stays_on_what_was_shown(first, second)


# --------------------------------------------------------------------------- п. 2: подбор анализов не о цене

_NOT_PRICE = {
    "Подскажите, есть ли у вас полный чекап здоровья?": (),  # LLM: ни одна фраза не услуга (none_fit)
    "Мне нужно сдать ВПЧ скрининг количественным методом": ("ВПЧ скрининг количественным методом",),
    "Что входит в анализ «Обязательный комплекс»?": ("Обязательный комплекс",),
}


def _assist_turns(turns: list[str], world_dir, monkeypatch, session: str, mode: str, llm: _Oracle):
    """Ходы через эндпоинт; на ход — запись эталона и выдача подбора анализов (`test_assist`: note, строки)."""

    from messengers_router.endpoint import MessengerGenerateRequest, messenger_generate_once
    from messengers_router.memory import MemoryStore
    from messengers_router.services import Services
    from tests.test_price_turns_golden import _turn_record

    _deploy(monkeypatch)
    _parse_on(monkeypatch, mode)
    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", llm)

    async def run():
        memory, services, out = MemoryStore(), Services(), []
        for text in turns:
            req = MessengerGenerateRequest(session_id=session, text=text, debug=True)
            body = json.loads((await messenger_generate_once(req, memory=memory, services=services)).body)
            items = ((body.get("state_update") or {}).get("debug") or {}).get("evidence", {}).get("items") or {}
            assist = items.get("test_assist") or {}
            picked = (assist.get("note"), [row.get("serviceName") for row in assist.get("tests") or []])
            out.append((_turn_record(body, dict(memory.get(session).last_entities or {})), picked))
        return out

    with ev_world.FrozenWorld(world_dir, "day") as world:
        result = asyncio.run(run())
        rows = list(world.telemetry)
    return result, rows


@pytest.mark.parametrize("reply", sorted(_NOT_PRICE), ids=["hpv", "checkup", "complex"])
def test_test_assist_without_a_price_question_keeps_the_rules_segmentation(reply, mini_prod_world, monkeypatch):  # noqa: F811
    """Разбор «одна услуга» на подборе анализов без вопроса о цене — не мнение о списке: выдача подбора та же,
    что под off (G8: g0162, g0282, g0313, g0359, g0416, g0604, g0693 — шаблон «уточните симптомы» вместо
    данных), а не пустая."""

    [(_, off)], _rows = _assist_turns([reply], mini_prod_world, monkeypatch, "t3-g8-off", "off", _Oracle(_NOT_PRICE[reply]))
    [(_, on)], rows = _assist_turns([reply], mini_prod_world, monkeypatch, "t3-g8-on", "on", _Oracle(_NOT_PRICE[reply]))
    assert rows[0]["tu_parse_status"] in ("picked", "none_fit") and rows[0]["label"] == "TEST_ASSIST"
    assert on == off and "multi-item" in str(on[0]) and on[1], on


@pytest.mark.parametrize("follow_up", _FOLLOW_UPS)
def test_follow_up_after_test_assist_stays_on_what_was_shown(follow_up, mini_prod_world, monkeypatch):  # noqa: F811
    reply = "Мне нужно сдать ВПЧ скрининг количественным методом"
    [(first, _), (second, _)], _rows = _assist_turns([reply, follow_up], mini_prod_world, monkeypatch, f"t3-g8-two-{follow_up}", "on", _Oracle(_NOT_PRICE[reply]))
    _assert_follow_up_stays_on_what_was_shown(first, second)


# --------------------------------------------------------------------------- п. 4: путь одной услуги и фраза разбора (отменено)
#
# Поиск по выбранной LLM слабой фразе сделан и отменён (журнал, п. 4): фраза LLM бывает короче услуги
# («удаление» из «удаление паховой грыжи», g0512) — по ней правила дают чужие строки, и выбор строки LLM
# может подать одну из них ценой. Путь одной услуги ищет по реплике, как раньше; класс g0609 открыт.

_SINGLE = "Сколько стоит и когда готов результат посева мочи с идентификацией культур"
_URINE = "Посев мочи с идентификацией культур — 1 050"


def _slot(parse, call, monkeypatch):
    from tests.test_turn_parse_readers import _run_with

    _deploy(monkeypatch)
    return _run_with(monkeypatch, parse, call)


def _picked_one(reply: str, text: str):
    from messengers_router import turn_parse
    from messengers_router.choice import Strength

    start = reply.index(text)
    phrase = turn_parse.ParsedPhrase(text=text, start=start, end=start + len(text), strength=Strength.WEAK)
    return turn_parse.PhraseParse(reply=reply, status="picked", source="llm", items=(phrase,))


@pytest.mark.xfail(strict=True, reason="T3 п. 4 отменён: путь одной услуги ищет по реплике; слова вопроса топят фразу («Уточните название») — класс g0609 открыт")
def test_one_weak_phrase_in_a_question_is_found(mini_prod_world, monkeypatch):  # noqa: F811
    [turn], _rows = _dialog([_SINGLE], mini_prod_world, monkeypatch, "t3-single-phrase", _Oracle(("посева мочи",)))
    assert _URINE in turn["text"], turn["text"]


@pytest.mark.parametrize("reader", ["price_info", "service_bundle_info"])
def test_a_too_short_phrase_does_not_replace_what_the_reply_finds(reader, mini_prod_world, monkeypatch):  # noqa: F811
    """Инвариант, ради которого п. 4 отменён: выбранная LLM фраза короче услуги («посев» из «посев мочи с
    идентификацией культур»; g0512 — «удаление» из «удаление паховой грыжи») не уводит путь одной услуги к
    чужим строкам (посев на хламидии, удаление зуба)."""

    from messengers_router.services import Services

    reply = "Сколько стоит у вас посев мочи с идентификацией культур"
    entities = {"__price_question": True, "__runtime_llm_mode": "hybrid"}
    with ev_world.FrozenWorld(mini_prod_world, "day"):
        services = Services()
        out = _slot(_picked_one(reply, "посев"), lambda: getattr(services, reader)(reply, dict(entities)), monkeypatch)
    rows = out.get("prices") or out.get("retail_prices") or out.get("family_variants") or []
    names = {row["serviceName"] for row in rows}
    assert "Посев мочи с идентификацией культур" in names and "Посев на хламидии" not in names, out


@pytest.mark.parametrize("follow_up", _FOLLOW_UPS)
def test_follow_up_after_the_one_phrase_answer_stays_on_what_was_shown(follow_up, mini_prod_world, monkeypatch):  # noqa: F811
    first, second = _dialog([_SINGLE, follow_up], mini_prod_world, monkeypatch, f"t3-single-two-{follow_up}", _Oracle(("посева мочи",)))[0]
    _assert_follow_up_stays_on_what_was_shown(first, second)


# --------------------------------------------------------------------------- п. 8: «Не распознал» без обвязки

_WRAPPED = "Стоимость анализов: ферритин, кальпротектин, лактозная непереносимость"
_WRAPPED_ORACLE = (("ферритин", "кальпротектин", "лактозная непереносимость"), ("Непереносимость лактозы", "Кальпротектин"))


def test_unrecognized_does_not_print_wrapper_words(mini_prod_world, monkeypatch):  # noqa: F811
    """«анализов» из «Стоимость анализов: …» — фрагмент по разделителям вне выбранных фраз, но услугу он не
    называет: в «Не распознал» его нет (G9: g0074, g0127, g0187, g0375, g0425, g0561)."""

    [turn], _rows = _dialog([_WRAPPED], mini_prod_world, monkeypatch, "t3-wrapper", _Oracle(*_WRAPPED_ORACLE))
    assert "Кальпротектин в кале — 2 730" in turn["text"] and "Непереносимость лактозы — 1 100" in turn["text"]
    assert "Не распознал" not in turn["text"], turn["text"]


def test_unrecognized_still_names_a_service_the_price_lacks(mini_prod_world, monkeypatch):  # noqa: F811
    """Граница класса: позиция, которую LLM разбора не выбрала, но слова которой не обвязка, — по-прежнему в
    «Не распознал» (S3: не замалчиваем)."""

    reply = "Стоимость анализов: ферритин, глюкоза, спирометрия"
    [turn], _rows = _dialog([reply], mini_prod_world, monkeypatch, "t3-wrapper-kept", _Oracle(("ферритин", "глюкоза")))
    assert "Ферритин — 530" in turn["text"] and "Не распознал: спирометрия." in turn["text"], turn["text"]


@pytest.mark.parametrize("follow_up", _FOLLOW_UPS)
def test_follow_up_after_the_trimmed_basket_stays_on_what_was_shown(follow_up, mini_prod_world, monkeypatch):  # noqa: F811
    first, second = _dialog([_WRAPPED, follow_up], mini_prod_world, monkeypatch, f"t3-wrapper-two-{follow_up}", _Oracle(*_WRAPPED_ORACLE))[0]
    _assert_follow_up_stays_on_what_was_shown(first, second)


# --------------------------------------------------------------------------- п. 7: корзина из фраз — бюджет и повторы

_REPEAT = "сколько стоит посев на грибы Candida и грибы рода Candida с чувствительностью"
_REPEAT_ORACLE = (("посев на грибы Candida", "грибы рода Candida с чувствительностью"), ("Посев на грибы рода Candida",))
_CANDIDA = "Посев на грибы рода Candida (чувствительность к антимикотическим препаратам) — 710"


def test_one_price_row_is_shown_once_in_the_basket(mini_prod_world, monkeypatch):  # noqa: F811
    """Две фразы, LLM выбрала для обеих одну строку — строка один раз, вторая позиция слита с первой (не «Не
    распознал» и не уход в путь одной услуги по всей реплике) (g0167)."""

    [turn], [row] = _dialog([_REPEAT], mini_prod_world, monkeypatch, "t3-repeat", _Oracle(*_REPEAT_ORACLE))
    assert row["tu_parse_n"] == 2 and "planner_list_to_cart" in row["tu_acted"]
    assert turn["text"].count(_CANDIDA) == 1 and "Не распознал" not in turn["text"], turn["text"]


_MANY_WEAK = "сколько стоит: витамин д, посев, папилломавирус, чекап, мазок, спирометрия, ггтп, квантовый анализ"


def test_weak_positions_without_candidates_do_not_eat_the_budget(mini_prod_world, monkeypatch):  # noqa: F811
    """Восемь слабых позиций, у трёх кандидатов нет: прежняя доля `30 // 8 = 3` резала варианты остальных
    (g0561 — ПЦР кол./п-кол. не дошли до LLM); теперь недобор переходит к позициям с кандидатами."""

    from messengers_router.services import _basket_select as BS
    from messengers_router.services._prices_helpers import _catalog_rows

    phrases = tuple(part.strip() for part in _MANY_WEAK.split(":", 1)[1].split(","))
    oracle = _Oracle(phrases, ("Посев мочи",))
    [turn], [row] = _dialog([_MANY_WEAK], mini_prod_world, monkeypatch, "t3-budget", oracle)
    assert row["tu_parse_n"] == len(phrases)
    [prompt] = oracle.rows_prompts
    with ev_world.FrozenWorld(mini_prod_world, "day"):
        catalog = _catalog_rows(None)
        from messengers_router.services._prices_helpers import phrase_price_items
        from tests.test_turn_parse_readers import _phrase

        items = phrase_price_items([_phrase(text, _MANY_WEAK) for text in phrases], catalog)
        sizes = {item["fragment"]: len(BS._candidates(item, catalog)) for item in items}
    shown = {fragment: prompt.count(f"«{fragment}» →") for fragment in phrases}
    assert sum(shown.values()) <= max(BS._MAX_OPTIONS, 2 * len(phrases))
    rich = [fragment for fragment, size in sizes.items() if size > 3]
    assert rich and all(shown[fragment] > 3 for fragment in rich), (shown, sizes)
    assert "Посев мочи с идентификацией культур" in turn["text"]


@pytest.mark.parametrize("follow_up", _FOLLOW_UPS)
def test_follow_up_after_the_merged_basket_stays_on_what_was_shown(follow_up, mini_prod_world, monkeypatch):  # noqa: F811
    first, second = _dialog([_REPEAT, follow_up], mini_prod_world, monkeypatch, f"t3-repeat-two-{follow_up}", _Oracle(*_REPEAT_ORACLE))[0]
    assert _CANDIDA in first["text"]
    _assert_follow_up_stays_on_what_was_shown(first, second)
