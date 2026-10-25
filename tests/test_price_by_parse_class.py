"""Класс: цена читает фразу разбора хода, а не реплику, срезанную вырезателем правил (цена, этап 2, DESIGN-2026-10-25-PRICE-STAGE2).

Решения владельца 22.10: что в реплике услуга — решает LLM разметки; 28 списков стоп-слов удаляются; без LLM услугу не угадываем —
дословное название или код клиники, иначе «уточните название» / оффер. Вырезатель `_extract_price_service_from_query` (11 списков)
играл четыре роли: фраза услуги, «названа ли услуга», ключ сильного совпадения и слова поиска. Под `price_by_parse = on` их берут
из разбора хода. Поведение по ходам — эталоны `tests/golden/price_stage2_*_on.json`; здесь — инварианты на всём прайсе «мини-прод».

Инварианты под `price_by_parse = on`:
- Ш1, ключ сильного совпадения: фраза сильна, только если она сама — название, синоним МИС или код; реплика с обвязкой вопроса
  («сколько стоит <название>») сильной не бывает — ни у одного названия прайса (вырезатель делал сильными почти все);
- Ш1, число: код услуги, только если это не номер из показанного прошлым ответом (коды 1–20 в прайсе все заняты: «1» после
  списка цен было услугой с кодом 1 — находка № 8); число больше показанного — код, как было;
- Ш2, фраза услуги: в тексте пациенту в «ёлочках» — только срез его реплики (фраза разбора) или название прайса; слов реплики,
  склеенных вырезателем («стоят ферритин», «прием уролог»), и реплики целиком пациент не читает (по всем ходам эталона on);
- Ш3, «названа ли услуга»: решение понимания этой реплики (`TurnUnderstanding.names_service`), а не буквы — строчное «лг»,
  «тг» буквы не видят (ждут заглавной), и выбор строки не звался: список правил с чужой строкой при работающей LLM; понимания
  этой реплики нет — прежнее определение;
- Ш4, без LLM: разбор реплики не решён (сбой LLM) — пациент не видит ни одной цены, если не назвал услугу дословно или кодом
  (тогда её решают данные без LLM, `strong_whole`); ни списка правил с ценами («лг» внутри «аналгезии»), ни корзины правил, ни
  срока по словам; догадка заземления по буквам в контексте не остаётся (свип по прайсу мини-прод через эндпоинт);
- Ш5, журнал синонимов клинике (K7): только фраза разбора этой реплики — срез реплики; разбора нет, он не решён или решил «услуги
  нет» — записи нет; ни фразы вырезателя, ни упоминания правил понимания (контракт `_journal_phrase` и все записи эталона on);
- «без LLM» — только «LLM разметки не ответила» (`unavailable`): ответ не по контракту, письмо длиннее 40 слов, сбой кода — модель
  работает, строку выбирает LLM, «уточните» не ставится; «ни одна фраза не услуга» (`none_fit`) — не «услуги нет»: поиск по реплике,
  как раньше (карточка Ш7, 25.10: на обоих классах «уточните» и пустой поиск стоили 8 верных ответов dev из 867);
- off — как было (контраст: вырезатель делает реплику с обвязкой сильной и печатает склейки).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from messengers_router import shown_context
from messengers_router.services._strong_match import strong_rows
from tests.test_price_turns_golden import FIXTURE, mini_prod_world  # noqa: F401 — фикстура

_WRAPPED = "сколько стоит {}"


def _rows() -> list[dict[str, Any]]:
    path = next((FIXTURE / "price_by_region").glob("*.jsonl"))
    return [row for row in map(json.loads, path.read_text(encoding="utf-8").splitlines()) if isinstance(row, dict)]


ROWS = _rows()
NAMES = sorted({str(row.get("serviceName") or "").strip() for row in ROWS} - {""})
CODES = sorted({str(row.get("serviceHomecode") or "").strip() for row in ROWS if str(row.get("serviceHomecode") or "").strip().isdigit()}, key=int)


def _switch(monkeypatch, mode: str) -> None:
    from messengers_router.runtime_config import config as cfg

    monkeypatch.setattr(cfg, "MR_PRICE_BY_PARSE", mode, raising=False)


@pytest.fixture()
def shown():
    """Что показал прошлый ответ (`shown_context.previous()`): названия строк; сброс — после теста."""

    tokens = []

    def _set(names: tuple[str, ...]) -> None:
        tokens.append(shown_context._PREVIOUS.set(names))

    yield _set
    for token in reversed(tokens):
        shown_context._PREVIOUS.reset(token)


def test_every_name_is_strong_and_never_with_question_wrapper_under_on(monkeypatch):
    _switch(monkeypatch, "on")
    weak_names = [name for name in NAMES if not strong_rows(name, ROWS)]
    wrapped_strong = [name for name in NAMES if strong_rows(_WRAPPED.format(name), ROWS)]
    assert weak_names == []
    assert wrapped_strong == []


def test_wrapped_reply_was_strong_by_extractor_under_off(monkeypatch):
    _switch(monkeypatch, "off")
    wrapped_strong = [name for name in NAMES if strong_rows(_WRAPPED.format(name), ROWS)]
    # Контраст: под off вырезатель снимает обвязку — реплика сильна (на мини-проде 25.10: 185 названий из 398; у остальных он
    # калечит само название — режет до 8 слов, выбрасывает слова списков, — и сила зависит от букв, а не от данных).
    assert len(wrapped_strong) > len(NAMES) // 4


def test_number_within_shown_is_a_choice_not_a_code_under_on(monkeypatch, shown):
    _switch(monkeypatch, "on")
    small = [code for code in CODES if int(code) <= 5]
    assert small, "в мини-проде нет кодов 1–5 — тест не проверяет класс"
    shown(("Ферритин", "Глюкоза", "Холестерол", "ЭКГ", "Липидограмма"))
    assert {code: strong_rows(code, ROWS) for code in small if strong_rows(code, ROWS)} == {}
    beyond = [code for code in CODES if int(code) > 5][:20]
    assert all(strong_rows(code, ROWS) for code in beyond)


def test_number_is_a_code_when_nothing_was_shown_under_on(monkeypatch, shown):
    _switch(monkeypatch, "on")
    shown(())
    assert all(strong_rows(code, ROWS) for code in CODES[:20])


def test_turn_spans_take_a_shown_number_for_a_choice_under_on(monkeypatch, shown):
    from messengers_router.services import _turn_spans

    _switch(monkeypatch, "on")
    code = CODES[0]
    shown(tuple(NAMES[: int(code)]))
    decided = _turn_spans.decide_without_llm(code, ROWS)
    assert decided is not None and decided.status != "strong_whole"
    assert all(not item.rows for item in decided.items)
    _switch(monkeypatch, "off")
    assert _turn_spans.decide_without_llm(code, ROWS).status == "strong_whole"


# --------------------------------------------------------------------------- Ш2: фраза услуги в тексте пациенту

_GOLDEN = Path(__file__).resolve().parent / "golden"
_QUOTED = re.compile(r"«([^»]+)»")


def _norm(text: str) -> str:
    from messengers_router.services._common import _normalise_catalog_text

    return _normalise_catalog_text(text)


def _foreign_quotes(golden: str) -> dict[str, list[str]]:
    """Фразы в «ёлочках», которые не срез реплики хода и не название прайса (перечень корзины — по частям)."""

    from tests.test_price_stage2_turns import CASES

    names = {_norm(name) for name in NAMES}
    data = json.loads((_GOLDEN / golden).read_text(encoding="utf-8"))
    out: dict[str, list[str]] = {}
    for key, record in data.items():
        case = CASES[key.rsplit("-", 1)[0]]
        for reply, turn in zip(case.turns, record["turns"]):
            for quoted in _QUOTED.findall(turn["text"]):
                if not _covered(quoted.split(", "), lambda part: _norm(part) in names or _norm(part) in _norm(reply)):
                    out.setdefault(key, []).append(quoted)
    return out


def _covered(segments: list[str], ok) -> bool:
    """Перечень «A, B, C» — подряд идущие части, каждая годится (`ok`); у названия прайса бывают свои запятые."""

    reach = [True] + [False] * len(segments)
    for end in range(1, len(segments) + 1):
        reach[end] = any(reach[start] and ok(", ".join(segments[start:end])) for start in range(end))
    return reach[-1]


@pytest.mark.parametrize("mode", ["llm", "down"])
def test_patient_reads_only_reply_slices_or_catalog_names_under_on(mode):
    foreign = _foreign_quotes("price_stage2_turns_on.json")
    assert {key: quotes for key, quotes in foreign.items() if key.endswith(f"-{mode}")} == {}


def test_extractor_glued_reply_words_under_off():
    foreign = _foreign_quotes("price_stage2_turns.json")
    assert "стоят ферритин" in foreign.get("one_strong_parse_select_down-llm", [])  # контраст: склейка вырезателя на проде


# --------------------------------------------------------------------------- Ш3: «названа ли услуга» — решение понимания хода

_NAMES_CORPUS = ("сколько стоит лг", "лг сколько", "а тг почем", "сколько стоит ферритин", "сколько стоит", "а цена?")


@pytest.mark.parametrize("decided", [True, False])
def test_names_service_is_the_understanding_decision_under_on(monkeypatch, decided):
    from messengers_router import turn_understanding as TU
    from messengers_router.services import _price_select_llm as PS

    _switch(monkeypatch, "on")
    monkeypatch.setattr(TU, "of_reply", lambda text: TU.TurnUnderstanding(reply=text, names_service=decided))
    assert {reply: PS.names_service(reply) for reply in _NAMES_CORPUS} == dict.fromkeys(_NAMES_CORPUS, decided)


def test_names_service_without_understanding_keeps_letters(monkeypatch):
    from messengers_router import turn_understanding as TU
    from messengers_router.services import _price_select_llm as PS

    _switch(monkeypatch, "off")
    letters = {reply: PS.names_service(reply) for reply in _NAMES_CORPUS}
    assert letters["сколько стоит лг"] is False  # контраст: строчное сокращение буквы не видят
    _switch(monkeypatch, "on")
    monkeypatch.setattr(TU, "of_reply", lambda text: None)
    assert {reply: PS.names_service(reply) for reply in _NAMES_CORPUS} == letters


# --------------------------------------------------------------------------- Ш4: без LLM — только дословное название или код


def test_down_mode_shows_prices_only_for_verbatim_replies_under_on():
    """Эталон ходов on, LLM недоступна: цена в ответе — только у реплики, которую решили данные без LLM."""

    data = json.loads((_GOLDEN / "price_stage2_turns_on.json").read_text(encoding="utf-8"))
    priced = {
        key: turn["telemetry"].get("tu_parse_status")
        for key, record in data.items()
        if key.endswith("-down")
        for turn in record["turns"]
        if "руб." in turn["text"] and turn["telemetry"].get("tu_parse_status") not in ("strong_whole", "strong_all")
    }
    assert priced == {}


_SWEEP = NAMES[::10]  # каждое десятое название мини-прода (40 реплик): реплика с обвязкой вопроса, LLM недоступна


class _Down:
    async def generate(self, **_kwargs):
        raise ConnectionError("тест: LLM недоступна")


def test_down_sweep_no_price_without_verbatim_name_under_on(mini_prod_world, monkeypatch):  # noqa: F811
    from evalbench import world as ev_world
    from messengers_router import llm_runtime
    from messengers_router.runtime_config import config as cfg
    from tests.test_price_stage2_turns import PROD_D17
    from tests.test_price_turns_golden import run_dialog

    for key, value in {**PROD_D17, "MR_PRICE_BY_PARSE": "on"}.items():
        monkeypatch.setattr(cfg, key, value, raising=False)
    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", _Down())
    monkeypatch.setattr(llm_runtime, "_BREAKER", llm_runtime._Breaker())
    priced: dict[str, str] = {}
    guessed: dict[str, str] = {}
    with ev_world.FrozenWorld(mini_prod_world, "day") as world:
        for k, name in enumerate(_SWEEP):
            reply = _WRAPPED.format(name)
            (turn,) = run_dialog([reply], f"stage2-sweep-down-{k}")
            if "руб." in turn["text"]:
                priced[reply] = turn["text"][:120]
            if turn["service_name"] and not turn["handoff"] and "руб." not in turn["text"]:
                guessed[reply] = turn["service_name"]
        attempts = list(world.net.attempts)
    assert attempts == []
    assert priced == {}
    assert guessed == {}  # догадка заземления по буквам, которой пациент не видел, в контексте не остаётся


# --------------------------------------------------------------------------- Ш5: журнал синонимов клинике — только фраза разбора


@pytest.mark.parametrize("parsed", ["флороценоз", "анализ на флороценоз", "", None])
def test_journal_phrase_is_the_parse_phrase_only_under_on(monkeypatch, parsed):
    from messengers_router.services import _price_by_parse
    from messengers_router.services import _price_select_llm as PS

    _switch(monkeypatch, "on")
    monkeypatch.setattr(_price_by_parse, "phrase", lambda text: parsed)
    assert PS._journal_phrase("сколько стоит анализ на флороценоз", "фраза вырезателя") == (parsed or "", "price")


def test_journal_entries_are_reply_slices_under_on():
    from tests.test_price_stage2_turns import CASES

    data = json.loads((_GOLDEN / "price_stage2_turns_on.json").read_text(encoding="utf-8"))
    foreign = {
        key: entry["phrase"]
        for key, record in data.items()
        for entry in record["journal"]
        if not any(_norm(entry["phrase"]) in _norm(reply) for reply in CASES[key.rsplit("-", 1)[0]].turns)
    }
    assert foreign == {}


# --------------------------------------------------------------------------- «без LLM» и «услуги нет» — что это значит для цены


def _parse(status: str, items: tuple[str, ...] = ()):
    from messengers_router import turn_parse
    from messengers_router.choice import Strength

    phrases = tuple(turn_parse.ParsedPhrase(text=t, start=0, end=len(t), strength=Strength.WEAK, rows=()) for t in items)
    return turn_parse.PhraseParse(reply="r", status=status, source="llm", items=phrases)


@pytest.mark.parametrize(
    ("status", "items", "failed", "phrase"),
    [
        ("unavailable", (), True, None),
        ("invalid", (), False, None),
        ("merged", (), False, None),
        ("too_many", (), False, None),
        ("error", (), False, None),
        ("none_fit", (), False, None),
        ("empty", (), False, ""),
        ("picked", ("ферритин",), False, "ферритин"),
        ("picked", ("ферритин", "глюкоза"), False, None),
    ],
)
def test_without_llm_means_unavailable_and_none_fit_keeps_search(monkeypatch, status, items, failed, phrase):
    from messengers_router import turn_understanding as TU
    from messengers_router.services import _price_by_parse

    parse = _parse(status, items)
    monkeypatch.setattr(TU, "parse_status", lambda text: parse.status)
    monkeypatch.setattr(TU, "parse_for", lambda text: parse if parse.decided else None)
    assert (_price_by_parse.failed("r"), _price_by_parse.phrase("r")) == (failed, phrase)
