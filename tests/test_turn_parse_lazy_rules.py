"""Ленивые правила понимания под `turn_parse = on` (T7, шаг 4; `parse_agree_telemetry = off`) — через обработчик
эндпоинта, мини-прод.

Под `on` правила понимания хода (`understand_price_turn` → `price_list_services` → разрез списка на прайсе) считались
на каждом ценовом ходе параллельно выбору фраз LLM (O3b), а их результат нужен только запасному пути (разбор не
годится для действия: `acting_for` — None), решённому разбору одной слабой фразы (`single`: упоминание и «список ли»
— правил) и телеметрии согласия `tu_parse_agree`. Под `parse_agree_telemetry = off` согласия нет, правила — лениво.

Инварианты класса: (а) ответ хода, состояние сессии и понимание хода (`tu_list`, `tu_facet`, `tu_rule`,
`tu_mention_*`) — те же, что при правилах параллельно разбору, при живой LLM (оракул) и при недоступной
(базовый эталон пути цены без изменений); (б) разбор решён не одной слабой фразой — правила не
считаются вовсе; (в) запасной путь (`empty`, сбой LLM без разделителей) и `single` — считаются, ответ прежний;
(г) по умолчанию (`on`) — как было: правила на каждом ходе, согласие пишется; (д) два хода — короткий вопрос
после корзины не получает цену услуги, которой пациент не видел.
"""

from __future__ import annotations

import pytest

from evalbench import world as ev_world
from messengers_router import llm_runtime, turn_understanding
from tests.test_price_one_or_many_class import CASES, TRUE_CUT, _configure, _OracleLLM, _priced_sections
from tests.test_price_turns_golden import DIALOGS, _golden, mini_prod_world, run_dialog, synonym_fork  # noqa: F401 — фикстуры

_TU_FIELDS = ("tu_list", "tu_facet", "tu_rule", "tu_mention_src", "tu_mention_strength", "tu_parse_status", "tu_parse_src", "tu_parse_n")


def _agree(monkeypatch, value: str) -> None:
    from messengers_router.runtime_config import config as cfg

    monkeypatch.setattr(cfg, "MR_PARSE_AGREE_TELEMETRY", value, raising=False)


class _Calls:
    """Счётчик правил понимания: `understand_price_turn` и его разрез списка (`price_list_services`)."""

    def __init__(self, monkeypatch) -> None:
        self.understand = 0
        self.lists = 0
        real_understand, real_lists = turn_understanding.understand_price_turn, turn_understanding.price_list_services

        def understand(text):
            self.understand += 1
            return real_understand(text)

        def lists(*args, **kwargs):
            self.lists += 1
            return real_lists(*args, **kwargs)

        monkeypatch.setattr(turn_understanding, "understand_price_turn", understand)
        monkeypatch.setattr(turn_understanding, "price_list_services", lists)


def _run(turns, *, agree: str, llm: str, case_name: str, world_dir, monkeypatch, session: str):
    """Диалог под набором деплоя с разбором (`deploy_parse`); `llm` — oracle | down. Возврат: ходы, строки
    телеметрии, счётчик правил."""

    with monkeypatch.context() as patch:
        _configure(patch, "deploy_parse")
        _agree(patch, agree)
        if llm == "oracle":
            patch.setattr(llm_runtime, "_OLLAMA_CLIENT", _OracleLLM(CASES[case_name], TRUE_CUT[case_name]))
        calls = _Calls(patch)
        with ev_world.FrozenWorld(world_dir, "day") as world:
            record = run_dialog(turns, session)
            rows, attempts = list(world.telemetry), list(world.net.attempts)
    assert attempts == []
    return record, rows, calls


def _tu(row: dict) -> dict:
    return {key: row.get(key) for key in _TU_FIELDS}


# --------------------------------------------------------------------------- (а) ответы и понимание — те же


@pytest.mark.parametrize("llm", ["oracle", "down"])
@pytest.mark.parametrize("name", sorted(CASES))
def test_lazy_rules_change_no_answer_and_no_understanding(name, llm, mini_prod_world, monkeypatch):  # noqa: F811
    turns = [CASES[name].text]
    eager, eager_rows, _ = _run(turns, agree="on", llm=llm, case_name=name, world_dir=mini_prod_world, monkeypatch=monkeypatch, session=f"lazy-a-{name}-{llm}-on")
    lazy, lazy_rows, _ = _run(turns, agree="off", llm=llm, case_name=name, world_dir=mini_prod_world, monkeypatch=monkeypatch, session=f"lazy-a-{name}-{llm}-off")
    assert lazy == eager
    assert [_tu(row) for row in lazy_rows] == [_tu(row) for row in eager_rows]
    assert all("tu_parse_agree" not in row for row in lazy_rows)


@pytest.mark.parametrize("case", sorted(DIALOGS))
def test_lazy_rules_keep_the_parse_on_golden(case, mini_prod_world, request, monkeypatch):  # noqa: F811
    """Эталон `turn_parse = on` при недоступной LLM (`test_turn_parse_readers`) — без изменений."""

    from messengers_router.runtime_config import config as cfg

    # Эталон `turn_parse = on` — при понимании хода в тени: режим явно, а не умолчанием кода
    # (STEP-2026-10-15-DEFAULTS-A).
    monkeypatch.setattr(cfg, "MR_TURN_UNDERSTANDING_PRICE", "shadow", raising=False)
    _agree(monkeypatch, "off")
    if case.startswith("synonym_fork"):
        request.getfixturevalue("synonym_fork")
    with ev_world.FrozenWorld(mini_prod_world, "day") as world:
        record = run_dialog(DIALOGS[case], f"lazy-golden-{case}")
        attempts = list(world.net.attempts)
    assert attempts == []
    assert record == _golden()[case]


# --------------------------------------------------------------------------- (б), (в), (г) когда правила считаются


@pytest.mark.parametrize(
    "name, status",
    [("list_true_three", "picked"), ("name_parens_commas", "strong_whole"), ("phrase_smear_pharynx", "picked")],
)
def test_decided_parse_skips_the_rules(name, status, mini_prod_world, monkeypatch):  # noqa: F811
    [_turn], [row], calls = _run([CASES[name].text], agree="off", llm="oracle", case_name=name, world_dir=mini_prod_world, monkeypatch=monkeypatch, session=f"lazy-b-{name}")
    assert row["tu_parse_status"] == status
    assert (calls.understand, calls.lists) == (0, 0)
    assert row["tu_rule"] == f"parse_{status}" and row["tu_facet"] == "price"


def test_by_default_the_rules_run_every_price_turn_and_the_agreement_is_written(mini_prod_world, monkeypatch):  # noqa: F811
    name = "list_true_three"
    [_turn], [row], calls = _run([CASES[name].text], agree="on", llm="oracle", case_name=name, world_dir=mini_prod_world, monkeypatch=monkeypatch, session="lazy-g")
    assert row["tu_parse_status"] == "picked"
    assert calls.understand == 1 and calls.lists >= 1
    assert row.get("tu_parse_agree") in ("same", "split", "merge", "other")


@pytest.mark.parametrize("text", [CASES["list_caps_run_on_bare"].text, CASES["list_caps_run_on_priced"].text])
def test_failed_parse_without_separators_acts_without_the_rules(text, mini_prod_world, monkeypatch):  # noqa: F811
    """Сбой LLM без разделителей с 16.10 (Q2 отменено, T7 шаг 3) — разрез данными МИС или реплика целиком: разбор действует
    сам, ленивые правила не считаются, ответ тот же, что при правилах на каждом ходе (они там — только телеметрия)."""

    name = "list_caps_run_on_bare"
    eager, [eager_row], _ = _run([text], agree="on", llm="down", case_name=name, world_dir=mini_prod_world, monkeypatch=monkeypatch, session="lazy-q2-on")
    lazy, [row], calls = _run([text], agree="off", llm="down", case_name=name, world_dir=mini_prod_world, monkeypatch=monkeypatch, session="lazy-q2-off")
    assert row["tu_parse_status"] == "unavailable" and row["tu_rule"] == "parse_unavailable"
    assert calls.understand == 0
    assert lazy == eager


@pytest.mark.parametrize("text, llm, status", [("сколько стоит?", "oracle", "empty")])  # пустой разбор — запасной путь
def test_fallback_runs_the_rules_after_the_parse_and_answers_as_before(text, llm, status, mini_prod_world, monkeypatch):  # noqa: F811
    name = "list_caps_run_on_bare"
    eager, [eager_row], _ = _run([text], agree="on", llm=llm, case_name=name, world_dir=mini_prod_world, monkeypatch=monkeypatch, session=f"lazy-c-{status}-on")
    lazy, [row], calls = _run([text], agree="off", llm=llm, case_name=name, world_dir=mini_prod_world, monkeypatch=monkeypatch, session=f"lazy-c-{status}-off")
    assert row["tu_parse_status"] == status
    assert calls.understand == 1
    assert lazy == eager and _tu(row) == _tu(eager_row)


def test_single_weak_phrase_takes_the_rules_mention(mini_prod_world, monkeypatch):  # noqa: F811
    """`single` — кандидат один, слабый: упоминание и «список ли» — правил, поэтому правила считаются."""

    text = "сколько стоит ттг"
    eager, [eager_row], _ = _run([text], agree="on", llm="down", case_name="list_true_three", world_dir=mini_prod_world, monkeypatch=monkeypatch, session="lazy-s-on")
    lazy, [row], calls = _run([text], agree="off", llm="down", case_name="list_true_three", world_dir=mini_prod_world, monkeypatch=monkeypatch, session="lazy-s-off")
    assert row["tu_parse_status"] == "single"
    assert calls.understand == 1
    assert lazy == eager and _tu(row) == _tu(eager_row)


def test_rules_failure_on_the_fallback_does_not_break_the_turn(mini_prod_world, monkeypatch):  # noqa: F811
    def broken(_text):
        raise RuntimeError("тест: правила упали")

    with monkeypatch.context() as patch:
        _configure(patch, "deploy_parse")
        _agree(patch, "off")
        patch.setattr(turn_understanding, "understand_price_turn", broken)
        with ev_world.FrozenWorld(mini_prod_world, "day") as world:
            [turn] = run_dialog(["сколько стоит?"], "lazy-fail")
            row = world.take_telemetry()
    assert turn["text"] and row.get("tu_rule") == "error" and row.get("tu_parse_status") == "empty"


# --------------------------------------------------------------------------- (д) два хода


@pytest.mark.parametrize("follow_up", ["а сколько?", "да", "2"])
@pytest.mark.parametrize("llm", ["oracle", "down"])
def test_two_turns_follow_up_after_the_basket_is_as_before(llm, follow_up, mini_prod_world, monkeypatch):  # noqa: F811
    name = "list_true_three"
    turns = [CASES[name].text, follow_up]
    eager, _, _ = _run(turns, agree="on", llm=llm, case_name=name, world_dir=mini_prod_world, monkeypatch=monkeypatch, session=f"lazy-d-{llm}-{follow_up}-on")
    lazy, _, calls = _run(turns, agree="off", llm=llm, case_name=name, world_dir=mini_prod_world, monkeypatch=monkeypatch, session=f"lazy-d-{llm}-{follow_up}-off")
    assert lazy == eager
    first, second = lazy
    shown = {row for section in _priced_sections(first["text"]) for row in section}
    assert {row for section in _priced_sections(second["text"]) for row in section} <= shown, (second["text"], shown)
    if llm == "oracle":
        assert calls.understand <= 1  # первый ход решён разбором — правил нет; второй — не ценовой или свой
