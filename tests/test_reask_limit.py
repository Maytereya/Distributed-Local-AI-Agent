"""Предел переспросов — D2.4 (`[MESSENGER_ROUTER] reask_limit = off | on`, по умолчанию off — как было).

Петли без счётчика (аудит 14.10, пп. 2–3): «Подтвердите запись» → «нет, другое время» → тот же вопрос снова;
«Отменить текущий процесс записи?» → «да, отмените» → тот же вопрос — сколько угодно раз. Инвариант `3bacf58`
(пункт 5 принципов): шаг дважды без продвижения → назвать недостающее или оператор.

Инвариант класса (под `on`): один и тот же вопрос бот задаёт не больше двух раз подряд; третьего нет — бот называет,
какой ответ был нужен, и предлагает оператора одним вопросом, прежний вопрос снят. Первый переспрос — как раньше (не
«сразу оператор»: лавина переводов); «да» оффера — перевод, а не подтверждение снятого вопроса; срочное — первым.
Под `off` — ответы прежние.

Через обработчик эндпоинта в мире мини-прода, диалоги из 3–4 ходов (помощники — `tests/test_expects_observer.py`).
"""

from __future__ import annotations

import pytest

from messengers_router import expects
from messengers_router.policies import handoff_message
from messengers_router.recovery_policy import UNCLEAR_OPERATOR_OFFER_TEXT
from tests.test_expects_observer import (
    _ABORT_TEXT,
    _CONFIRM_REASK,
    _catalog_doctor,
    _fake_nlu,
    _play,
    _seed_collecting,
    _seed_confirm,
    anesthesia_services,  # noqa: F401 — фикстура модуля
)
from tests.test_price_turns_golden import mini_prod_world  # noqa: F401 — фикстура модуля

_NEED_YES_NO = "Не понял ответ — нужен был ответ «да» или «нет»."
_CONFIRMED = "Передаю заявку оператору для окончательного подтверждения записи"


def _mode(monkeypatch, mode: str) -> None:
    from messengers_router.runtime_config import config as cfg

    monkeypatch.setattr(cfg, "MR_REASK_LIMIT", mode, raising=False)


@pytest.mark.parametrize(("raw", "expected"), [(None, "off"), ("off", "off"), ("on", "on"), ("ON", "on"), (" on ", "on"), ("x", "off"), ("", "off")])
def test_setting_is_read_with_a_safe_default(raw, expected, monkeypatch):
    from messengers_router.runtime_config import config as cfg

    if raw is None:
        monkeypatch.delattr(cfg, "MR_REASK_LIMIT", raising=False)
    else:
        monkeypatch.setattr(cfg, "MR_REASK_LIMIT", raw, raising=False)
    assert expects.reask_limit_mode() == expected


def test_config_default_is_off():
    from agent_logic_2 import config

    assert config.MR_REASK_LIMIT == "off"


def test_limit_text_names_the_needed_answer():
    yes_no = expects.Asked(expects.Question.APPOINTMENT_ABORT, expects.YES_NO, (), True, 3)
    pick = expects.Asked(expects.Question.APPOINTMENT_ACTION, expects.PICK, expects.APPOINTMENT_ACTION_OPTIONS, True, 3)
    assert expects.reask_limit_text(yes_no) == f"{_NEED_YES_NO} {UNCLEAR_OPERATOR_OFFER_TEXT}"
    assert expects.reask_limit_text(pick) == f"Не понял ответ — нужно было выбрать: «отменить» или «перенести». {UNCLEAR_OPERATOR_OFFER_TEXT}"


# --------------------------------------------------------------------------- off — как было


@pytest.mark.parametrize(
    ("seed", "turns", "question"),
    [
        (_seed_collecting, ["нет", "да, отмените", "да, отмените же", "да, отмените уже"], _ABORT_TEXT),
        (_seed_confirm, ["нет, другое время", "нет, другое время!", "нет же, другое время"], _CONFIRM_REASK),
    ],
    ids=["abort", "confirm"],
)
def test_off_keeps_asking_as_before(seed, turns, question, mini_prod_world, monkeypatch):  # noqa: F811
    _mode(monkeypatch, "off")
    played = _play(turns, mini_prod_world, seed=seed)
    assert [t.text for t in played] == [question] * len(turns)
    assert not any(t.handoff for t in played)
    assert played[-1].expects["asks"] >= 3  # наблюдатель считает, решения не меняет


# --------------------------------------------------------------------------- on — третьего раза нет


def test_on_abort_question_twice_then_the_operator_offer(mini_prod_world, monkeypatch):  # noqa: F811
    _mode(monkeypatch, "on")
    first, second, third, fourth = _play(["нет", "да, отмените", "да, отмените же", "да"], mini_prod_world, seed=_seed_collecting)
    assert first.text == second.text == _ABORT_TEXT  # второй раз — тот же вопрос, без оффера
    assert third.text == f"{_NEED_YES_NO} {UNCLEAR_OPERATOR_OFFER_TEXT}" and not third.handoff
    assert third.expects == {"kind": "yes_no", "type": "operator_offer", "question": third.text, "options": [], "asks": 1}
    assert [row.get("answer_source") for row in third.telemetry] == ["reask_limit"]
    assert fourth.handoff and fourth.text.startswith(handoff_message("manual_operator"))


def test_on_no_to_the_offer_continues_without_the_dropped_question(mini_prod_world, monkeypatch):  # noqa: F811
    _mode(monkeypatch, "on")
    *_, limit, after = _play(["нет", "да, отмените", "да, отмените же", "нет"], mini_prod_world, seed=_seed_collecting)
    assert limit.text.endswith(UNCLEAR_OPERATOR_OFFER_TEXT)
    assert after.text == "Хорошо, продолжаем диалог. Можете задать другой вопрос." and not after.handoff
    assert after.expects is None


def test_on_confirm_loop_never_becomes_a_confirmed_booking(mini_prod_world, monkeypatch):  # noqa: F811
    """Карточка «Подтвердите» — первый раз, переспрос — второй, третьего нет; «да» оффера — не подтверждение."""

    _mode(monkeypatch, "on")
    first, second, third = _play(["нет, другое время", "нет, другое время!", "да"], mini_prod_world, seed=_seed_confirm)
    assert first.text == _CONFIRM_REASK and first.expects["asks"] == 2
    assert second.text == f"{_NEED_YES_NO} {UNCLEAR_OPERATOR_OFFER_TEXT}"
    assert third.handoff and _CONFIRMED not in third.text


def test_on_answer_on_the_second_ask_is_taken_as_before(mini_prod_world, monkeypatch):  # noqa: F811
    _mode(monkeypatch, "on")
    *_, answer = _play(["нет", "да, отмените", "да"], mini_prod_world, seed=_seed_collecting)
    assert answer.text.startswith("Процесс записи отменён") and answer.expects is None


def test_on_urgent_comes_first_where_the_limit_would_be(mini_prod_world, monkeypatch):  # noqa: F811
    _mode(monkeypatch, "on")
    *_, urgent = _play(["нет", "да, отмените", "задыхаюсь"], mini_prod_world, seed=_seed_collecting)
    assert urgent.handoff and UNCLEAR_OPERATOR_OFFER_TEXT not in urgent.text
    assert [row.get("answer_source") for row in urgent.telemetry][0].startswith("precheck:urgent")


def test_on_catalog_confirm_loop_drops_the_guess(mini_prod_world, monkeypatch):  # noqa: F811
    _mode(monkeypatch, "on")
    _catalog_doctor(monkeypatch)
    first, second, third, fourth = _play(["Запишите к Евграфовой", "именно", "именно так", "нет"], mini_prod_world)
    assert second.text == first.text
    assert third.text == f"{_NEED_YES_NO} {UNCLEAR_OPERATOR_OFFER_TEXT}"
    # «нет» — ответ на оффер, а не отказ от снятой догадки (вопроса о фамилии нет)
    assert fourth.text == "Хорошо, продолжаем диалог. Можете задать другой вопрос."


def test_on_appointment_action_loop_names_the_choice(mini_prod_world, monkeypatch):  # noqa: F811
    from messengers_router.mess_types import RouteDecision

    _mode(monkeypatch, "on")
    _fake_nlu(monkeypatch, lambda _t: RouteDecision(label="APPOINTMENT", confidence=0.9, entities={"appointment_action": "unknown"}, flags={"rule_appointment"}))
    first, second, third = _play(["что-то с визитом к врачу", "не знаю", "хз"], mini_prod_world)
    assert first.text == second.text == "Хотите отменить или перенести запись?"
    assert third.text == f"Не понял ответ — нужно было выбрать: «отменить» или «перенести». {UNCLEAR_OPERATOR_OFFER_TEXT}"
    assert third.expects["type"] == "operator_offer"


def test_on_anesthesia_reask_is_answered_by_both_memos_as_before(anesthesia_services, mini_prod_world, monkeypatch):  # noqa: F811
    from tests.test_prepare_kb_memo_variants import _WITH, _WITHOUT

    _mode(monkeypatch, "on")
    *_, both = _play(["как подготовиться к колоноскопии", "да", "нет"], mini_prod_world, services=anesthesia_services)
    assert _WITH.text in both.text and _WITHOUT.text in both.text  # свой предел у вопроса: один переспрос, потом обе
