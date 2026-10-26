"""«Анализы и ЭКГ — без записи»: решает LLM по правилу клиники, детектор по буквам — кандидат (класс 4, решения владельца
21.10, принципы разметки 10 и 14).

Инвариант класса под `walkin_by_llm = on` (до 26.10 — под `service_answer_from_data`; связку владелец развязал 26.10):
- ветка правил `rule_nonbookable_walkin` тему не решает — зовётся LLM-классификатор (правило — кандидат);
- переклейка после NLU в адреса (`policy_nonbookable_walkin`) — только когда LLM подтвердила «сдать анализы / сделать ЭКГ»;
- фраза «… без записи, в порядке живой очереди» в ответе адресами есть, только если LLM решила «без записи», и называет то,
  что назвала LLM (анализы, ЭКГ), а не буквы реплики;
- сбой LLM — как было (детектор); off — как было, без вызова LLM.
Через обработчик эндпоинта; LLM — подделка настоящего типа.
"""

from __future__ import annotations

import json

import pytest

from tests.test_llm_primary_rules_fallback_class import _CLASSIFIER_MARK
from tests.test_service_answer_class import _classifier, answer_world  # noqa: F401 — фикстура
from tests.test_service_answer_class import turn as _svc_turn


def turn(world_dir, monkeypatch, reply: str, *, switch: str, answers: dict, cfg_extra: dict | None = None) -> dict:
    """Ход с `walkin_by_llm = switch`; ответ по данным (`service_answer_from_data`) — off, как на проде."""

    return _svc_turn(world_dir, monkeypatch, reply, switch="off", answers=answers, cfg_extra={"MR_WALKIN_BY_LLM": switch, **(cfg_extra or {})})

_WALKIN_MARK = "хочет ли пациент сдать анализы или сделать ЭКГ"
_WALKIN_REPLIES = ["Хочу записаться на анализы", "Где можно сдать анализ кала?", "Хочу сдать кровь на гормоны, нужно записываться?", "Запишите на ЭКГ на завтра"]
_SENTENCE = "без записи, в порядке живой очереди"


def _walkin(picks: tuple[int, ...] = (), outcome: str = "none") -> str:
    return json.dumps({"picks": list(picks), "outcome": outcome})


def _answers(walkin_answer, classifier_label: str = "ADDRESS") -> dict:
    return {_CLASSIFIER_MARK: _classifier(classifier_label), _WALKIN_MARK: walkin_answer}


@pytest.mark.parametrize("reply", _WALKIN_REPLIES)
def test_off_is_as_before_without_llm(reply, answer_world, monkeypatch):  # noqa: F811
    out = turn(answer_world, monkeypatch, reply, switch="off", answers=_answers(_walkin(outcome="other")))

    assert out["telemetry"].get("nlu_rule") == "rule_nonbookable_walkin", out["telemetry"]
    assert _WALKIN_MARK not in out["calls"] and _CLASSIFIER_MARK not in out["calls"]
    assert _SENTENCE in out["text"], out["text"]  # буквы решили «без записи»


@pytest.mark.parametrize("reply", _WALKIN_REPLIES)
def test_rule_is_a_candidate_topic_is_decided_by_llm(reply, answer_world, monkeypatch):  # noqa: F811
    out = turn(answer_world, monkeypatch, reply, switch="on", answers=_answers(_walkin((1,)), classifier_label="APPOINTMENT"))

    tel = out["telemetry"]
    assert tel.get("nlu_topic_by_llm") == "llm" and tel.get("tu_label_nlu") == "APPOINTMENT", tel
    assert "rule_nonbookable_walkin" in (tel.get("nlu_rule") or ""), tel


@pytest.mark.parametrize("reply", _WALKIN_REPLIES)
def test_no_walkin_sentence_when_llm_says_otherwise(reply, answer_world, monkeypatch):  # noqa: F811
    out = turn(answer_world, monkeypatch, reply, switch="on", answers=_answers(_walkin(outcome="other")))

    assert out["telemetry"].get("walkin_llm") == "no", out["telemetry"]
    assert _SENTENCE not in out["text"], out["text"]
    assert "филиал" in out["text"], out["text"]  # адреса — по теме LLM-классификатора


@pytest.mark.parametrize(("picks", "named"), [((1,), "Анализы выполняются"), ((2,), "ЭКГ выполняются"), ((1, 2), "Анализы и ЭКГ выполняются")])
def test_what_is_walkin_is_named_by_llm_not_by_letters(picks, named, answer_world, monkeypatch):  # noqa: F811
    out = turn(answer_world, monkeypatch, "Хочу записаться на анализы", switch="on", answers=_answers(_walkin(picks)))

    assert out["text"].startswith(f"{named} без записи, в порядке живой очереди."), out["text"]


@pytest.mark.parametrize("failure", [ConnectionError("тест: LLM недоступна"), "не json"])
def test_walkin_llm_failure_keeps_detector(failure, answer_world, monkeypatch):  # noqa: F811
    reply = _WALKIN_REPLIES[0]
    off = turn(answer_world, monkeypatch, reply, switch="off", answers=_answers(_walkin(outcome="other")))
    on = turn(answer_world, monkeypatch, reply, switch="on", answers=_answers(failure))

    assert on["text"] == off["text"]
    assert on["telemetry"].get("walkin_llm") == "failed", on["telemetry"]


# «Запишите сдать ОАК»: правило — запись, детектор после NLU переклеивал её в адреса (`policy_nonbookable_walkin`).
_RELABELED = "Запишите сдать ОАК"


def test_policy_relabel_needs_llm_confirmation(answer_world, monkeypatch):  # noqa: F811
    kept = turn(answer_world, monkeypatch, _RELABELED, switch="on", answers=_answers(_walkin(outcome="booking")))
    moved = turn(answer_world, monkeypatch, _RELABELED, switch="on", answers=_answers(_walkin((1,))))
    off = turn(answer_world, monkeypatch, _RELABELED, switch="off", answers=_answers(_walkin(outcome="booking")))

    assert "policy_nonbookable_walkin:APPOINTMENT>ADDRESS" in (off["telemetry"].get("tu_relabel_src") or []), off["telemetry"]
    assert kept["telemetry"].get("tu_label_final") == "APPOINTMENT", kept["telemetry"]
    assert not kept["telemetry"].get("tu_relabel_src"), kept["telemetry"]
    assert moved["telemetry"].get("tu_label_final") == "ADDRESS", moved["telemetry"]
    assert moved["text"].startswith("Анализы выполняются без записи"), moved["text"]
    assert moved["calls"].count(_WALKIN_MARK) == 1  # решение хода спрашивается один раз


# --------------------------------------------------------------------------- класс A разбора 95 цен (26.10)
#
# Вопрос о цене или о наличии анализа, который детектор «без записи» ловит по буквам («сдать» + «анализ» / «ЭКГ»), получал
# адресный лист вместо цены или строки прайса: регулярка цены не видит «сколько будет стоить», а ветка правила решала тему до
# LLM. Под `walkin_by_llm = on` тему решает LLM-классификатор; адреса — только если LLM сказала ADDRESS. Синтетика класса.
_CLASS_A = [
    "Сколько будет стоить сдать анализ на ферритин?",
    "Хочу сдать анализ на витамин Д, сколько будет стоить?",
    "Можно ли у вас сдать кровь на пролактин?",
    "Нужно сдать анализы на гормоны: ТТГ и Т4 свободный",
    "Сдать ЭКГ сколько будет стоить?",
]
_PROD_ON = {"MR_TEST_ASSIST_FROM_DATA": "on"}


@pytest.mark.parametrize("reply", _CLASS_A)
def test_class_a_is_an_address_list_under_off(reply, answer_world, monkeypatch):  # noqa: F811
    out = turn(answer_world, monkeypatch, reply, switch="off", answers=_answers(_walkin(outcome="other")), cfg_extra=_PROD_ON)

    assert out["telemetry"].get("nlu_rule") == "rule_nonbookable_walkin", out["telemetry"]
    assert out["telemetry"].get("tu_label_final") == "ADDRESS" and _SENTENCE in out["text"], out["text"]


@pytest.mark.parametrize("label", ["TEST_ASSIST", "PRICE"])
@pytest.mark.parametrize("reply", _CLASS_A)
def test_class_a_topic_is_decided_by_llm_not_by_letters(reply, label, answer_world, monkeypatch):  # noqa: F811
    out = turn(answer_world, monkeypatch, reply, switch="on", answers=_answers(_walkin(outcome="other"), classifier_label=label), cfg_extra=_PROD_ON)

    tel = out["telemetry"]
    assert _CLASSIFIER_MARK in out["calls"] and tel.get("nlu_topic_by_llm") == "llm", tel
    assert tel.get("tu_label_final") == label, tel  # детектор после NLU метку LLM не перебил
    assert _SENTENCE not in out["text"] and "доступны филиалы" not in out["text"], out["text"]


@pytest.mark.parametrize("reply", _CLASS_A[:2])
def test_service_answer_alone_does_not_turn_walkin_on(reply, answer_world, monkeypatch):  # noqa: F811
    # Связка 21.10 развязана (решение владельца 26.10): у ответа по данным и у walk-in — свои выключатели и карточки.
    out = _svc_turn(
        answer_world, monkeypatch, reply, switch="on", answers=_answers(_walkin((1,)), classifier_label="TEST_ASSIST"), cfg_extra=_PROD_ON
    )

    assert out["telemetry"].get("nlu_rule") == "rule_nonbookable_walkin", out["telemetry"]
    assert _WALKIN_MARK not in out["calls"] and _CLASSIFIER_MARK not in out["calls"]


@pytest.mark.parametrize("path", ["messengers_router/prompts/choose_walkin.txt", "app_data/prompts/mr_choose_walkin.txt"])
def test_prompt_says_outcome_none_when_a_number_is_picked(path):
    # Карточка 26.10 (`p2on`, живая LLM): без этой строки модель отвечала {"picks": [1], "outcome": "other"} — номер «сдать
    # анализы» и исход «о другом» сразу; модуль выбора при исходе номера не берёт, и на «нужно ли записываться на анализы» фраза
    # «Анализы выполняются без записи» пропадала (g0075, g0092, g0294, g0370 …). Исход при выбранном номере — «none».
    import pathlib

    text = (pathlib.Path(__file__).resolve().parents[1] / path).read_text(encoding="utf-8")
    assert 'Если номер выбран, "outcome" — всегда "none".' in text
