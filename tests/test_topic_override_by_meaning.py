"""Тема реестра перебивает метку только по смыслу — подтверждает LLM (решение владельца 18.10; принцип 2).

Было: реестр тем (`topic_registry.yaml`) сравнивает основы слов в обе стороны, и «Ново-Садовая 13» совпадала с
«новость» и «новости» (основа «новос» начинается с «ново»): реплику с адресом NLU-метка OTHER меняла на акции — бот
отвечал про акции на вопрос о филиале (на живой переписке 18 реплик пациентов совпадают только по основе, почти все —
улицы). Слова решали маршрут — против принципа 2 («правила предлагают, LLM выбирает»).

Стало под `[MESSENGER_ROUTER] topic_override_llm = on`: тема перебивает OTHER только после подтверждения модуля выбора
(`choose`, задача `topic`): «ни одна», сбой или мусор — метка остаётся, тема — подсказка во флагах
(`topic_registry_llm_rejected`). Подтверждение и подсказка (без смены метки) LLM не зовут. Под off — как было.

Инварианты класса: совпадение букв реестра не меняет маршрут без подтверждения смысла; настоящие вопросы об акциях
(их решают правила) не страдают.
"""

from __future__ import annotations

import asyncio

import pytest

from evalbench import world as ev_world
from messengers_router import choice, topic_confirm
from messengers_router.choice import Choice, Option, Status
from tests.test_price_turns_golden import mini_prod_world, run_dialog  # noqa: F401 — фикстура

_STREET = ["Ново-садовая 13", "А на Ново-Садовой не делают?"]


def _play(world_dir, monkeypatch, turns, *, switch: str, verdict: Status | None = None):
    from messengers_router.runtime_config import config as cfg

    calls: list[str] = []
    with ev_world.FrozenWorld(world_dir, "day"), monkeypatch.context() as patch:
        patch.setattr(cfg, "MR_TOPIC_OVERRIDE_LLM", switch, raising=False)
        if verdict is not None:

            async def fake_choose(task, question, options, **_kw):
                calls.append(question)
                ids = (options[0].id,) if verdict is Status.PICKED else ()
                return Choice(verdict, ids=ids)

            patch.setattr(topic_confirm, "choose", fake_choose)
        records = run_dialog(turns, f"topic-{switch}-{verdict}")
    return records, calls


def test_street_name_is_not_promotions_without_the_llm(mini_prod_world, monkeypatch):  # noqa: F811
    """LLM гейта недоступна: подтверждения нет — метка OTHER остаётся, про акции бот не отвечает."""

    records, _calls = _play(mini_prod_world, monkeypatch, _STREET, switch="on")

    assert [r["label"] for r in records] == ["OTHER", "OTHER"]
    assert not any("акци" in r["text"].lower() for r in records), [r["text"] for r in records]


@pytest.mark.parametrize(("verdict", "label"), [(Status.PICKED, "NEWS"), (Status.NONE_FIT, "OTHER"), (Status.UNAVAILABLE, "OTHER")])
def test_override_follows_the_llm_verdict(verdict, label, mini_prod_world, monkeypatch):  # noqa: F811
    records, calls = _play(mini_prod_world, monkeypatch, _STREET[:1], switch="on", verdict=verdict)

    assert records[0]["label"] == label
    assert calls == [_STREET[0]]  # LLM спрашивали ровно о реплике, где тема сменила бы метку


@pytest.mark.parametrize("question", ["есть акции или новости?", "какие новости и скидки"])
def test_real_promotion_questions_are_untouched(question, mini_prod_world, monkeypatch):  # noqa: F811
    on, calls = _play(mini_prod_world, monkeypatch, [question], switch="on", verdict=Status.NONE_FIT)
    off, _ = _play(mini_prod_world, monkeypatch, [question], switch="off")

    assert on[0]["label"] == off[0]["label"] == "NEWS"
    assert calls == []  # метку решили правила — реестр её не меняет, LLM не нужна


def test_switch_off_does_not_call_the_llm(mini_prod_world, monkeypatch):  # noqa: F811
    _records, calls = _play(mini_prod_world, monkeypatch, _STREET, switch="off", verdict=Status.NONE_FIT)

    assert calls == []


def test_prompt_renders_question_and_topic():
    system, user = choice.render(topic_confirm.TOPIC_TASK, "Ново-садовая 13", [Option(id="promo_news_offers", text="Акции, новости, скидки, предложения")])

    assert "Совпадение слова — ещё не тема" in system
    assert "«Ново-садовая 13»" in user and "1. Акции, новости, скидки, предложения" in user


def test_confirms_is_false_on_any_failure(monkeypatch):
    from messengers_router.topic_registry import TopicMatch

    match = TopicMatch(topic_id="promo_news_offers", title="Акции", label="NEWS", priority=1, score=2, matched_keywords=("новость",), matched_regex=())
    for status in (Status.NONE_FIT, Status.UNAVAILABLE, Status.INVALID):

        async def fake(task, question, options, _status=status, **_kw):
            return Choice(_status)

        monkeypatch.setattr(topic_confirm, "choose", fake)
        assert asyncio.run(topic_confirm.confirms("Ново-садовая 13", match)) is False
