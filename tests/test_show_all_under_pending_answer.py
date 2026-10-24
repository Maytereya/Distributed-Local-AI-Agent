"""BUG-2026-10-14-SHOW-ALL-AS-CATALOG-SERVICE под одной функцией ответа на висящий вопрос (`pending_reply = on`).

С 17.10 «контекст = показанное» — единственный путь (выключатель `context_shown` снят), и «все» после списка семейства
не ищет каталог по реплике. Задача B облака 19.10: проверить, что и под `pending_reply = on`, где ответ на подсказку
«напишите: "все"» разбирает одна функция (`pending_answer`, тип `show_all`), — включая ответ другими словами, который
понимает только LLM («покажите все варианты пожалуйста», «давай все»), — в контексте после раскрытия только показанное,
а «а как подготовиться?» не про «УЗИ … все зоны (детское)».

Инвариант: ответ на подсказку бота — не упоминание услуги; в контексте после хода — только показанное.
"""

from __future__ import annotations

import json

import pytest
from ollama import GenerateResponse

from messengers_router import choice, llm_runtime
from tests import test_open_bugs_2026_10_14 as bugs
from tests.test_price_turns_golden import mini_prod_world  # noqa: F401 — фикстура

_PENDING_MARK = "понять, что пациент ответил на вопрос, который бот задал ему последним"


class _PendingLLM:
    """Отвечает на разбор ответа на висящий вопрос: выбран показанный пункт «все»; прочие места — «недоступна»."""

    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, *, prompt: str = "", system: str | None = None, **_kwargs):
        if _PENDING_MARK in f"{system}\n{prompt}":
            self.calls += 1
            return GenerateResponse(model="m", response=json.dumps({"picks": [1], "outcome": "none"}), done=True)
        raise ConnectionError("тест: LLM недоступна для этого места")


@pytest.mark.parametrize(("reply", "by_llm"), [("все", False), ("Всё!", False), ("покажите все варианты пожалуйста", True), ("давай все", True)])
def test_show_all_under_one_pending_answer_holds_what_was_shown(reply, by_llm, mini_prod_world, monkeypatch):  # noqa: F811
    llm = _PendingLLM()
    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", llm)
    monkeypatch.setattr(llm_runtime, "_BREAKER", llm_runtime._Breaker())
    monkeypatch.setitem(bugs._ALL_SETS, "deploy_pending_on", {**bugs._SETS["deploy"]})
    choice._CACHE.clear()

    turns = [bugs._FAMILY_QUESTION, reply, bugs._PREPARE]
    first, second, third = bugs._play(turns, "deploy_pending_on", monkeypatch, mini_prod_world, f"show-all-pending-{reply}")
    hint = bugs._HINT.search(first["text"])
    assert hint and hint.group("word") == "все", first["text"]
    assert len(bugs._NUMBERED.findall(second["text"])) > len(bugs._NUMBERED.findall(first["text"])), second["text"]  # раскрыт весь список
    shown = {m.group("name") for m in bugs._NUMBERED.finditer(first["text"])} | {m.group("name") for m in bugs._NUMBERED.finditer(second["text"])}
    bugs._assert_context_is_what_was_shown([first, second, third], shown, bugs._QUERY.search(first["text"]).group("query"))
    if by_llm:
        assert llm.calls >= 1  # ответ другими словами понимает только LLM одной функции
