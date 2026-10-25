"""BUG-2026-10-17-P2-JUDGE-COLLECTION-CONDITIONS: вопрос о сборе / подготовке рядом с названием — не условие услуги.

Свип 16.10 (g0585): «Делается ли анализ кала ХМС по Осипову? Какая цена и условия сбора?» — судья П2 (правило 3 промпта
`choose_price_service`: «условие, которого нет ни в одной позиции») счёл «условия сбора» свойством услуги — `none_fit,
qualifier_absent` — и цена одиночной услуги стала «Не нашёл». Путь из записи — LLM-first, без словаря: уточнить правило 3.

Суд условия (`judge_qualifier`) идёт промптом `choose_price_service_qualifier` — правило 3 с этим уточнением всегда (с 22.10:
выключатель `price_candidates_by_meaning` снят); `[MESSENGER_ROUTER] price_judge_conditions = on` (умолчание off) даёт то же
уточнение выбору строки без суда (`choose_price_service_conditions`); off — прежний промпт байт в байт.

Что делает живая модель с новым правилом — замер на Mac (28 случаев `p2g_*` + синтетика). Здесь — проводка на уровне суда
условия (`select_price_rows(judge_qualifier=True)`): через эндпоинт в мире гейта до суда без живого разбора хода не дойти
(сильная фраза «ферритин» + слова рядом — решение разбора). Модель-оракул подчиняется правилу промпта, который получила
(видит уточнение — выбирает строку, нет — `qualifier_absent`), и «по ОМС» отсекает при любом промпте.
"""

from __future__ import annotations

import json

import pytest
from ollama import GenerateResponse

from evalbench import world as ev_world
from messengers_router import choice, llm_runtime
from messengers_router.choice import Option
from messengers_router.services import _price_select_llm as S
from tests.test_price_turns_golden import mini_prod_world  # noqa: F401 — фикстура

_PRICE_MARK = "ответить пациенту на вопрос о цене"
_NEW_RULE = "какие условия сбора, — не условие услуги"
_QUESTIONS = [
    ("сколько стоит ферритин и какие условия сбора?", "Ферритин"),
    ("Делается ли анализ на ферритин? Какая цена и условия сбора?", "Ферритин"),
    ("сколько стоит глюкоза и как подготовиться к сдаче", "Глюкоза"),
]


class _RuleFollowingModel:
    """Выбор строки прайса: «по омс» — условия нет (`qualifier_absent`) всегда; вопрос о сборе / подготовке — условие
    только по прежнему правилу 3 (без уточнения); иначе — строки с названием из реплики."""

    def __init__(self) -> None:
        self.saw_new_rule: list[bool] = []

    async def generate(self, *, prompt: str = "", system: str | None = None, **_kwargs):
        if _PRICE_MARK not in f"{system}\n{prompt}":
            raise ConnectionError("тест: LLM недоступна для этого места")
        new_rule = _NEW_RULE in (system or "")
        self.saw_new_rule.append(new_rule)
        question = prompt.split("Сообщение пациента: «", 1)[1].split("»", 1)[0].lower()
        asks_conditions = any(word in question for word in ("условия сбора", "подготов"))
        if "омс" in question or (asks_conditions and not new_rule):
            answer = {"picks": [], "reason": "qualifier_absent", "search_terms": []}
        else:
            picks = []
            for line in prompt.splitlines():
                head, dot, rest = line.partition(". ")
                name = rest.rsplit(" — ", 1)[0].lower() if dot and head.isdigit() else ""
                if name and name.split(" ")[0] in question:
                    picks.append(int(head))
            answer = {"picks": picks[:1], "reason": "none" if picks else "other_service", "search_terms": []}
        return GenerateResponse(model="m", response=json.dumps(answer, ensure_ascii=False), done=True)


def _judge(world_dir, monkeypatch, question: str, service: str, switch: str):
    """Суд условия П2 (`select_price_rows(judge_qualifier=True)`, как у `prices._p2_llm_verdict` и суда сильной фразы):
    кандидаты — строки названия; ответ модели-оракула."""

    import asyncio

    from messengers_router.runtime_config import config as cfg
    from messengers_router.services import _prices_helpers as H

    model = _RuleFollowingModel()
    with ev_world.FrozenWorld(world_dir, "day"), monkeypatch.context() as patch:
        patch.setattr(cfg, "MR_PRICE_JUDGE_CONDITIONS", switch, raising=False)
        patch.setattr(cfg, "MR_CHOOSE_CACHE_TTL_S", 0, raising=False)
        patch.setattr(llm_runtime, "_OLLAMA_CLIENT", model)
        patch.setattr(llm_runtime, "_BREAKER", llm_runtime._Breaker())
        choice._CACHE.clear()
        rows = H._catalog_rows(None)
        candidates = [row for row in rows if str(row.get("serviceName") or "") == service]
        selection = asyncio.run(
            S.select_price_rows(question, rows, candidates=candidates, shown=candidates[:1], hints=(service,), judge_qualifier=True)
        )
    return selection, model


@pytest.mark.parametrize(("question", "service"), _QUESTIONS)
def test_collection_question_beside_the_name_is_not_an_absent_condition(question, service, mini_prod_world, monkeypatch):  # noqa: F811
    # Суд условия идёт промптом `choose_price_service_qualifier`, в котором уточнение правила 3 уже есть (выключатель
    # `price_candidates_by_meaning` снят 22.10 — так на проде с деплоя 11): `price_judge_conditions` суда не касается.
    for mode in ("on", "off"):
        selection, model = _judge(mini_prod_world, monkeypatch, question, service, mode)
        assert model.saw_new_rule and all(model.saw_new_rule), (mode, "суд условия — промпт с уточнением правила 3")
        assert selection is not None and [str(r.get("serviceName")) for r in selection.rows] == [service], (mode, selection)


@pytest.mark.parametrize("question", ["сколько стоит ферритин по омс", "сколько стоит ферритин по омс и какие условия сбора"])
def test_oms_is_still_an_absent_condition(question, mini_prod_world, monkeypatch):  # noqa: F811
    on, _model = _judge(mini_prod_world, monkeypatch, question, "Ферритин", "on")

    assert on is not None and not on.rows and on.reason == "qualifier_absent", on  # П2: платной цены нет


def test_off_prompt_is_the_previous_prompt_byte_for_byte():
    old_system, _ = choice.render(S.PRICE_TASK, "ферритин", [Option(id="0", text="Ферритин — 530 руб.")])
    new_system, _ = choice.render(S.PRICE_CONDITIONS_TASK, "ферритин", [Option(id="0", text="Ферритин — 530 руб.")])

    assert _NEW_RULE not in old_system and _NEW_RULE in new_system
    # Отличается только правило 3 (формулировка своя — иначе отпечаток стенда путает два места вызова).
    assert [a for a, b in zip(old_system.splitlines(), new_system.splitlines()) if a != b] == [line for line in old_system.splitlines() if line.startswith("3. ")]
