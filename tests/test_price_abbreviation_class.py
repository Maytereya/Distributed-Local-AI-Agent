"""Сокращение и синоним МИС доходят до выбора строки цены (`price_candidates_by_meaning`, test-срез 20.10) — через эндпоинт.

Test-срез 20.10 под on: «ЛГ стоимость» (g0850) и «Цена аТ ТГ» (g0794) получали «Скажите, пожалуйста, название
услуги/анализа» — планировщик решал «услуга не названа» по прежнему определению (`price_question_names_service`:
сокращения в две буквы — шум), хотя выбор строки под on уже считал сокращение с заглавной словами об услуге. Два
определения одного вопроса разошлись; теперь оно одно (`_price_select_llm.names_service`). «ЭхоКГ узи сердца» (g0501)
получал «…плода при многоплодной беременности»: строки «Эхокардиография» в списке LLM не было — буквы «эхокг» в
названиях не встречаются, а синоним МИС «ЭхоКГ» засчитывался только как сильное совпадение всей фразы.

Инварианты класса (под on; под off — как было):
- сокращение с заглавной («ЛГ», «аТ ТГ», «ФСГ», «СРБ») — названная услуга: не уточнение названия, а её строка;
- синоним МИС внутри названной услуги — её строка в списке LLM, первой (слово клиники сильнее букв правил);
- выбирает LLM: показывается только выбранная строка.
"""

from __future__ import annotations

import json

import pytest
from ollama import GenerateResponse

from evalbench import world as ev_world
from messengers_router import llm_runtime
from messengers_router.policies import CLARIFY_TEXT_MAP
from tests.test_price_turns_golden import mini_prod_world, run_dialog  # noqa: F401 — фикстура модуля

_PRICE_MARK = "ответить пациенту на вопрос о цене"
_ASK_NAME = CLARIFY_TEXT_MAP["PRICE"]


class _PickLLM:
    """LLM отвечает только на выбор строки прайса: отмечает строку с названием `target`, если её показали; списки
    позиций запоминает. Остальное — «недоступна», как в мире гейта."""

    def __init__(self, target: str) -> None:
        self.target = target
        self.offered: list[list[str]] = []

    async def generate(self, *, prompt: str = "", system: str | None = None, **_kwargs):
        if _PRICE_MARK not in f"{system}\n\n{prompt}":
            raise ConnectionError("тест: LLM недоступна для этого места")
        names, picks = [], []
        for line in prompt.splitlines():
            head, dot, rest = line.partition(". ")
            if dot and head.isdigit():
                name = " ".join(rest.rsplit(" — ", 1)[0].split())
                names.append(name)
                if name == self.target:
                    picks.append(int(head))
        self.offered.append(names)
        if system is None:
            answer = {"match": picks, "search_terms": []}
        else:
            answer = {"picks": picks, "reason": "none" if picks else "other_service", "search_terms": []}
        return GenerateResponse(model="m", response=json.dumps(answer, ensure_ascii=False), done=True)


def _turn(world_dir, monkeypatch, question: str, *, mode: str, llm: _PickLLM, synonyms: dict[str, str] | None = None) -> dict:
    from messengers_router.runtime_config import config as cfg
    from messengers_router.services import _biomaterial as bio

    monkeypatch.setattr(cfg, "MR_PRICE_CANDIDATES_BY_MEANING", mode, raising=False)
    monkeypatch.setattr(cfg, "MR_CHOOSE_CACHE_TTL_S", 0, raising=False)
    if synonyms is not None:
        real = bio._vocabularies

        def with_synonyms() -> bio._MisVocabularies:
            vocab = real()
            vocab.synonym_to_service = dict(synonyms)
            vocab.synonym_candidates = {key: (name,) for key, name in synonyms.items()}
            return vocab

        monkeypatch.setattr(bio, "_vocabularies", with_synonyms)
    with ev_world.FrozenWorld(world_dir, "day"):
        monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", llm)
        [turn] = run_dialog([question], f"abbr-{mode}-{abs(hash(question))}")
    return turn


ABBREVIATIONS = [
    ("ЛГ стоимость", "ЛГ (LH) Лютеинизирующий гормон"),  # g0850
    ("Цена аТ ТГ", "АТ - ТГ (Anti-Tg) антитела к тиреоглобулину"),  # g0794
    ("ФСГ цена", "ФСГ (FSH) Фолликулостимулирующий гормон"),
    ("СРБ сколько стоит", "Высокочувствительный С-реактивный белок (кардио), (СРБ высокочувствительный, high sensitivity CRP, hs-CRP)"),
]


@pytest.mark.parametrize(("question", "row"), ABBREVIATIONS)
def test_capitalised_abbreviation_gets_its_row_not_a_request_for_the_name(question, row, mini_prod_world, monkeypatch):  # noqa: F811
    llm = _PickLLM(row)
    turn = _turn(mini_prod_world, monkeypatch, question, mode="on", llm=llm)
    assert turn["text"] != _ASK_NAME
    assert any(row in offered for offered in llm.offered), "строка сокращения не дошла до выбора LLM"
    assert row in turn["text"] and "руб" in turn["text"]


@pytest.mark.parametrize("question", ["ЛГ стоимость", "Цена аТ ТГ"])
def test_off_keeps_asking_for_the_name_of_a_two_letter_abbreviation(question, mini_prod_world, monkeypatch):  # noqa: F811
    llm = _PickLLM("")
    turn = _turn(mini_prod_world, monkeypatch, question, mode="off", llm=llm)
    assert turn["text"] == _ASK_NAME
    assert not llm.offered


def test_bare_price_question_still_asks_for_the_name(mini_prod_world, monkeypatch):  # noqa: F811
    # Слов об услуге нет (предлоги строчными — не сокращение): уточнение, как было.
    llm = _PickLLM("")
    turn = _turn(mini_prod_world, monkeypatch, "а за сколько?", mode="on", llm=llm)
    assert " руб." not in turn["text"]
    assert not llm.offered


def test_mis_synonym_inside_the_named_service_offers_its_row_first(mini_prod_world, monkeypatch):  # noqa: F811
    llm = _PickLLM("Эхокардиография")
    turn = _turn(
        mini_prod_world, monkeypatch, "Сколько стоит ЭхоКГ узи сердца?", mode="on", llm=llm, synonyms={"эхокг": "Эхокардиография"}
    )
    assert llm.offered and llm.offered[0][0] == "Эхокардиография"
    assert "Эхокардиография —" in turn["text"] and "скрининг" not in turn["text"]
