"""Второй круг выбора строки цены ищет и по словам терминов LLM (BUG-2026-10-19-PRICE-ROUND-TWO-POOL).

Первый круг: LLM не нашла услугу среди кандидатов правил и назвала до 3 терминов, под которыми она бывает в прайсе.
Второй круг искал по каждому термину целиком: поиск правил требует всех слов сразу, и «Эзофагогастродуоденоскопия
с pH-метрией» не находит «Эзофагогастродуоденоскопия (ФГДС)». Пул выходил пустым или из пары чужих строк, LLM честно
говорила «ни одна» — и пациент получал «Не нашёл» (до 19.10 этот круг отрезал лимит хода, и прятал дефект прежний
список правил). В первом круге запасной поиск по отдельным словам есть — во втором его не было. Пустой пул к тому же
уходил в LLM: решение из лимита хода и код причины, ответ на который вариантов не было.

Инварианты класса (через обработчик эндпоинта, мини-прод, оракул LLM настоящего типа):
- термин целиком в прайсе не найден, а его слова есть в названии строки — второй круг показывает LLM эту строку;
  выбрала — цена этой строки;
- второй круг пуст и по словам — без вызова LLM: «не нашёл», вызов выбора строки ровно один, без `budget`;
- LLM не выбрала и во втором круге — честное «не нашёл», цены нет (чужая цена не возвращается).
"""

from __future__ import annotations

import json
import re

import pytest
from ollama import GenerateResponse

from evalbench import world as ev_world
from tests.test_price_turns_golden import mini_prod_world  # noqa: F401 — фикстура
from tests.test_qualifier_round_two_class import _calls, _priced, _run
from tests.test_turn_parse_t3 import _ROWS_MARKS, _Oracle

# Реплика · фраза разбора · термины первого круга (целиком их в прайсе нет) · строка мини-прайса, которую LLM выбрала бы.
# Стенд 19.10 (полный прайс): «ФГДС с определением кислотности» — термины «Эзофагогастродуоденоскопия с pH-метрией»…;
# «мазок на микрофлору с поверхности задней стенки глотки» — «Мазок из зева на микрофлору»… — пул 2 и 0 строк.
_CASES = {
    "gastro": (
        "Сколько стоит гастроскопия?",
        "гастроскопия",
        ["Эзофагогастродуоденоскопия диагностическая", "ЭГДС с pH-метрией"],
        "Эзофагогастродуоденоскопия (ФГДС)",
    ),
    "smear": (
        "сколько стоит мазок из зева на микрофлору",
        "мазок из зева на микрофлору",
        ["Бактериологический посев на микрофлору из зева", "Микробиологическое исследование отделяемого зева"],
        "Посев на микрофлору",
    ),
}


class _RoundTwo(_Oracle):
    """Разбор — фраза `phrase`; выбор строки: первый круг — «ни одна» с терминами `terms`, следующие — строки, чьё
    название начинается с `pick` (пусто — «ни одна» и там). Пулы выбора строки — в `pools`."""

    def __init__(self, phrase: str, terms: list[str], pick: str = "") -> None:
        super().__init__((phrase,))
        self.terms = terms
        self.pick = pick
        self.pools: list[list[str]] = []

    async def generate(self, *, prompt: str = "", system: str | None = None, **kwargs):
        seen = f"{system}\n\n{prompt}" if system else prompt
        if not any(mark in seen for mark in _ROWS_MARKS):
            return await super().generate(prompt=prompt, system=system, **kwargs)
        options = [(int(head), rest) for head, dot, rest in (line.partition(". ") for line in prompt.splitlines()) if dot and head.isdigit()]
        self.pools.append([text for _n, text in options])
        if len(self.pools) == 1:
            answer = {"picks": [], "reason": "other_service", "search_terms": self.terms}
        else:
            picks = [n for n, text in options if self.pick and text.startswith(self.pick)]
            answer = {"picks": picks, "reason": "none" if picks else "other_service", "search_terms": []}
        return GenerateResponse(model="m", response=json.dumps(answer, ensure_ascii=False), done=True)


def _sid(*parts: str) -> str:
    return "r2p-" + "-".join(re.sub(r"\W+", "_", part)[-14:] for part in parts)


@pytest.mark.parametrize("case", _CASES)
def test_precondition_whole_terms_miss_the_row(case, mini_prod_world):  # noqa: F811
    """Корень: поиск правил по термину целиком (как второй круг до фикса) нужной строки не находит."""

    from agent_logic_2.nayka_api import api_price

    from messengers_router.services import _price_select_llm as S
    from messengers_router.services import _prices_helpers as H

    _reply, _phrase, terms, target = _CASES[case]
    with ev_world.FrozenWorld(mini_prod_world, "day"):
        rows = [row for row in api_price.load_price_by_region(H.SAMARA_PRICE_REGION_ID) if isinstance(row, dict)]
    assert any(S._row_name(row).startswith(target) for row in rows), target
    strict = S._build_pool(terms, rows, per_text=S._PER_TERM_LIMIT)
    assert not any(S._row_name(row).startswith(target) for row in strict), [S._row_name(row) for row in strict]


@pytest.mark.parametrize("case", _CASES)
def test_round_two_offers_the_row_by_term_words(case, mini_prod_world, monkeypatch):  # noqa: F811
    """Термин целиком не найден — второй круг показывает строку по его словам; LLM выбрала её — цена этой строки
    (до фикса: пул без неё, «ни одна» → «Не нашёл»)."""

    reply, phrase, terms, target = _CASES[case]
    oracle = _RoundTwo(phrase, terms, target)
    [turn], [row] = _run([reply], mini_prod_world, monkeypatch, llm=oracle, session=_sid("pick", case))
    assert len(oracle.pools) == 2, oracle.pools
    assert any(text.startswith(target) for text in oracle.pools[1]), oracle.pools[1]
    assert target in turn["text"] and _priced(turn["text"]), turn["text"]
    assert _calls(row) == 2 and _calls(row, "budget") == 0, row.get("llm")
    assert row.get("price_policy") == "picked"


@pytest.mark.parametrize("case", _CASES)
def test_round_two_declined_is_honest_not_found(case, mini_prod_world, monkeypatch):  # noqa: F811
    """LLM не выбрала и во втором круге — «Не нашёл» без цены: чужая цена правил не возвращается."""

    reply, phrase, terms, _target = _CASES[case]
    oracle = _RoundTwo(phrase, terms)
    [turn], [row] = _run([reply], mini_prod_world, monkeypatch, llm=oracle, session=_sid("none", case))
    assert len(oracle.pools) == 2 and oracle.pools[1], oracle.pools
    assert not _priced(turn["text"]) and "Не нашёл" in turn["text"], turn["text"]
    assert row.get("price_policy") == "not_found"


def test_round_two_empty_pool_does_not_ask_llm(mini_prod_world, monkeypatch):  # noqa: F811
    """Ни термин, ни его слова в прайсе не найдены — LLM не спрашиваем: выбирать не из чего. Один вызов выбора строки,
    без `budget`, «Не нашёл» (до фикса — второй вызов на пустом списке, решение из лимита хода)."""

    reply, phrase, _terms, _target = _CASES["gastro"]
    oracle = _RoundTwo(phrase, ["Квинтэссенция зззфыв"])
    [turn], [row] = _run([reply], mini_prod_world, monkeypatch, llm=oracle, session=_sid("empty"))
    assert len(oracle.pools) == 1, oracle.pools
    assert _calls(row) == 1 and _calls(row, "budget") == 0, row.get("llm")
    assert not _priced(turn["text"]) and "Не нашёл" in turn["text"], turn["text"]
    assert row.get("price_policy") == "not_found"
