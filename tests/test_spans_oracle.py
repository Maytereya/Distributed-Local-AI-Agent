"""Контракт оракула разметки (`tests/spans_oracle.py`): его ответ проходит настоящий путь `turn_spans` — `llm_runtime` →
`_OLLAMA_CLIENT.generate` → разбор контракта — и даёт ровно заданные фразы. Иначе тесты пути прода на оракуле проверяли бы
путь сбоя разметки (`invalid`), а не прода."""

from __future__ import annotations

import asyncio
import json

import pytest
from ollama import GenerateResponse

from messengers_router import llm_runtime
from messengers_router.services import _turn_spans as S
from tests.spans_oracle import SpansOracle

_CATALOG = [
    {"serviceName": "Ферритин", "serviceHomecode": "408", "cost": 530},
    {"serviceName": "Глюкоза", "serviceHomecode": "152", "cost": 190},
    {"serviceName": "Общий анализ крови (Le, Er, Hb, СОЭ)", "serviceHomecode": "502", "cost": 390},
]


def _parse(monkeypatch, reply: str, oracle: SpansOracle):
    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", oracle)
    monkeypatch.setattr(llm_runtime, "_BREAKER", llm_runtime._Breaker())
    return asyncio.run(S.parse_turn_spans(reply, facet=None, catalog=_CATALOG))


@pytest.mark.parametrize(
    "reply, phrases",
    [
        ("сколько стоит ферритин", ("ферритин",)),
        ("Сколько стоит анализ на ферритин и глюкоза", ("анализ на ферритин", "глюкоза")),
        ("ОАК, ферритин", ("ОАК", "ферритин")),
        ("сколько стоит ферритин по ОМС", ("ферритин по омс",)),
    ],
)
def test_oracle_phrases_are_the_parse_phrases(monkeypatch, reply, phrases) -> None:
    oracle = SpansOracle(phrases)
    parse = _parse(monkeypatch, reply, oracle)
    assert oracle.asked == ["spans"]
    assert parse.status == "picked" and parse.source == "llm"
    assert [item.text.lower() for item in parse.items] == [p.lower() for p in phrases]


def test_no_phrases_is_none_and_none_is_down(monkeypatch) -> None:
    assert _parse(monkeypatch, "сколько стоят анализы", SpansOracle(())).items == ()
    down = _parse(monkeypatch, "сколько стоят анализы", SpansOracle(None))
    assert down.status == "unavailable" and down.items == ()


def test_other_places_go_to_the_inner_fake(monkeypatch) -> None:
    class Inner:
        async def generate(self, **_kwargs):
            return GenerateResponse(model="m", response=json.dumps({"label": "PRICE"}), done=True)

    oracle = SpansOracle(("ферритин",), inner=Inner())
    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", oracle)
    answer = asyncio.run(oracle.generate(prompt="классифицируй"))
    assert json.loads(answer.response) == {"label": "PRICE"} and oracle.asked == ["inner"]
