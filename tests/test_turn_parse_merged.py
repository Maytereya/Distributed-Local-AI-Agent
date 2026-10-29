"""Слияние списка в одну фразу (L-15b, шаг 6; разбор 95 ответов, G4: «витамин D + паратгормон», «Комплекс анализов кортизол,
ТТГ, Т3, Т4»): LLM выбрала всю реплику одной слабой фразой, хотя разделители режут её на ≥2 фразы и одного названия прайса на неё нет.

Статус `merged` (сбой: читатели берут разрез по разделителям) ставил только генератор окон — снят 30.10 вместе с путём off
`turn_spans_llm` (К1 карты мёртвого кода), с ним сняты тесты бюджета ответа (`phrases_task`) и статуса `merged`. У разметки прода
`merged` нет — тест через эндпоинт на пути прода остаётся `xfail(strict=True)` (находка STEP-2026-10-27-DEFAULTS-PROD).
"""

from __future__ import annotations

import re

import pytest
from evalbench import world as ev_world
from messengers_router import llm_runtime
from tests.test_price_one_or_many_class import CASES, _configure, _OracleLLM, _violations
from tests.test_price_turns_golden import mini_prod_world, run_dialog  # noqa: F401 — фикстура модуля

_PRICED = re.compile(r"^\s*\d+[.)]\s*(?:Розничная цена:\s*)?(?P<name>.+) — \d[\d\s]* руб", re.M)


class _MergingOracle(_OracleLLM):
    """Оракул строк корзины, а на выборе фраз — вся реплика одной фразой (слияние G4)."""

    def __init__(self, case, whole: str) -> None:
        super().__init__(case, (whole,))


def _priced(text: str) -> set[str]:
    return {" ".join(m.group("name").split()) for m in _PRICED.finditer(text)}


@pytest.mark.xfail(
    strict=True,
    reason="STEP-2026-10-27-DEFAULTS-PROD, находка «слияние списка разметкой»: у разметки `turn_spans` (прод с деплоя 17) нет "
    "статуса `merged` (дизайн DESIGN-2026-10-23-NO-STOPWORDS §2 его сохранял) — список, размеченный одной слабой фразой, идёт "
    "путём одной услуги: строки разных позиций одним списком (чужой цены нет)",
)
def test_merged_list_through_the_endpoint_prices_every_position(mini_prod_world, monkeypatch):  # noqa: F811
    case = CASES["list_true_three"]
    _configure(monkeypatch, "deploy_parse")
    oracle = _MergingOracle(case, "ферритин, ттг и витамин д")
    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", oracle)
    with ev_world.FrozenWorld(mini_prod_world, "day") as world:
        first, second = run_dialog([case.text, "а сколько?"], "merged-list-two-turns")
        assert list(world.net.attempts) == []
    assert "phrases" in oracle.calls
    violations = _violations(case, first["text"], "oracle", oracle.picked)
    assert not violations, "\n".join([*violations, "--- ответ:", first["text"]])
    # Два хода: короткий вопрос не берёт услугу, которой пациент не видел (контекст = показанное).
    shown = _priced(first["text"])
    unseen = sorted(_priced(second["text"]) - shown)
    assert unseen == [], "\n".join([*unseen, "--- первый ответ:", first["text"], "--- второй:", second["text"]])


