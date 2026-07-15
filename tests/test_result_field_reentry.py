"""Пере-ввод данных результата с изменённым полем перезаписывает stale (прод #673).

#673: пациент прислал «Петрова, 1960, БН, 385» → ссылка, затем «Петрова, 1960,
БР, 385» (сменил филиал БН→БР) — но поля уже заполнены (missing=[]), парсер
`_fill_test_result_entities` гейтился по missing → return до парсинга → филиал
остался БН, ссылка по СТАРОМУ филиалу.

Инвариант (класс): полный ordered ре-ввод «фамилия, год, филиал, номер»
перезаписывает поля результата ВСЕГДА (явный повторный ввод всех 4), даже если
они уже заполнены. Точечный ввод (только год / только слово) — под прежним
missing-гейтом (случайное слово не перетирает заполненное поле).
"""

from __future__ import annotations

from messengers_router.policies import _fill_test_result_entities


def test_full_reentry_overwrites_filled_filial():
    out = {}
    _fill_test_result_entities("Петрова, 1960, БР, 385", [], out)  # missing=[] — данные уже были
    assert out.get("filial") == "БР", "полный ре-ввод должен перезаписать stale филиал"
    assert out.get("surname") == "Петрова"
    assert out.get("year") == 1960
    assert out.get("number") == 385


def test_full_reentry_space_form_overwrites():
    out = {}
    _fill_test_result_entities("Иванов 1985 Самара 777", [], out)
    assert out.get("filial") == "Самара"
    assert out.get("number") == 777


def test_partial_word_does_not_overwrite_when_filled():
    """Анти-регресс: одиночное слово при заполненных полях НЕ перетирает
    (только полный ordered ре-ввод перезаписывает без гейта)."""
    out = {}
    _fill_test_result_entities("Самара", [], out)  # missing=[] — точечный, не ordered
    assert out.get("filial") is None, "точечное слово при заполненных полях не перетирает"


def test_partial_still_fills_when_missing():
    """Анти-регресс: точечный ввод по-прежнему заполняет отсутствующее поле."""
    out = {}
    _fill_test_result_entities("Самара", ["filial"], out)
    assert out.get("filial") == "Самара"


def test_router_reextracts_result_fields_on_reentry():
    """Слой 2 (router elif TEST_RESULT): при заполненных полях повторный ordered
    ре-ввод со сменой филиала переизвлекается в state (герметично, rule-путь).
    """
    import asyncio

    from messengers_router import router
    from messengers_router.memory import MemoryStore
    from messengers_router.services import Services

    async def run_turn(state, services, memory, text):
        memory.append_turn(state, "user", text)
        async for _env in router.patient_routing_stream(text, state, services, memory, debug=False):
            pass

    services, memory = Services(), MemoryStore()
    state = asyncio.run(memory.aget("reentry-router"))
    asyncio.run(run_turn(state, services, memory, "Петрова, 1960, БН, 385"))
    assert state.last_entities.get("filial") == "БН"
    asyncio.run(run_turn(state, services, memory, "Петрова, 1960, БР, 385"))
    assert state.last_entities.get("filial") == "БР", "смена филиала должна переписать stale БН"
