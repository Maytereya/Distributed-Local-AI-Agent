"""Давность прайса в ответе — ход (д), шаг 3 (`mis_honest_degradation`; решения владельца 08.10 и 18.10).

Было: при сбое скачивания бот отдавал последний срез прайса любой давности как свежий (DATA-9; замер стенда сбоев
17.10: прайс старше 8 дней — ответ как по свежему).

Стало под `on`: срез не старше 2 дней — молча; 3–7 — приписка в конце ответа «Цены актуальны для г. …, по данным на
ДД.ММ.» (вариант «Б»); старше 7 дней — перевод `service_error_prices`.

Инвариант класса: цена из устаревшего среза не выдаётся пациенту за актуальную.
Проверка — через эндпоинт, два хода, мир мини-прода (срез 08.10), «сегодня» загрузчиков сдвинуто стендом сбоев. Разметка хода
(`turn_spans_llm = on`, прод с 24.10) — оракул `tests/spans_oracle.py` с фразой «ферритин»: при лежащей целиком LLM цену получает
только реплика-название (`price_by_parse = on`), и «сколько стоит ферритин» — «Уточните название», до давности прайса не доходит.
Выключатель `mis_honest_degradation` снят 28.10 (on на проде с 18.10): путь off удалён.
"""

from __future__ import annotations

import pytest

from evalbench import faults
from evalbench import world as ev_world
from messengers_router import llm_runtime, renderer
from messengers_router.policies import handoff_message
from tests.spans_oracle import SpansOracle, keep_llm_breaker_closed
from tests.test_price_turns_golden import mini_prod_world, run_dialog  # noqa: F401 — фикстура

_DIALOG = ["сколько стоит ферритин", "а цена?"]


def _play(world_dir, monkeypatch, stale_days: int):
    from agent_logic_2.nayka_api import snapshots

    with ev_world.FrozenWorld(world_dir, "day") as world, monkeypatch.context() as patch:
        patch.setattr(llm_runtime, "_OLLAMA_CLIENT", SpansOracle(("ферритин",)))  # фраза услуги первого хода; у «а цена?» — нет
        keep_llm_breaker_closed(patch)
        patch.setattr(snapshots, "_last", {})
        if stale_days:
            with faults.Faults(world, snapshots="stale", stale_days=stale_days):
                return run_dialog(_DIALOG, f"fresh-{stale_days}")
        return run_dialog(_DIALOG, f"fresh-{stale_days}")


@pytest.mark.parametrize(("stale_days", "as_of"), [(0, None), (2, None), (3, "08.10"), (7, "08.10")])
def test_price_age_up_to_a_week_is_answered_with_or_without_a_date(stale_days, as_of, mini_prod_world, monkeypatch):  # noqa: F811
    first, second = _play(mini_prod_world, monkeypatch, stale_days)

    assert first["handoff"] is False and "Ферритин — 530 руб." in first["text"], first["text"]
    if as_of:
        assert first["text"].rstrip().endswith(f"по данным на {as_of}._"), first["text"]
    else:
        assert "по данным на" not in first["text"] + second["text"]


def test_price_older_than_a_week_goes_to_the_operator(mini_prod_world, monkeypatch):  # noqa: F811
    first, _second = _play(mini_prod_world, monkeypatch, 8)

    assert first["handoff"] is True and handoff_message("service_error_prices") in first["text"], first["text"]
    assert "530 руб" not in first["text"]


def test_disclaimer_line_with_a_date_is_one_line_and_idempotent():
    once = renderer._append_samara_disclaimer("1) Ферритин — 530 руб.", "08.10")
    assert once.endswith("_Цены актуальны для г. Самара, по данным на 08.10._")
    assert renderer._append_samara_disclaimer(once, "08.10") == once
    assert renderer._append_samara_disclaimer("1) Ферритин — 530 руб.").endswith("_Цены актуальны для г. Самара._")
