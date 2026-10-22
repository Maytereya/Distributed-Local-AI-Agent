"""Общая памятка о сдаче крови — вариант выбора LLM, а не запасной путь по буквам (BUG-2026-10-22-PREPARE-GENERAL-BY-LETTERS).

Симптом (стенд 22.10, набор «падежи», код прода, дважды): «Подготовка к анализу крови» → общая памятка клиники о сдаче
крови, а «Подготовка к анализам крови» → «Памятка клиники «Анализ мочи по Нечипоренко»». Корень: «анализам» нет в списке
общих слов подготовки, буквенный поиск находит мусорные карточки (гонорея, гарднерелла, Нечипоренко), а общую памятку бот
отдавал только запасным путём, когда карточек ноль; LLM выбирала наименее плохую из найденных.

Инварианты (выключатель `prepare_general_memo_llm`):
- off — как было: общей памятки среди вариантов LLM нет;
- on: LLM выбрала общую памятку — ответ общей памяткой клиники; выбрала карточку МИС — карточкой, как было; общая памятка
  вместе с карточкой — карточка;
- без LLM общая памятка не выбирается никогда (выбор без LLM — только названные карточки МИС).
"""

from __future__ import annotations

import asyncio

import pytest

from agent_logic_2 import config as cfg
from messengers_router.services import Services
from messengers_router.services import core as svc_mod
from messengers_router.services import prepare as prepare_mod
from messengers_router.services import _prepare_select_llm as select_mod
from messengers_router.services._prepare_select_llm import SOURCE_CLINIC_GENERAL, SOURCE_MIS, PrepareMemo

_NECHIPORENKO = PrepareMemo(SOURCE_MIS, "Анализ мочи по Нечипоренко", "Перед исследованием воздержаться от мочегонных.", score=0.72)
_QUESTION = "Подготовка к анализам крови"


def _run(monkeypatch, *, mode: str, pick) -> tuple[dict, list]:
    seen: list = []

    async def select(_question, memos, **_kwargs):
        seen.extend(memos)
        return pick(memos)

    monkeypatch.setattr(cfg, "MR_PREPARE_GENERAL_MEMO_LLM", mode, raising=False)
    monkeypatch.setattr(svc_mod.api_service_info, "load_service_info", lambda: [])
    monkeypatch.setattr(prepare_mod, "kb_patient_memos", lambda: ())
    monkeypatch.setattr(prepare_mod, "select_prepare_memos", select)
    svc = Services()
    svc.ensure_background_refresh_started = lambda: None
    return asyncio.run(svc.test_prepare(_QUESTION, {})), seen


def _general(memos) -> tuple:
    return tuple(m for m in memos if m.source == SOURCE_CLINIC_GENERAL)


def test_off_no_general_option(monkeypatch) -> None:
    _, seen = _run(monkeypatch, mode="off", pick=lambda memos: ())
    assert not _general(seen)


def test_on_llm_picks_general_answers_with_clinic_memo(monkeypatch) -> None:
    out, seen = _run(monkeypatch, mode="on", pick=_general)
    assert _general(seen)
    assert out["prepare"] == prepare_mod._BLOOD_COLLECTION_GUIDANCE
    assert "Нечипоренко" not in out["prepare"]


def test_on_llm_picks_mis_card_as_before(monkeypatch) -> None:
    out, _ = _run(monkeypatch, mode="on", pick=lambda memos: (_NECHIPORENKO,))
    assert "Нечипоренко" in out["prepare"]
    assert out.get("prepare_source_title") == "Анализ мочи по Нечипоренко"


def test_on_general_with_card_keeps_card(monkeypatch) -> None:
    out, _ = _run(monkeypatch, mode="on", pick=lambda memos: (_NECHIPORENKO, *_general(memos)))
    assert out.get("prepare_source_title") == "Анализ мочи по Нечипоренко"


@pytest.mark.parametrize("named", [True, False])
def test_without_llm_general_never_chosen(named: bool) -> None:
    general = PrepareMemo(SOURCE_CLINIC_GENERAL, "Общие правила сдачи крови", "Кровь сдается натощак.", score=1.0, named=named)
    assert select_mod._choose_without_llm([general]) == ()
