"""Подготовка к X берётся из одной памятки клиники, не из склейки (01.10, 02.10).

BUG-2026-10-01-PREPARE-FALLBACK-MIXES-DOCS: «Как подготовиться к введению спирали?» →
пункты из трёх разных документов базы знаний клиники: скрипт по спирали, скрипт по
кольпоскопии и памятка «ФКС + ФГДС с наркозом» — со слабительным и «Фортрансом».
Корень: поиск по базе (`main_index`, Meilisearch) склеивал найденные документы в один
текст без границ, и запасной путь без LLM нарезал пункты из всей склейки.

С 02.10 склейки нет: LLM выбирает ОДНУ памятку (`_prepare_select_llm`), и всё — обёртка,
запасной путь, короткий текст — работает только с ней. Скрипты администраторов (спираль,
кольпоскопия) кандидатами не становятся вовсе: решение владельца 02.10.

Инвариант класса `prepare_mixes_documents`: в ответ и в промпт обёртки попадает текст
одной выбранной памятки; памятки, которые LLM не выбрала, пациент не видит.
"""

from __future__ import annotations

import asyncio

from messengers_router.services import Services
from messengers_router.services import _common as _common_mod
from messengers_router.services import core as svc_mod
from messengers_router.services import prepare as prepare_mod
from messengers_router.services._prepare_select_llm import SOURCE_KB, PrepareMemo

_GASTRO = PrepareMemo(
    SOURCE_KB,
    "ПАМЯТКА Гастроскопия с наркозом",
    "Утром в день исследования не есть и не пить. Последний приём пищи — до 20:00 накануне.",
)
_COLONOSCOPY = PrepareMemo(
    SOURCE_KB,
    "ПАМЯТКА ФКС + ФГДС с наркозом",
    "За 3 дня до процедуры соблюдайте бесшлаковую диету. Если беспокоят запоры, выпейте "
    "любое слабительное средство. «ФОРТРАНС»: 2 литра раствора вечером перед процедурой.",
)
_FOREIGN = ("бесшлаков", "слабительн", "фортранс")


def run(coro):
    return asyncio.run(coro)


def _services(monkeypatch, *, pick: str | None, wrap_enabled: bool, gastro: PrepareMemo = _GASTRO) -> Services:
    svc = Services()
    monkeypatch.setattr(svc_mod.api_service_info, "load_service_info", lambda: [])
    monkeypatch.setattr(prepare_mod, "kb_patient_memos", lambda: (gastro, _COLONOSCOPY))

    async def select(question, memos, **kwargs):
        if pick is None:
            return ()
        return (next(memo for memo in memos if pick in memo.title),)

    monkeypatch.setattr(prepare_mod, "select_prepare_memos", select)
    monkeypatch.setattr(
        _common_mod,
        "_runtime_bool",
        lambda name, default: wrap_enabled if name == "MR_PREPARE_LLM_WRAP_ENABLED" else default,
    )
    return svc


def _long(memo: PrepareMemo) -> PrepareMemo:
    # Обёртка включается с 700 символов.
    filler = " ".join(f"Пункт {i}: возьмите паспорт и направление врача." for i in range(1, 25))
    return PrepareMemo(memo.source, memo.title, f"{memo.text} {filler}")


def test_answer_is_the_one_chosen_memo(monkeypatch):
    svc = _services(monkeypatch, pick="Гастроскопия", wrap_enabled=False)

    res = run(svc.test_prepare("Как подготовиться к гастроскопии под наркозом?", {}))

    assert res["prepare"] == _GASTRO.text
    assert res["prepare_source_title"] == _GASTRO.title


def test_fallback_after_rejected_llm_wrap_stays_within_one_memo(monkeypatch):
    # 01.10 на проде обёртка вернула негодный ответ, и запасной путь нарезал пункты
    # из склейки. Теперь запасному пути нечего смешивать.
    svc = _services(monkeypatch, pick="Гастроскопия", wrap_enabled=True, gastro=_long(_GASTRO))

    async def broken_wrap(prompt: str, **kwargs):
        return "не то"

    monkeypatch.setattr("messengers_router.services.prepare.llm_runtime_mod.generate_text", broken_wrap)

    res = run(svc.test_prepare("Как подготовиться к гастроскопии под наркозом?", {}))
    text = str(res.get("prepare") or "").lower()

    assert res.get("prepare_wrap_status") == "fallback_compact"
    assert "не есть и не пить" in text
    for foreign in _FOREIGN:
        assert foreign not in text, f"запасной путь смешал памятки: «{foreign}»"


def test_llm_wrap_sees_only_the_chosen_memo(monkeypatch):
    # До 02.10 обёртка получала склейку и сама выбирала нужное; теперь выбор сделан
    # раньше, и чужая памятка в промпт обёртки не попадает.
    svc = _services(monkeypatch, pick="Гастроскопия", wrap_enabled=True, gastro=_long(_GASTRO))
    prompts: list[str] = []

    async def wrap(prompt: str, **kwargs):
        prompts.append(prompt)
        return "не то"

    monkeypatch.setattr("messengers_router.services.prepare.llm_runtime_mod.generate_text", wrap)

    run(svc.test_prepare("Как подготовиться к гастроскопии под наркозом?", {}))

    assert prompts and "не есть и не пить" in prompts[-1].lower()
    for foreign in _FOREIGN:
        assert foreign not in prompts[-1].lower()


def test_memos_about_other_procedures_do_not_answer_for_it(monkeypatch):
    # Подходящей памятки нет — оффер оператора, а не чужая подготовка.
    svc = _services(monkeypatch, pick=None, wrap_enabled=False)

    res = run(svc.test_prepare("Как подготовиться к введению спирали?", {}))
    text = str(res.get("prepare") or "").lower()

    assert res.get("operator_offer") is True
    for foreign in (*_FOREIGN, "не есть и не пить"):
        assert foreign not in text
