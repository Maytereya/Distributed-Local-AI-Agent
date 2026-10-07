"""Подготовка к X берётся из одной памятки клиники, не из склейки (01.10, 02.10).

BUG-2026-10-01-PREPARE-FALLBACK-MIXES-DOCS: «Как подготовиться к введению спирали?» →
пункты из трёх разных документов базы знаний клиники: скрипт по спирали, скрипт по
кольпоскопии и памятка «ФКС + ФГДС с наркозом» — со слабительным и «Фортрансом».
Корень: поиск по базе (`main_index`, Meilisearch) склеивал найденные документы в один
текст без границ, и запасной путь без LLM нарезал пункты из всей склейки.

С 02.10 склейки нет: LLM выбирает ОДНУ памятку (`_prepare_select_llm`), и всё — обёртка,
запасной путь, короткий текст — работает только с ней. Скрипты администраторов (спираль,
кольпоскопия) кандидатами не становятся вовсе: решение владельца 02.10.

С 07.10 памятка базы знаний уходит как есть — без обёртки LLM (решение владельца: пересказ
ошибался в дозах и времени).

Инвариант класса `prepare_mixes_documents`: в ответ попадает текст одной выбранной памятки;
памятки, которые LLM не выбрала, пациент не видит.
"""

from __future__ import annotations

import asyncio

from messengers_router.services import Services
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


def _services(monkeypatch, *, pick: str | None, gastro: PrepareMemo = _GASTRO) -> Services:
    svc = Services()
    monkeypatch.setattr(svc_mod.api_service_info, "load_service_info", lambda: [])
    monkeypatch.setattr(prepare_mod, "kb_patient_memos", lambda: (gastro, _COLONOSCOPY))

    async def select(question, memos, **kwargs):
        if pick is None:
            return ()
        return (next(memo for memo in memos if pick in memo.title),)

    monkeypatch.setattr(prepare_mod, "select_prepare_memos", select)
    return svc


def _long(memo: PrepareMemo) -> PrepareMemo:
    filler = " ".join(f"Пункт {i}: возьмите паспорт и направление врача." for i in range(1, 25))
    return PrepareMemo(memo.source, memo.title, f"{memo.text} {filler}")


def test_answer_is_the_one_chosen_memo(monkeypatch):
    svc = _services(monkeypatch, pick="Гастроскопия")

    res = run(svc.test_prepare("Как подготовиться к гастроскопии под наркозом?", {}))

    assert _GASTRO.text in res["prepare"]
    assert res["prepare_source_title"] == _GASTRO.title


def test_long_memo_goes_as_is_without_llm_and_without_other_memos(monkeypatch):
    # До 07.10 длинную памятку ужимала LLM; 01.10 её негодный ответ заменял запасной путь,
    # который нарезал пункты из склейки. Теперь — текст выбранной памятки целиком.
    svc = _services(monkeypatch, pick="Гастроскопия", gastro=_long(_GASTRO))

    async def no_llm(*_args, **_kwargs):
        raise AssertionError("памятку не переписываем")

    monkeypatch.setattr("messengers_router.llm_runtime.generate_text", no_llm)

    res = run(svc.test_prepare("Как подготовиться к гастроскопии под наркозом?", {}))
    text = str(res.get("prepare") or "")

    assert res.get("prepare_wrap_status") == "not_wrapped"
    assert "Пункт 24: возьмите паспорт и направление врача." in text
    for foreign in _FOREIGN:
        assert foreign not in text.lower(), f"в ответ попала чужая памятка: «{foreign}»"


def test_memos_about_other_procedures_do_not_answer_for_it(monkeypatch):
    # Подходящей памятки нет — оффер оператора, а не чужая подготовка.
    svc = _services(monkeypatch, pick=None)

    res = run(svc.test_prepare("Как подготовиться к введению спирали?", {}))
    text = str(res.get("prepare") or "").lower()

    assert res.get("operator_offer") is True
    for foreign in (*_FOREIGN, "не есть и не пить"):
        assert foreign not in text
