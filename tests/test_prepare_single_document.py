"""Без LLM подготовка к X берётся из одного документа клиники, не из склейки (01.10).

BUG-2026-10-01-PREPARE-FALLBACK-MIXES-DOCS: «Как подготовиться к введению спирали?» →
пункты из трёх разных документов базы знаний клиники: скрипт по спирали, скрипт по
кольпоскопии и памятка «ФКС + ФГДС с наркозом» — со слабительным и «Фортрансом».
Корень: поиск по базе (`main_index`, Meilisearch) склеивает найденные документы в один
текст без границ. LLM-обёртка из склейки выбирает строки о процедуре пациента (так она
верно отвечает и про ФГДС, у которой есть и отдельная памятка, и общая «ФКС + ФГДС»);
но её ответ отвергли, и запасной путь без LLM нарезал пункты из всей склейки.

Инвариант класса `prepare_mixes_documents`: склейку видит только LLM-обёртка; всё, что
отвечает без LLM (запасной путь, короткий текст, выключенная обёртка), берёт один
документ — самый подходящий к вопросу.
"""

from __future__ import annotations

import asyncio

from messengers_router.services import Services
from messengers_router.services import _common as _common_mod
from messengers_router.services import core as svc_mod

_IUD = {
    "id": "iud",
    "title": "Консультация по введению и удалению внутриматочной спирали",
    "content": (
        "Внутриматочный контрацептив можно ввести в любой день менструального цикла. "
        "Перед введением спирали пройдите осмотр гинеколога и сдайте мазок на флору. "
        "Подготовка к введению спирали: за 2 дня исключите половые контакты."
    ),
}
_COLPOSCOPY = {
    "id": "colpo",
    "title": "Скрипт по кольпоскопии",
    "content": (
        "Кольпоскопия проводится с 7 по 14 день цикла. "
        "За 24 часа до исследования не пользуйтесь кремами в области гениталий."
    ),
}
_COLONOSCOPY = {
    "id": "fks",
    "title": "ПАМЯТКА ФКС + ФГДС с наркозом",
    "content": (
        "За 3 дня до процедуры соблюдайте бесшлаковую диету. "
        "Если беспокоят запоры, выпейте любое слабительное средство. "
        "«ФОРТРАНС»: 2 литра раствора вечером перед процедурой."
    ),
}


def run(coro):
    return asyncio.run(coro)


def _services(monkeypatch, documents, *, wrap_enabled: bool) -> Services:
    svc = Services()
    monkeypatch.setattr(svc_mod.api_service_info, "load_service_info", lambda: [])
    # Как настоящий поиск: склейка найденного для основного пути и те же документы
    # по отдельности — для путей без LLM.
    glued = "\n\n".join(doc["content"] for doc in documents)
    monkeypatch.setattr(svc_mod.meilisearch, "search_meili", lambda *args, **kwargs: glued)
    monkeypatch.setattr(
        "messengers_router.services.prepare._search_main_index_documents",
        lambda query, limit=3: [dict(doc) for doc in documents],
        raising=False,  # тест воспроизводит баг и на коде до 01.10, где помощника нет
    )
    monkeypatch.setattr(svc_mod.html_cleaner, "strip_html", lambda s: s)

    def runtime_bool(name: str, default: bool) -> bool:
        if name == "MR_PREPARE_LLM_WRAP_ENABLED":
            return wrap_enabled
        if name == "MR_PREPARE_RELEVANCE_LLM_ENABLED":
            return False
        return default

    monkeypatch.setattr(_common_mod, "_runtime_bool", runtime_bool)
    return svc


def test_without_llm_wrap_answer_comes_from_the_one_document_about_the_procedure(monkeypatch):
    svc = _services(monkeypatch, [_IUD, _COLPOSCOPY, _COLONOSCOPY], wrap_enabled=False)

    res = run(svc.test_prepare("Как подготовиться к введению спирали?", {}))
    text = str(res.get("prepare") or "").lower()

    assert "спирал" in text
    for foreign in ("кольпоскоп", "слабительн", "фортранс", "бесшлаков"):
        assert foreign not in text, f"в ответ про спираль попал чужой документ: «{foreign}»"


def test_fallback_after_rejected_llm_wrap_stays_within_one_document(monkeypatch):
    # 01.10 на проде обёртка вернула негодный ответ, и запасной путь нарезал пункты
    # из склейки. Длинный документ (обёртка включается с 700 символов) — и ни одной
    # строки соседних документов в запасном ответе.
    long_iud = dict(_IUD, content=_IUD["content"] + " " + " ".join(
        f"Пункт {i}: принесите результаты исследований к приёму гинеколога." for i in range(1, 25)
    ))
    svc = _services(monkeypatch, [long_iud, _COLPOSCOPY, _COLONOSCOPY], wrap_enabled=True)

    async def broken_wrap(prompt: str, **kwargs):
        return "не то"

    monkeypatch.setattr("messengers_router.services.prepare.llm_runtime_mod.generate_text", broken_wrap)

    res = run(svc.test_prepare("Как подготовиться к введению спирали?", {}))
    text = str(res.get("prepare") or "").lower()

    assert res.get("prepare_wrap_status") == "fallback_compact"
    assert "спирал" in text or "гинеколог" in text
    for foreign in ("кольпоскоп", "слабительн", "фортранс", "бесшлаков"):
        assert foreign not in text, f"запасной путь смешал документы: «{foreign}»"


def test_documents_without_the_procedure_do_not_answer_for_it(monkeypatch):
    # Нашлись только чужие документы — честное «не нашёл правил», а не чужая подготовка.
    svc = _services(monkeypatch, [_COLPOSCOPY, _COLONOSCOPY], wrap_enabled=False)

    res = run(svc.test_prepare("Как подготовиться к введению спирали?", {}))
    text = str(res.get("prepare") or res.get("message") or "").lower()

    for foreign in ("кольпоскоп", "слабительн", "фортранс"):
        assert foreign not in text


def test_llm_wrap_still_sees_all_found_documents(monkeypatch):
    # Основной путь не меняется: обёртка получает склейку и сама выбирает нужное
    # (ФГДС — и отдельная памятка, и общая «ФКС + ФГДС»).
    long_iud = dict(_IUD, content=_IUD["content"] + " " + " ".join(
        f"Пункт {i}: принесите результаты исследований к приёму гинеколога." for i in range(1, 25)
    ))
    svc = _services(monkeypatch, [long_iud, _COLPOSCOPY, _COLONOSCOPY], wrap_enabled=True)
    prompts: list = []

    async def wrap(prompt: str, **kwargs):
        prompts.append(prompt)
        return "не то"

    monkeypatch.setattr("messengers_router.services.prepare.llm_runtime_mod.generate_text", wrap)

    run(svc.test_prepare("Как подготовиться к введению спирали?", {}))

    assert prompts and "Фортранс".lower() in prompts[-1].lower()  # склейка ушла в обёртку целиком
