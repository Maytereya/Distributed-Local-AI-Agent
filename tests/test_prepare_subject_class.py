"""Название в ответе о подготовке — услуга клиники, а не кусок реплики (BUG-2026-10-20-PREPARE-SUBJECT-QUOTES-REPLY).

Карточка 20.10 (`evalbench/sets/sw_prepare_vs_booking_llm.jsonl`, `prepare_vs_booking_llm = on`): вопрос о подготовке
верно уходил в подготовку, но отказ «памятки нет» печатал в кавычках хвост реплики — «Памятки по подготовке к «узи нужно
ли не есть»», «… к «ли голодать перед узи брюшной полости»». Предмет строил вырезатель правил: всё после «подготовиться
к» или реплика без зачина, из двух фраз — самая длинная.

Инвариант класса: в кавычках ответа о подготовке — только название строки прайса по данным клиники (название, синоним
МИС) или показанная услуга; нет такой — ответ без названия. Выключатель на класс не влияет: вопрос, который и без него
шёл в подготовку («как подготовиться к узи»), держит тот же инвариант.

Через эндпоинт `messenger_generate_once`; мир — `mini_prod`; LLM — подделка настоящего типа: выбор «подготовка или
запись» — подготовка, остальные места — «недоступна».
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
from pathlib import Path

import pytest
from ollama import GenerateResponse

from evalbench import world as ev_world
from messengers_router import choice, llm_runtime
from messengers_router.services._common import _normalise_catalog_text
from messengers_router.services._prices_helpers import _catalog_rows

_TESTS = Path(__file__).resolve().parent
_PREPARE_OR_BOOKING_MARK = "есть слова и о записи, и о подготовке"
_QUOTED = re.compile(r"«([^«»]+)»")
_NO_MEMO = "у меня нет — подскажет оператор"


@pytest.fixture(scope="module")
def prepare_world(tmp_path_factory) -> Path:
    src = tmp_path_factory.mktemp("prepare_subject_src")
    shutil.copytree(_TESTS / "fixtures" / "mini_prod" / "snapshots", src, dirs_exist_ok=True)
    return ev_world.build_world(src, "20261008", tmp_path_factory.mktemp("prepare_subject") / "world")


class _LLM:
    """Выбор «подготовка или запись» — подготовка (номер 1); прочие места — «недоступна»."""

    async def generate(self, *, prompt: str = "", system: str | None = None, **_kwargs):
        if _PREPARE_OR_BOOKING_MARK in f"{system}\n\n{prompt}":
            return GenerateResponse(model="m", response=json.dumps({"picks": [1]}), done=True)
        raise ConnectionError("тест: LLM недоступна для этого места")


def _ask(world_dir: Path, monkeypatch, question: str, *, switch: str) -> tuple[str, str, set[str]]:
    """Ответ на один вопрос: (метка, текст, названия строк прайса мира — нормализованные)."""

    from messengers_router.endpoint import MessengerGenerateRequest, messenger_generate_once
    from messengers_router.memory import MemoryStore
    from messengers_router.services import Services

    async def run() -> tuple[str, str, set[str]]:
        req = MessengerGenerateRequest(session_id=f"prep-subject-{switch}-{question}", text=question, debug=True)
        body = json.loads((await messenger_generate_once(req, memory=MemoryStore(), services=Services())).body)
        label = (((body.get("state_update") or {}).get("debug") or {}).get("decision") or {}).get("label")
        names = {_normalise_catalog_text(str(row.get("serviceName") or "")) for row in _catalog_rows(None)}
        return str(label), str(body.get("text") or ""), names

    with ev_world.FrozenWorld(world_dir, "day"), monkeypatch.context() as patch:
        patch.setattr(llm_runtime, "_BREAKER", llm_runtime._Breaker())
        patch.setattr(llm_runtime, "_OLLAMA_CLIENT", _LLM())
        choice._CACHE.clear()
        return asyncio.run(run())


def _not_catalog_names(text: str, names: set[str]) -> list[str]:
    return [q for q in _QUOTED.findall(text) if _normalise_catalog_text(q) not in names]


# Карточка 20.10 (под on — подготовка) и вопросы, которые идут в подготовку и без выключателя.
_CASES = [
    ("как подготовиться к узи, нужно ли не есть", "on"),
    ("нужно ли голодать перед узи брюшной полости", "on"),
    ("нужна ли подготовка к экг", "on"),
    ("как подготовиться к гастроскопии и нужно ли записываться заранее", "on"),
    ("как подготовиться к узи", "off"),
    ("как подготовиться к кольпоскопии, нужно ли не есть", "off"),
]


@pytest.mark.parametrize(("question", "switch"), _CASES)
def test_no_memo_answer_quotes_only_a_clinic_service_name(question, switch, prepare_world, monkeypatch):
    label, text, names = _ask(prepare_world, monkeypatch, question, switch=switch)

    assert label == "PREPARE", (label, text)
    assert _NO_MEMO in text, text  # в мире гейта памяток к этим услугам нет — оффер оператора
    assert not _not_catalog_names(text, names), text


def test_service_named_by_clinic_data_is_kept(prepare_world, monkeypatch):
    """Строка прайса «ЭКГ» — название клиники, а не эхо: остаётся в ответе (нейтральный ответ — только без услуги)."""

    _label, text, _names = _ask(prepare_world, monkeypatch, "нужна ли подготовка к экг", switch="on")

    assert "«ЭКГ»" in text, text

