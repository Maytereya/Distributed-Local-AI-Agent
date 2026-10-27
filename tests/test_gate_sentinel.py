"""Тест-сторож гейта: бот работает без локальных срезов разработчика (ход (ж), решение 08.10).

До 10.10 гейт молча опирался на `agent_logic_2/nayka_api/apidata/` (вне git): нет среза врачей —
краснели 6–15 тестов (04.10), без среза «на сегодня» загрузчики шли качать у МИС. Сторож держит
два инварианта через эндпоинт, в мире стенда (`evalbench.world` — один модуль со стендом):

- на мини-прод фикстуре (`tests/fixtures/mini_prod/`, ~400 строк настоящего прайса и справочник
  разделов, решение 08.10) бот отвечает ценой из прайса, срочное — оператор, сеть не трогается,
  мир не меняется;
- на ПУСТОМ каталоге бот не падает (`route_error`/`endpoint_error`), а честно деградирует.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from evalbench import world as ev_world
from evalbench.prices import PriceCatalog

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "mini_prod" / "snapshots"
QUESTIONS = ("сколько стоит общий анализ крови", "задыхаюсь", "запишите к неврологу", "где сдать анализы", "привет")


def _ask(texts: tuple[str, ...]) -> list[dict]:
    from messengers_router.endpoint import MessengerGenerateRequest, messenger_generate_once
    from messengers_router.memory import MemoryStore
    from messengers_router.services import Services

    async def run() -> list[dict]:
        memory, services = MemoryStore(), Services()
        out = []
        for i, text in enumerate(texts):
            req = MessengerGenerateRequest(session_id=f"sentinel-{i}", text=text, debug=True)
            resp = await messenger_generate_once(req, memory=memory, services=services)
            out.append(json.loads(resp.body))
        return out

    return asyncio.run(run())


def _debug(body: dict) -> dict:
    return (body.get("state_update") or {}).get("debug") or {}


def _world(source: Path, tmp_path: Path) -> Path:
    return ev_world.build_world(source, "20261008", tmp_path / "world")


@pytest.mark.parametrize("source", ["mini_prod", "empty"])
def test_bot_answers_without_local_snapshots(source, tmp_path):
    src = FIXTURE if source == "mini_prod" else (tmp_path / "empty")
    src.mkdir(exist_ok=True)
    root = _world(src, tmp_path)
    with ev_world.FrozenWorld(root, "day") as world:
        bodies = _ask(QUESTIONS)
        attempts, cleanups = list(world.net.attempts), list(world.cleanup_calls)

    assert attempts == [] and cleanups == []
    assert ev_world.verify_world(root) == []
    for text, body in zip(QUESTIONS, bodies):
        dbg = _debug(body)
        assert body["text"].strip(), text
        assert "route_error" not in dbg and "endpoint_error" not in dbg, (text, dbg.get("route_error") or dbg.get("endpoint_error"))
    urgent = bodies[QUESTIONS.index("задыхаюсь")]
    assert urgent["handoff"] is True and urgent["text"].startswith("Соединяю с оператором.")


def test_mini_prod_price_comes_from_the_fixture_catalog(tmp_path):
    root = _world(FIXTURE, tmp_path)
    # Голое название: зонд проверяет прайс мира, а не разбор реплики. Мир гейта — LLM недоступна, и под `price_by_parse = on`
    # (прод с 26.10) цену без LLM получает только реплика-название (решение 22.10 (а)); до 27.10 тут было «сколько стоит общий
    # анализ крови» — теперь «Уточните название», без цен.
    with ev_world.FrozenWorld(root, "day"):
        (body,) = _ask(("ферритин",))
    catalog = PriceCatalog.from_world(root / "apidata")
    matched = [p for p in catalog.pairs(body["text"]) if p.status == "matched"]
    assert matched, body["text"][:200]
    assert all("ферритин" in (p.name or "").lower() for p in matched)


def test_mini_prod_fixture_is_a_small_public_extract():
    """Фикстура — выдержка прайса (300–500 строк) без служебных полей и без ПДн."""

    rows = [json.loads(line) for line in (FIXTURE / "price_by_region" / "price_region_3_20261008.jsonl").open(encoding="utf-8")]
    assert 300 <= len(rows) <= 500
    assert all(set(r) <= {"serviceId", "priceUnitId", "serviceHomecode", "serviceName", "deadline", "cost", "promotion"} for r in rows)
    assert not list(FIXTURE.rglob("doctors_*.jsonl")), "срез врачей (ФИО) в репозиторий не кладём"
