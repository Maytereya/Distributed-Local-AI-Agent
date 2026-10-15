"""Телеметрия хода (ход (а) рефакторинга, решение владельца 08.10): одна строка на ход.

Инварианты класса «на проде не видно, откуда ответ» (DEBT-1(А), DATA-13):
- через эндпоинт любой ход — ранний выход, путь пайплайна, сбой — даёт ровно одну строку
  с непустым `answer_source`;
- в строке нет текста пациента и сырого session_id (только числа и ключи);
- каждый вызов LLM попадает в строку с местом вызова и исходом, сбой — тоже;
- `nlu_llm` честно говорит, звалась ли LLM-классификация, независимо от подписи решения;
- выключатель снимает телеметрию целиком.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from ollama import GenerateResponse

from messengers_router import endpoint as endpoint_mod
from messengers_router import llm_runtime, turn_telemetry
from messengers_router.memory import MemoryStore
from messengers_router.services import Services


class _FakeOllama:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    async def generate(self, **_kwargs):
        if self.fail:
            raise ConnectionError("down")
        return GenerateResponse(model="m", response="да", done=True, prompt_eval_count=12, eval_count=1)


@pytest.fixture
def rows(monkeypatch):
    """Строки телеметрии; мир хода офлайн: LLM недоступна, дерево регионов пустое (как при сбое)."""

    from agent_logic_2.nayka_api import api_nayka, api_price

    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", _FakeOllama(fail=True))
    monkeypatch.setattr(api_nayka, "site_regions", lambda *a, **kw: [])
    monkeypatch.setattr(api_nayka, "get_all_doctors", lambda: [])  # срез врачей на «сегодня» не качаем

    def _no_download(*_a, **_kw):
        raise ConnectionError("priceUnits на «сегодня» тест не качает")

    monkeypatch.setattr(api_price.SESSION, "get", _no_download)
    captured: list[dict] = []
    monkeypatch.setattr(turn_telemetry, "_emit", captured.append)
    return captured


def _client() -> TestClient:
    services = Services()
    services.ensure_background_refresh_started = lambda: None
    memory = MemoryStore()
    app = FastAPI()
    app.include_router(endpoint_mod.router)
    app.dependency_overrides[endpoint_mod.get_memory_store] = lambda: memory
    app.dependency_overrides[endpoint_mod.get_services] = lambda: services
    return TestClient(app)


@pytest.mark.parametrize(
    "text, source",
    [
        ("задыхаюсь", "precheck:urgent"),
        ("соедините с оператором", "precheck:manual_operator"),
        ("сколько стоит общий анализ крови", None),
        ("привет", None),
    ],
)
def test_every_turn_through_endpoint_writes_one_row_without_patient_text(rows, text, source):
    body = _client().post(
        "/api/messenger-generate-once", json={"session_id": "tg_123456789", "text": text}
    ).json()

    assert body["text"]
    assert len(rows) == 1
    row = rows[0]
    assert row["answer_source"]
    if source:
        assert row["answer_source"] == source
    assert row["endpoint"] == "once" and row["handoff"] == body["handoff"]
    dumped = json.dumps(row, ensure_ascii=False)
    assert text not in dumped and "tg_123456789" not in dumped
    assert row["world_id"]
    assert all(call["outcome"] != "ok" for call in row["llm"])  # мир офлайн: LLM только падает


def test_llm_classifier_call_is_visible_even_when_decision_is_signed_llm_primary(rows):
    """DEBT-1: подпись решения бывает `llm_primary` и у правила — факт вызова пишется отдельно."""

    _client().post("/api/messenger-generate-once", json={"session_id": "s", "text": "хочу что-нибудь"})
    row = rows[0]
    assert row["nlu_llm"] is any(call["site"] == "classifier" for call in row["llm"])


def test_stream_endpoint_writes_one_row(rows):
    resp = _client().post("/api/messenger-generate", json={"session_id": "s1", "text": "задыхаюсь"})
    assert resp.status_code == 200
    assert len(rows) == 1 and rows[0]["endpoint"] == "stream" and rows[0]["handoff"] is True


@pytest.mark.parametrize("fail, outcome", [(False, "ok"), (True, "error:ConnectionError")])
def test_llm_call_is_recorded_with_site_and_outcome(monkeypatch, fail, outcome):
    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", _FakeOllama(fail=fail))
    captured: list[dict] = []
    monkeypatch.setattr(turn_telemetry, "_emit", captured.append)

    async def turn():
        token = turn_telemetry.begin("once", session_id="s")
        try:
            await llm_runtime.generate_text("промпт", timeout_s=5, queue_timeout_ms=1000)
        except ConnectionError:
            pass
        turn_telemetry.finish(token, handoff=False, chars=0)

    asyncio.run(turn())
    call = captured[0]["llm"][0]
    assert call["site"] == "test_turn_telemetry" and call["outcome"] == outcome
    if not fail:
        assert call["in_tok"] == 12


def test_switch_off_writes_nothing(rows, monkeypatch):
    monkeypatch.setattr(turn_telemetry._cfg, "MR_TURN_TELEMETRY", False, raising=False)
    _client().post("/api/messenger-generate-once", json={"session_id": "s", "text": "задыхаюсь"})
    assert rows == []


def test_a_side_row_goes_to_its_own_journal_and_merges_into_its_turn(monkeypatch, tmp_path):
    """Строка после ответа (тень `pending_reply`, 15.10): свой журнал `<вид>_<дата>.jsonl` рядом с журналом ходов —
    журнал ходов и сводки по нему не меняются; `merge_side` возвращает её поля, вызовы LLM и решения выбора в строку
    её хода по `sess` + `tid`, чужой ход не трогает."""

    monkeypatch.setattr(turn_telemetry, "_journal_path", lambda: tmp_path / "turns.jsonl")

    async def turn_with_side(session_id: str):
        token = turn_telemetry.begin("once", session_id=session_id)
        turn_telemetry.note(tu_pending_type="operator_offer")
        ref = turn_telemetry.current_ref()
        row = turn_telemetry.finish(token, handoff=False, chars=3)
        side = turn_telemetry.side_begin(ref)
        turn_telemetry.note_llm("choose_pending_reply", "ok", 1200, shadow=True)
        turn_telemetry.note_choice("pending_reply", "outcome", options=0, ms=1200, outcome="no", shadow=True)
        side_row = turn_telemetry.side_finish(side, "pending_shadow", tu_pending_outcome="no")
        return row, side_row

    (first, side), (other, _other_side) = asyncio.run(turn_with_side("s1")), asyncio.run(turn_with_side("s2"))
    assert (side["sess"], side["tid"]) == (first["sess"], first["tid"]) and first["tid"] != other["tid"]
    turns = [json.loads(line) for line in (tmp_path / "turns.jsonl").read_text(encoding="utf-8").splitlines()]
    [side_file] = list(tmp_path.glob("pending_shadow_*.jsonl"))
    sides = [json.loads(line) for line in side_file.read_text(encoding="utf-8").splitlines()]
    assert [row["tid"] for row in turns] == [first["tid"], other["tid"]] and all("kind" not in row for row in turns)
    assert [row["kind"] for row in sides] == ["pending_shadow", "pending_shadow"]
    assert first["llm"] == [] and "tu_pending_outcome" not in first  # строка хода записана до решения тени
    merged = turn_telemetry.merge_side([*turns, *sides])
    assert len(merged) == 2
    assert merged[0]["tu_pending_outcome"] == "no" and merged[0]["ms"] == first["ms"]
    assert [call["site"] for call in merged[0]["llm"]] == ["choose_pending_reply"]
    assert [entry["task"] for entry in merged[0]["choose"]] == ["pending_reply"]
