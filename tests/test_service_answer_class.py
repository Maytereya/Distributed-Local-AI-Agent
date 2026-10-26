"""Ответ о клинике — только по её данным (решения владельца 21.10: «ни в коем случае не выдумывать»).

Класс BUG-2026-10-21-FACT-WITHOUT-DATA: на вопрос «вы берёте / делаете / принимаете / есть ли у вас …» свободный текст LLM
утверждал факт о клинике, которого нет в данных («Вы берёте анализы крови у детей 2 лет?» → «Да, мы проводим анализы
крови для детей», g0509 — так отвечал прод). Инвариант: под `service_answer_from_data = on` утверждение о клинике — только
из пунктов данных (прайс, памятки), выбранных LLM; данных нет — честный оффер оператора; «не о клинике» и сбой LLM —
как было. Под off — как было. Свободного текста на ходе подбора анализов на проде нет и под off: с 24.10
(`test_assist_from_data = on`) ход отвечает по данным хода (`answer_source = test_assist_data`) — он же запасной путь при сбое и
«не о клинике» (до переворота умолчаний 27.10 здесь был свободный текст, и подделка рендера «выдумывала»).

Через обработчик эндпоинта; LLM — подделка настоящего типа (`GenerateResponse`, поток — как у ollama).
"""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from ollama import GenerateResponse

from evalbench import world as ev_world
from messengers_router import choice, llm_runtime
from tests.test_llm_primary_rules_fallback_class import _CLASSIFIER_MARK

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "mini_prod" / "snapshots"
_DOCTORS = Path(__file__).resolve().parent / "fixtures" / "doctors_excerpt" / "doctors_20261008.jsonl"
_SERVICE_ANSWER_MARK = "ответить на вопрос пациента о клинике ТОЛЬКО по данным клиники"
_RENDER_MARK = "Ты помощник клиники в мессенджере"
FABRICATED = "Да, мы проводим это в нашей клинике для всех возрастов."

# Строки прайса и памятка, которых нет в выдержке mini_prod: g0277 «Делают ли ФКС с биопсией?» (данные — синтетика по
# образцу среза: название и цена строки прайса, строка памятки).
_EXTRA_ROWS = [
    {"serviceId": 990001, "priceUnitId": 291, "serviceHomecode": "30.1.2.4", "serviceName": "Биопсия при ФКС (без стоимости исследования)", "deadline": " ", "cost": 1200.0, "promotion": 0},
]
_KB = [
    {
        "title": "ПАМЯТКА ФКС без наркоза",
        "content": "<p>Подготовка к исследованию.</p><p>Дополнительно, в процессе процедуры, можно провести манипуляции:</p>"
        "<p>1. Биопсия при КС (без стоимости исследования) - 900 руб.</p><p>2. Полипэктомия 1 категория - 1500 руб.</p>",
    },
    {"title": "Скрипт для администраторов", "content": "<p>Биопсия: что говорить пациенту.</p>"},
]


# Карточки справочника услуг МИС (синтетика по образцу среза): что это и подготовка — для ответа на голое название.
SERVICE_INFO = [
    {
        "serviceId": 6278,
        "serviceHomecode": "247",
        "serviceName": "T-SPOT (диагностика латентной и активной туберкулезной инфекции у детей и взрослых)",
        "description": "<p>Тест выявляет туберкулёзную инфекцию по реакции клеток крови. Подходит детям и взрослым.</p>",
        "preparation": "<p>Кровь сдаётся утром. Накануне исключить жирную пищу.</p>",
        "biomatNames": "кровь из вены",
    },
]


@pytest.fixture(scope="module")
def answer_world(tmp_path_factory) -> Path:
    source = tmp_path_factory.mktemp("svc_src")
    for sub in ("price_by_region", "price_units"):
        (source / sub).mkdir()
        for path in (_FIXTURE / sub).iterdir():
            text = path.read_text(encoding="utf-8")
            if sub == "price_by_region":
                text = text.rstrip("\n") + "\n" + "\n".join(json.dumps(r, ensure_ascii=False) for r in _EXTRA_ROWS) + "\n"
            (source / sub / path.name).write_text(text, encoding="utf-8")
    (source / "service_info").mkdir()
    (source / "service_info" / "service_info_20261008.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in SERVICE_INFO) + "\n", encoding="utf-8"
    )
    shutil.copy2(_DOCTORS, source / _DOCTORS.name)  # врачи — выдержка из репо (путь «врачей по услуге не нашёл»)
    root = ev_world.build_world(source, "20261008", tmp_path_factory.mktemp("svc_world") / "world")
    (root / "captured").mkdir()
    (root / "captured" / "kb_main_index.json").write_text(json.dumps(_KB, ensure_ascii=False), encoding="utf-8")
    return root


def _branch(idx: int, addr: str, **kinds: bool) -> dict[str, Any]:
    return {"id": idx, "name": f"Самара {addr}", "addressForSite": f"г. Самара, {addr}", "city": "Самара", "analysis": True, **kinds}


# Живой /regions — синтетика: два лабораторных филиала (один с ЭКГ), чтобы ответ адресами был полным.
_REGIONS = [_branch(1, "ул.Тестовая, 1", ecg=True), _branch(2, "ул.Пробная, 2")]


class _LLM:
    """Подделка клиента ollama: ответ по метке промпта (строка или функция данных хода); поток — одним куском."""

    def __init__(self, answers: dict[str, Any]):
        self.answers = answers
        self.calls: list[str] = []
        self.prompts: dict[str, str] = {}

    def _answer(self, seen: str, prompt: str) -> str:
        for mark, answer in self.answers.items():
            if mark in seen:
                self.calls.append(mark)
                self.prompts[mark] = prompt
                if isinstance(answer, BaseException):
                    raise answer
                return answer(prompt) if callable(answer) else answer
        raise ConnectionError("тест: LLM недоступна для этого места")

    async def generate(self, *, prompt: str = "", system: str | None = None, stream: bool = False, **_kwargs):
        text = self._answer(f"{system}\n\n{prompt}" if system else prompt, prompt)
        if not stream:
            return GenerateResponse(model="m", response=text, done=True)

        async def chunks():
            yield GenerateResponse(model="m", response=text, done=True)

        return chunks()


def _classifier(label: str) -> str:
    return json.dumps({"label": label, "confidence": 0.9, "context_action": "new_topic", "entities": {}})


def verdict(outcome: str = "none", picks: tuple[int, ...] = (), terms: tuple[str, ...] = ()) -> str:
    return json.dumps({"picks": list(picks), "outcome": outcome, "search_terms": list(terms)})


def turn(world_dir, monkeypatch, reply: str, *, switch: str, answers: dict[str, Any], cfg_extra: dict[str, str] | None = None) -> dict[str, Any]:
    from messengers_router.endpoint import MessengerGenerateRequest, messenger_generate_once
    from messengers_router.memory import MemoryStore
    from messengers_router.runtime_config import config as cfg
    from messengers_router.services import Services
    from messengers_router.services import _prepare_select_llm

    fake = _LLM(answers)
    memory = MemoryStore()
    session = f"svc-{switch}-{reply}"

    async def run() -> dict[str, Any]:
        req = MessengerGenerateRequest(session_id=session, text=reply, debug=True)
        body = json.loads((await messenger_generate_once(req, memory=memory, services=Services())).body)
        return {"text": str(body.get("text") or ""), "handoff": bool(body.get("handoff"))}

    with ev_world.FrozenWorld(world_dir, "day") as world, monkeypatch.context() as patch:
        patch.setattr(cfg, "MR_NLU_ENGINE", "legacy_v2", raising=False)
        patch.setattr(cfg, "MR_SERVICE_ANSWER_FROM_DATA", switch, raising=False)
        for key, value in (cfg_extra or {}).items():
            patch.setattr(cfg, key, value, raising=False)
        patch.setattr(llm_runtime, "_BREAKER", llm_runtime._Breaker())
        patch.setattr(llm_runtime, "_OLLAMA_CLIENT", fake)
        patch.setitem(_prepare_select_llm._kb_cache, "memos", ())

        async def regions(_self):
            return list(_REGIONS)

        patch.setattr(Services, "_ensure_regions_loaded", regions)
        choice._CACHE.clear()
        out = asyncio.run(run())
        out["telemetry"] = world.take_telemetry() or {}
    out["calls"] = list(fake.calls)
    out["prompts"] = dict(fake.prompts)
    out["pending_offer"] = bool(memory.get(session).last_entities.get("_operator_offer_pending"))
    return out


# --------------------------------------------------------------------------- класс 1: утверждение о клинике без данных

# Формулировки «вы берёте / делаете / принимаете / есть ли у вас …», которые на правилах доходят до свободного текста LLM.
CLAIM_QUESTIONS = [
    "Вы берёте анализы крови у детей 2 лет?",
    "Делаете ли анализы детям до года?",
    "Принимаете ли вы анализы у беременных без направления?",
    "Принимаете ли кал на анализ в субботу?",
    "Берут ли в воскресенье анализы крови?",
    "Вы делаете анализы по ДМС?",
    "Есть ли пакеты анализов?",
    "В какой клинике в Самаре можно сдать анализы ребенку?",
]


def _free_text_answers(service_answer: Any) -> dict[str, Any]:
    return {_CLASSIFIER_MARK: _classifier("TEST_ASSIST"), _SERVICE_ANSWER_MARK: service_answer, _RENDER_MARK: FABRICATED}


@pytest.mark.parametrize("reply", CLAIM_QUESTIONS)
def test_claim_without_data_is_an_honest_operator_offer(reply, answer_world, monkeypatch):
    out = turn(answer_world, monkeypatch, reply, switch="on", answers=_free_text_answers(verdict("no_data")))

    assert out["telemetry"].get("answer_source") == "service_answer", out["telemetry"]
    assert FABRICATED not in out["text"] and "Да" not in out["text"], out["text"]
    assert "нет данных" in out["text"] and "Перевести?" in out["text"], out["text"]
    assert out["pending_offer"] and not out["handoff"]
    assert _RENDER_MARK not in out["calls"]  # свободный текст не писался


@pytest.mark.parametrize("reply", CLAIM_QUESTIONS)
def test_off_is_as_before(reply, answer_world, monkeypatch):
    out = turn(answer_world, monkeypatch, reply, switch="off", answers=_free_text_answers(verdict("no_data")))

    assert _SERVICE_ANSWER_MARK not in out["calls"]
    # Как на проде: ход подбора анализов отвечает по данным хода, свободного текста нет (`test_assist_from_data = on`).
    assert out["telemetry"].get("answer_source") == "test_assist_data", out["telemetry"]
    assert FABRICATED not in out["text"] and _RENDER_MARK not in out["calls"], out["text"]


@pytest.mark.parametrize("failure", [ConnectionError("тест: LLM недоступна"), "не json", verdict("not_about_clinic")])
def test_failure_or_not_about_clinic_answers_as_off(failure, answer_world, monkeypatch):
    reply = CLAIM_QUESTIONS[0]
    off = turn(answer_world, monkeypatch, reply, switch="off", answers=_free_text_answers(verdict("no_data")))
    on = turn(answer_world, monkeypatch, reply, switch="on", answers=_free_text_answers(failure))

    assert on["text"] == off["text"]
    assert on["telemetry"].get("answer_source") == "test_assist_data"  # как off на проде: ответ подбора по данным


def test_claim_is_printed_from_the_picked_price_row(answer_world, monkeypatch):
    out = turn(
        answer_world,
        monkeypatch,
        "Чекап делаете?",
        switch="on",
        answers=_free_text_answers(lambda prompt: verdict(picks=(_number_of(prompt, "Прайс клиники: Ежегодный Чекап"),))),
    )

    assert out["telemetry"].get("answer_source") == "service_answer", out["telemetry"]
    assert out["text"].startswith("Да, по данным клиники это есть:"), out["text"]
    assert "— Ежегодный Чекап — 2 055 руб." in out["text"], out["text"]  # название и цена — из прайса, не из LLM
    assert FABRICATED not in out["text"]


def test_llm_cannot_pick_outside_the_data(answer_world, monkeypatch):
    out = turn(answer_world, monkeypatch, CLAIM_QUESTIONS[0], switch="on", answers=_free_text_answers(verdict(picks=(99,))))

    assert out["telemetry"].get("answer_source") == "test_assist_data"  # номер вне списка — мусор: не верим, ход как под off
    assert out["telemetry"].get("svc_answer") == "failed", out["telemetry"]


def _number_of(prompt: str, text: str) -> int:
    for line in prompt.splitlines():
        head, _, rest = line.partition(". ")
        if head.isdigit() and rest.startswith(text):
            return int(head)
    raise AssertionError(f"нет пункта «{text}» в промпте:\n{prompt}")


def test_fallthrough_returns_the_turn_budget(answer_world, monkeypatch):
    out = turn(
        answer_world,
        monkeypatch,
        CLAIM_QUESTIONS[0],
        switch="on",
        answers=_free_text_answers(verdict("not_about_clinic")),
        cfg_extra={"MR_LLM_DECISION_TURN_CALLS": 2},  # первое решение хода — разбор фраз
    )

    assert out["telemetry"].get("answer_source") == "test_assist_data"  # ход как под off (прод: ответ подбора по данным)
    assert out["telemetry"].get("svc_answer_refund") == 1, out["telemetry"]  # ход как был — лимит решений не потрачен
