"""Свободный текст LLM не называет денежных сумм (BUG-2026-10-10-LLM-FREE-MONEY).

Пункт 1 принципов (решение владельца 08.10): цену печатает код из прайса МИС, LLM чисел своим
текстом не пишет. Стенд 10.10: в ветке `llm_free` модель писала цены сама — копировала числа
из данных хода под пересказанным названием (цена чужой строки) или считала их сама («скидка
7%», «сертификат на 10 000»).

Инварианты класса:
- любой ответ ветки `llm_free`, дошедший через эндпоинт, не содержит денежных сумм — в какой
  бы форме модель их ни написала («1515 рублей», «1 990 руб.», «530 ₽», «190 р.», «скидка
  7%», «сертификат на 10 000») и как бы поток ни резал токены; суммы были — к ответу
  дописан готовый оффер ценового пути;
- встречно: prebuilt-ответ цены из прайса суммы сохраняет; свободный текст без сумм (адрес,
  сроки, «стоит прийти за 15 минут») проходит дословно;
- промпт `renderer_patient` (обе локации, оба режима) запрещает суммы — первая линия.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from ollama import GenerateResponse

from messengers_router import endpoint as endpoint_mod
from messengers_router import llm_runtime, turn_telemetry
from messengers_router import router as router_mod
from messengers_router.memory import MemoryStore
from messengers_router.mess_types import Evidence, RouteDecision
from messengers_router.money_guard import strip_money
from messengers_router.nlu_pipeline import NLUResult
from messengers_router.policies import CLARIFY_TEXT_MAP
from messengers_router.services import Services

_REPO = Path(__file__).resolve().parents[1]
PRICE_OFFER = CLARIFY_TEXT_MAP["PRICE"]
# Независимый от guard'а детектор суммы: число рядом с валютой или процент скидки.
_MONEY_IN_ANSWER = re.compile(r"\d[\d\s ]*\s*(?:руб|₽|р\.)|₽|\bруб|\d\s*%", re.IGNORECASE)


class _FakeOllama:
    """LLM настоящего типа: поток рендера режет текст на куски по 5 символов."""

    def __init__(self, render_text: str) -> None:
        self.render_text = render_text

    async def generate(self, **kwargs):
        if kwargs.get("stream"):
            text = self.render_text

            async def chunks():
                for i in range(0, len(text), 5):
                    yield GenerateResponse(model="m", response=text[i:i + 5], done=False)
                yield GenerateResponse(model="m", response="", done=True, prompt_eval_count=10, eval_count=5)

            return chunks()
        return GenerateResponse(model="m", response="{}", done=True)


@pytest.fixture
def world(monkeypatch):
    """Мир хода офлайн; строки телеметрии — для проверки ветки ответа."""

    from agent_logic_2.nayka_api import api_nayka, api_price

    monkeypatch.setattr(api_nayka, "site_regions", lambda *a, **kw: [])
    monkeypatch.setattr(api_nayka, "get_all_doctors", lambda: [])

    def _no_download(*_a, **_kw):
        raise ConnectionError("тест в сеть не ходит")

    monkeypatch.setattr(api_price.SESSION, "get", _no_download)
    rows: list[dict] = []
    monkeypatch.setattr(turn_telemetry, "_emit", rows.append)
    return rows


def _route_to_free_answer(monkeypatch) -> None:
    """Ход с данными, для которых готового ответа нет, — он уходит в свободный текст LLM."""

    async def fake_analyze_with_candidates(_text, _state, runtime_options=None):
        return NLUResult(
            decision=RouteDecision(label="OTHER", confidence=0.9, entities={}, flags=set(), source="llm_primary"),
            candidates=[],
            merged_from="llm",
        )

    async def fake_execute_plan(_plan, _state, _services):
        return Evidence(items={"clinic_note": {"text": "Анализы сдаются без предварительной записи."}})

    monkeypatch.setattr(router_mod, "analyze_with_candidates", fake_analyze_with_candidates)
    monkeypatch.setattr(router_mod, "execute_plan", fake_execute_plan)
    monkeypatch.setattr(
        router_mod,
        "_env_flag",
        lambda name, default: True if name == "MR_ROUTER_V2_ENABLE" else (False if name == "MR_ROUTER_V2_SHADOW" else default),
    )


def _ask(text: str) -> dict:
    services = Services()
    services.ensure_background_refresh_started = lambda: None
    memory = MemoryStore()
    app = FastAPI()
    app.include_router(endpoint_mod.router)
    app.dependency_overrides[endpoint_mod.get_memory_store] = lambda: memory
    app.dependency_overrides[endpoint_mod.get_services] = lambda: services
    return TestClient(app).post("/api/messenger-generate-once", json={"session_id": "money", "text": text}).json()


# Формы, в которых модель писала суммы на стенде 10.10 (обезличено), + число без валюты.
MONEY_ANSWERS = [
    ("rubles_word", "Приём терапевта стоит 1515 рублей. Клиника работает каждый день.", "Клиника работает каждый день."),
    ("rub_abbr_spaced", "Стоимость этой услуги — 1 990 руб. Анализы сдаются без предварительной записи.",
     "Анализы сдаются без предварительной записи."),
    ("invented_discount", "Срок выполнения: 1-2 дня. Стоимость: 1515 рублей с учетом акции (скидка 7%).",
     "Срок выполнения: 1-2 дня."),
    ("invented_certificate", "Можем оформить сертификат на 10 000. Подробности расскажет администратор.",
     "Подробности расскажет администратор."),
    ("list_with_ruble_sign",
     "Вот информация по анализам:\n\n- **Ферритин**: 530 ₽, срок 1-2 дня.\n- Глюкоза — 190 р.\n\n"
     "Анализы сдаются без предварительной записи.",
     "Анализы сдаются без предварительной записи."),
    ("nested_list_price",
     "Доступные услуги:\n\n1. **УЗИ щитовидной железы**\n   - Стоимость: 2100 руб.\n   - Срок готовности: в день приёма\n",
     "Срок готовности: в день приёма"),
]


@pytest.mark.parametrize("name, llm_text, kept", MONEY_ANSWERS, ids=[m[0] for m in MONEY_ANSWERS])
def test_free_llm_answer_reaches_patient_without_money(monkeypatch, world, name, llm_text, kept):
    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", _FakeOllama(llm_text))
    _route_to_free_answer(monkeypatch)

    answer = _ask("подскажите по анализам")["text"]

    assert world[0]["answer_source"] == "llm_free", world[0]
    assert not _MONEY_IN_ANSWER.search(answer), f"[{name}] сумма дошла до пациента: {answer!r}"
    for number in re.findall(r"\d[\d ]{2,}\d", llm_text):
        assert number not in answer, f"[{name}] число {number!r} из текста LLM дошло до пациента"
    assert kept in answer, f"[{name}] предложение без суммы потеряно: {answer!r}"
    assert answer.rstrip().endswith(PRICE_OFFER), f"[{name}] нет оффера ценового пути: {answer!r}"
    assert world[0]["money_cut"] >= 1


@pytest.mark.parametrize(
    "llm_text",
    [
        "Анализы сдаются без предварительной записи. Адрес: г. Самара, пр. Ленина, 5.",
        "Общий анализ мочи готовится в течение 1-2 дней. Стоит прийти за 15 минут до приёма.",
        "Для уточнения списка анализов и их стоимости напишите, какие анализы вас интересуют.",
    ],
)
def test_free_llm_answer_without_money_passes_verbatim(monkeypatch, world, llm_text):
    """Анти-over-block: адреса, сроки, «стоит прийти», слово «стоимость» без числа — не суммы."""

    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", _FakeOllama(llm_text))
    _route_to_free_answer(monkeypatch)

    answer = _ask("подскажите по анализам")["text"]

    assert world[0]["answer_source"] == "llm_free"
    assert answer == llm_text
    assert "money_cut" not in world[0]


def test_prebuilt_price_answer_keeps_sums(monkeypatch, world):
    """Встречный: цену из прайса печатает код — эту ветку guard не трогает."""

    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", _FakeOllama("Приём стоит 1515 рублей."))

    answer = _ask("сколько стоит общий анализ крови")["text"]

    assert world[0]["answer_source"] == "prebuilt", world[0]
    assert re.search(r"\d+\s*руб", answer), answer
    assert "1515" not in answer  # текст LLM сюда не попадает вовсе
    assert "money_cut" not in world[0]


@pytest.mark.parametrize(
    "text",
    [
        "Анализы сдаются без предварительной записи. Адрес: г. Самара, пр. Ленина, 5.",
        "Проба 1 и проба 2 в общем анализе мочи — разные порции. Проба 1 — утренняя.",
        "1. **Гликированный гемоглобин (HbA1c)**\n2. **Анализ мочи на сахар**",
        "Цитология готовится 12–27 дней. Оценка результата — у врача.",
        "Если хотите узнать стоимость и сроки, пришлите точные названия анализов.",
    ],
)
def test_strip_money_leaves_non_money_text_unchanged(text):
    assert strip_money(text) == (text, 0)


@pytest.mark.parametrize("key", ["renderer_patient", "renderer_patient_rich"])
def test_render_prompts_forbid_money_in_both_locations(key):
    for path in (_REPO / "messengers_router" / "prompts" / f"{key}.txt", _REPO / "app_data" / "prompts" / f"mr_{key}.txt"):
        body = path.read_text(encoding="utf-8")
        assert "Цены, суммы, скидки и стоимость НЕ пиши никогда" in body, path
        assert "можно сообщить срок, стоимость" not in body, path
        assert "обязательно укажи конкретную стоимость" not in body, path
