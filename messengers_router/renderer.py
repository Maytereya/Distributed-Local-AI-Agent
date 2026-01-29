from __future__ import annotations

import asyncio
from typing import Any, AsyncGenerator

from ollama import AsyncClient

from agent_logic_2 import config as c, ollama_settings
from agent_logic_2.llama_func_call import with_retries, timeout
from agent_logic_2.ollama_settings import LLMName
from mess_types import Evidence, RouteDecision, ResponseEnvelope
from policies import sanitize_for_patient

ollama_client = AsyncClient(c.ollama_url)


@with_retries(tries=2)
async def ollama_call(prompt: str, llm: str = LLMName.get(), think: bool = None, ) -> AsyncGenerator[str, Any]:
    if not llm:
        raise ValueError("Model is not specified yet")
    think = ollama_settings.resolve_think(think)

    stream = await asyncio.wait_for(
        ollama_client.generate(
            model=llm,
            prompt=prompt,
            options=ollama_settings.options_set(),
            stream=True,
            think=think,
        ),
        timeout=timeout,
    )

    async for _chunk in stream:
        delta = _chunk.get("response", "")
        if delta:
            yield delta


def _final_prompt(user_text: str, decision: RouteDecision, evidence: Evidence) -> str:
    return f"""
Ты помощник клиники в мессенджере.
Правила:
- Не ставь диагноз и не назначай лечение.
- Не выдумывай цены, врачей, адреса, акции. Используй ТОЛЬКО факты из "Данные".
- Если данных недостаточно — задай 1 уточняющий вопрос или предложи соединить с оператором.
- Будь кратким.

Запрос пациента: {user_text}

Классификация: {decision.label}, flags={sorted(decision.flags)}

Данные:
{evidence.items}
""".strip()


def render_urgent() -> ResponseEnvelope:
    txt = (
        "Похоже, ситуация может быть срочной.\n\n"
        "Если есть угроза жизни (трудно дышать, сильная боль, кровь, потеря сознания) — вызовите скорую помощь.\n"
        "Если это не экстренно — напишите, что нужно: запись к врачу/адрес/стоимость, и я помогу."
    )
    return ResponseEnvelope(text=txt, handoff=True)


def render_medical_advice() -> ResponseEnvelope:
    txt = (
        "Я не могу поставить диагноз или назначить лечение в чате.\n\n"
        "Могу помочь:\n"
        "1) записаться к подходящему специалисту,\n"
        "2) подсказать адрес/стоимость/подготовку,\n"
        "3) соединить с оператором.\n\n"
        "Напишите кратко: возраст, основные симптомы и как давно."
    )
    return ResponseEnvelope(text=txt, handoff=True)


def render_complaint() -> ResponseEnvelope:
    txt = (
        "Мне жаль, что так получилось. Я помогу передать обращение.\n\n"
        "Напишите, пожалуйста:\n"
        "1) дату и филиал,\n"
        "2) что произошло (2–3 предложения),\n"
        "3) контакт для обратной связи.\n\n"
        "Могу соединить с оператором."
    )
    return ResponseEnvelope(text=txt, handoff=True)


async def render_stream(user_text: str, decision: RouteDecision, evidence: Evidence) -> AsyncGenerator[str, None]:
    prompt = _final_prompt(user_text, decision, evidence)
    async for chunk in ollama_call(prompt):
        yield sanitize_for_patient(chunk)
