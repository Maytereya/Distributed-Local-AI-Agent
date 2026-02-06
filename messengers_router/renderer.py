from __future__ import annotations

import asyncio
import json
from datetime import datetime, date, time
from functools import lru_cache
from pathlib import Path
from typing import Any, AsyncGenerator

from ollama import AsyncClient

from agent_logic_2 import config as c, ollama_settings
# from agent_logic_2.llama_func_call import timeout
from agent_logic_2.ollama_settings import LLMName
from .mess_types import Evidence, RouteDecision, ResponseEnvelope
from .policies import sanitize_for_patient

ollama_client = AsyncClient(c.ollama_url)
timeout = 300
_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


@lru_cache
def _load_prompt(name: str) -> str:
    path = _PROMPTS_DIR / name
    return path.read_text(encoding="utf-8")

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
    tmpl = _load_prompt("renderer_patient.txt")
    flags = ", ".join(sorted(decision.flags))
    evidence_txt = json.dumps(evidence.items, ensure_ascii=False)
    return (
        tmpl.replace("<<USER_TEXT>>", user_text)
        .replace("<<LABEL>>", decision.label)
        .replace("<<FLAGS>>", flags)
        .replace("<<EVIDENCE>>", evidence_txt)
    ).strip()


def _parse_date_iso(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s).date()
    except Exception:
        return None


def _parse_time_hhmm(s: str | None) -> time | None:
    if not s:
        return None
    try:
        return datetime.strptime(s[:5], "%H:%M").time()
    except Exception:
        return None


def _filter_slots(slots: list[str], t_from: time | None, t_to: time | None) -> list[str]:
    if not slots:
        return []
    if not (t_from or t_to):
        return [s[:5] for s in slots if s][:8]
    out: list[str] = []
    for s in slots:
        if not s:
            continue
        try:
            t = datetime.strptime(s[:5], "%H:%M").time()
        except Exception:
            continue
        if t_from and t < t_from:
            continue
        if t_to and t > t_to:
            continue
        out.append(s[:5])
        if len(out) >= 8:
            break
    return out


def format_doctor_schedule_for_patient(payload: dict[str, Any], entities: dict[str, Any]) -> str:
    docs = payload.get("schedule")
    if not isinstance(docs, list) or not docs:
        return "К сожалению, расписание не найдено. Уточните фамилию врача или город."

    target = (entities.get("city") or entities.get("branch_name") or entities.get("region") or "").strip()
    date_from = _parse_date_iso(entities.get("date_from"))
    date_to = _parse_date_iso(entities.get("date_to")) or date_from
    t_from = _parse_time_hhmm(entities.get("time_from"))
    t_to = _parse_time_hhmm(entities.get("time_to"))

    lines: list[str] = []

    for i, doc in enumerate(docs[:3], 1):
        fio = str(doc.get("fio") or "Врач")
        spec = str(doc.get("specialization") or "").strip()
        regions = doc.get("regions") or []
        schedule = doc.get("schedule") or {}

        # filter by city/branch if provided
        if target:
            regions = [r for r in regions if target.lower() in str(r).lower()] or regions
            schedule = {k: v for k, v in schedule.items() if target.lower() in str(k).lower()} or schedule

        if spec:
            lines.append(f"{i}. {fio} — {spec}")
        else:
            lines.append(f"{i}. {fio}")

        if regions:
            lines.append(f"Адреса приема: {', '.join(regions)}")

        has_any = False
        for region_name, days in schedule.items():
            if region_name:
                lines.append(f"{region_name}:")
            for day in days or []:
                day_date = day.get("date")
                if not day_date:
                    continue
                try:
                    d_obj = _parse_date_iso(str(day_date))
                except Exception:
                    d_obj = None
                if date_from and d_obj and date_to:
                    if d_obj < date_from or d_obj > date_to:
                        continue

                slots = _filter_slots(day.get("slots") or [], t_from, t_to)
                if slots:
                    lines.append(f"• {day_date}: свободно {', '.join(slots)}")
                    has_any = True
                else:
                    start = (day.get("start") or "")[:5]
                    end = (day.get("end") or "")[:5]
                    if start or end:
                        lines.append(f"• {day_date}: {start}-{end}")
                        has_any = True

        if not has_any:
            if date_from:
                lines.append("Свободных окон на выбранную дату не найдено.")
            else:
                lines.append("Свободных окон в ближайшие дни не найдено.")
        lines.append("")

    lines.append("Если нужно записаться — напишите удобное время или уточните врача/филиал.")
    return "\n".join([l for l in lines if l is not None]).strip()


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
        # yield chunk
        yield sanitize_for_patient(chunk)
