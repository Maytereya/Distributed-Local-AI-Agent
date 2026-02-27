"""Рендеринг финального ответа пациенту.

Формирует prompt для LLM по decision+evidence, отдает поток текстовых чанков
и предоставляет шаблонные ответы для safety/early-exit веток.

Ответственность модуля:
1) Преобразовать evidence в пациентский текст (включая deterministic-форматтеры).
2) Использовать LLM-рендер там, где нет надежного шаблона.
3) Возвращать безопасные fallback-ответы для критических сценариев.
"""

from __future__ import annotations

import json
from datetime import datetime, date, time
from typing import Any, AsyncGenerator

from .llm_mode_policy import RuntimeOptions
from .llm_runtime import generate_stream_text, generate_text
from .mess_types import Evidence, RouteDecision, ResponseEnvelope
from .policies import sanitize_for_patient
from .prompt_registry import load_prompt_text
from .self_check import build_critic_prompt, parse_critic_result, should_regenerate

timeout = 300


def _final_prompt(user_text: str, decision: RouteDecision, evidence: Evidence) -> str:
    tmpl = load_prompt_text("renderer_patient")
    flags = ", ".join(sorted(decision.flags))
    evidence_txt = json.dumps(evidence.items, ensure_ascii=False)
    return (
        tmpl.replace("<<USER_TEXT>>", user_text)
        .replace("<<LABEL>>", decision.label)
        .replace("<<FLAGS>>", flags)
        .replace("<<EVIDENCE>>", evidence_txt)
    ).strip()


def _final_prompt_rich(
    user_text: str,
    decision: RouteDecision,
    evidence: Evidence,
    critique: str = "",
) -> str:
    tmpl = load_prompt_text("renderer_patient_rich")
    flags = ", ".join(sorted(decision.flags))
    evidence_txt = json.dumps(evidence.items, ensure_ascii=False)
    return (
        tmpl.replace("<<USER_TEXT>>", user_text)
        .replace("<<LABEL>>", decision.label)
        .replace("<<FLAGS>>", flags)
        .replace("<<EVIDENCE>>", evidence_txt)
        .replace("<<CRITIQUE>>", critique or "Нет")
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


def format_doctor_info_for_patient(payload: dict[str, Any], entities: dict[str, Any]) -> str:
    docs = payload.get("doctors")
    if not isinstance(docs, list) or not docs:
        return "К сожалению, информация о враче не найдена. Уточните фамилию или специальность."

    doctor_hint = str(entities.get("doctor_name") or "").strip().lower().replace("ё", "е")
    if doctor_hint:
        narrowed = []
        for d in docs:
            fio = str(d.get("fio") or "").strip().lower().replace("ё", "е")
            if doctor_hint in fio:
                narrowed.append(d)
        if narrowed:
            docs = narrowed

    lines: list[str] = []
    for i, doc in enumerate(docs[:3], 1):
        fio = str(doc.get("fio") or "Врач").strip()
        spec = str(doc.get("specialization") or "").strip()
        regions = doc.get("regions") or []
        lines.append(f"{i}. {fio}")
        if spec:
            # specialization уже сжат в services, оставляем человекочитаемый блок.
            lines.append(spec)
        if isinstance(regions, list) and regions:
            clean_regions = [str(r).strip() for r in regions if str(r).strip()]
            if clean_regions:
                lines.append(f"Адреса приема: {', '.join(clean_regions)}")
        lines.append("")

    lines.append("Если нужно — могу показать расписание этого врача или помочь с записью.")
    return "\n".join([l for l in lines if l is not None]).strip()


def format_address_for_patient(
    payload: dict[str, Any],
    entities: dict[str, Any],
    *,
    nonbookable_service: str | None = None,
) -> str:
    target_city = str(entities.get("city") or "").strip()
    addresses_raw = payload.get("addresses")
    branches_raw = payload.get("branches")
    addresses = [str(a).strip() for a in addresses_raw] if isinstance(addresses_raw, list) else []
    addresses = [a for a in addresses if a]

    branches: list[dict[str, str]] = []
    if isinstance(branches_raw, list):
        for row in branches_raw:
            if not isinstance(row, dict):
                continue
            addr = str(row.get("address") or "").strip()
            if not addr:
                continue
            city = str(row.get("city") or "").strip()
            if target_city and city and target_city.lower() not in city.lower():
                continue
            branches.append(
                {
                    "address": addr,
                    "phone": str(row.get("phone") or "").strip(),
                    "work_time": str(row.get("work_time") or "").strip(),
                }
            )

    if not branches:
        for a in addresses:
            if target_city and target_city.lower() not in a.lower():
                continue
            branches.append({"address": a, "phone": "", "work_time": ""})

    if not branches:
        return "Не нашёл филиалы по этому городу. Уточните город или адрес, пожалуйста."

    lines: list[str] = []
    if nonbookable_service:
        svc = str(nonbookable_service).strip()
        svc = svc.replace("экг", "ЭКГ").replace("Экг", "ЭКГ")
        if svc:
            svc = svc[:1].upper() + svc[1:]
        lines.append(f"{svc} выполняются без записи, в порядке живой очереди.")
        lines.append("")

    if target_city:
        lines.append(f"В городе {target_city} доступны филиалы:")
    else:
        lines.append("Доступные филиалы:")

    for i, b in enumerate(branches, 1):
        lines.append(f"{i}. {b['address']}")
        if b.get("phone"):
            lines.append(f"Телефон: {b['phone']}")
        if b.get("work_time"):
            lines.append(f"График: {b['work_time']}")
        lines.append("")

    return "\n".join([x for x in lines if x is not None]).strip()


def format_news_for_patient(payload: dict[str, Any], entities: dict[str, Any]) -> str:
    news = payload.get("news")
    if not isinstance(news, list):
        news = []

    if not news:
        city = str(entities.get("city") or "").strip()
        if city:
            return (
                f"По вашему запросу в городе {city} сейчас нет подходящих активных акций. "
                "Могу подсказать адреса филиалов или стоимость нужной услуги."
            )
        return "По вашему запросу сейчас нет подходящих активных акций. Могу подсказать адреса филиалов или стоимость услуги."

    lines: list[str] = ["Нашёл актуальные предложения:"]
    for i, item in enumerate(news[:5], 1):
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or item.get("name") or item.get("subject") or "").strip()
        url = str(item.get("url") or item.get("link") or "").strip()
        if not title:
            title = "Акция"
        lines.append(f"{i}. {title}")
        if url:
            lines.append(url)
    lines.append("")
    lines.append("Если нужно, могу уточнить условия акции по вашему филиалу.")
    return "\n".join(lines).strip()


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


async def _rich_generate_once(
    user_text: str,
    decision: RouteDecision,
    evidence: Evidence,
    *,
    queue_timeout_ms: int,
    critique: str = "",
) -> str:
    prompt = _final_prompt_rich(user_text, decision, evidence, critique=critique)
    raw = await generate_text(
        prompt,
        timeout_s=timeout,
        queue_timeout_ms=queue_timeout_ms,
    )
    return sanitize_for_patient(raw.strip())


async def _rich_self_check(
    user_text: str,
    decision: RouteDecision,
    evidence: Evidence,
    candidate_answer: str,
    *,
    queue_timeout_ms: int,
) -> dict[str, Any]:
    prompt = build_critic_prompt(
        user_text=user_text,
        label=decision.label,
        flags=sorted(decision.flags),
        evidence_items=dict(evidence.items or {}),
        candidate_answer=candidate_answer,
    )
    raw = await generate_text(
        prompt,
        timeout_s=timeout,
        queue_timeout_ms=queue_timeout_ms,
        fmt="json",
    )
    return parse_critic_result(raw)


async def render_stream(
    user_text: str,
    decision: RouteDecision,
    evidence: Evidence,
    runtime_options: RuntimeOptions | None = None,
) -> AsyncGenerator[str, None]:
    opts = runtime_options or RuntimeOptions()
    queue_timeout_ms = int(opts.queue_timeout_ms)

    if opts.llm_mode != "rich":
        prompt = _final_prompt(user_text, decision, evidence)
        async for chunk in generate_stream_text(
            prompt,
            timeout_s=timeout,
            queue_timeout_ms=queue_timeout_ms,
        ):
            yield sanitize_for_patient(chunk)
        return

    answer = await _rich_generate_once(
        user_text,
        decision,
        evidence,
        queue_timeout_ms=queue_timeout_ms,
    )
    if not opts.self_check:
        yield answer
        return

    critique_reason = ""
    max_tries = max(0, int(opts.self_check_max_retries))
    for _ in range(max_tries + 1):
        verdict = await _rich_self_check(
            user_text,
            decision,
            evidence,
            answer,
            queue_timeout_ms=queue_timeout_ms,
        )
        regen, reason = should_regenerate(verdict, threshold=float(opts.self_check_threshold))
        if not regen:
            break
        critique_reason = reason or critique_reason
        answer = await _rich_generate_once(
            user_text,
            decision,
            evidence,
            queue_timeout_ms=queue_timeout_ms,
            critique=critique_reason,
        )
    yield answer
