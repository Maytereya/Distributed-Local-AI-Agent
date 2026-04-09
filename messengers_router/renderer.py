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
import re
from datetime import datetime, date, time
from typing import Any, AsyncGenerator

from .llm_mode_policy import RuntimeOptions
from .llm_runtime import generate_stream_text, generate_text
from .mess_types import Evidence, RouteDecision, ResponseEnvelope
from .policies import sanitize_for_patient
from .prompt_registry import load_prompt_text
from .self_check import build_critic_prompt, parse_critic_result, should_regenerate
from .runtime_config import config as c

timeout = 300
try:
    DOCTORS_TOP_N = max(1, int(c.MR_DOCTORS_TOP_N))
except Exception:
    DOCTORS_TOP_N = 4

_RU_MONTH_GEN = {
    1: "января",
    2: "февраля",
    3: "марта",
    4: "апреля",
    5: "мая",
    6: "июня",
    7: "июля",
    8: "августа",
    9: "сентября",
    10: "октября",
    11: "ноября",
    12: "декабря",
}


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


def _format_date_ru_short(value: str | None) -> str:
    raw = str(value or "").strip()
    if not raw:
        return raw
    parsed = _parse_date_iso(raw)
    if not parsed:
        return raw
    month = _RU_MONTH_GEN.get(parsed.month)
    if not month:
        return raw
    return f"{parsed.day} {month}"


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
    any_free_slots_global = False
    rendered_doctors_count = 0
    visible_regions_global: set[str] = set()
    known_doctor = bool(str(entities.get("doctor_name") or entities.get("doctor_id") or "").strip())
    known_branch = bool(str(entities.get("branch_name") or "").strip())

    for i, doc in enumerate(docs[:3], 1):
        fio = str(doc.get("fio") or "Врач")
        regions = doc.get("regions") or []
        schedule = doc.get("schedule") or {}

        # filter by city/branch if provided
        if target:
            regions = [r for r in regions if target.lower() in str(r).lower()] or regions
            schedule = {k: v for k, v in schedule.items() if target.lower() in str(k).lower()} or schedule

        lines.append(f"{i}. {fio}")
        rendered_doctors_count += 1

        if regions:
            lines.append(f"Адреса приема: {', '.join(regions)}")

        has_any_content = False
        has_free_slots = False
        visible_regions: list[str] = []
        rendered_schedule_lines: list[str] = []
        for region_name, days in schedule.items():
            region_lines: list[str] = []
            region_has_content = False
            if region_name:
                region_lines.append(f"{region_name}:")
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
                day_label = _format_date_ru_short(str(day_date))
                if slots:
                    region_lines.append(f"• {day_label}: свободно в {', '.join(slots)}")
                    region_has_content = True
                    has_free_slots = True
                else:
                    start = (day.get("start") or "")[:5]
                    end = (day.get("end") or "")[:5]
                    if start or end:
                        region_lines.append(f"• {day_label}: {start}-{end}")
                        region_has_content = True

            if region_has_content:
                has_any_content = True
                if region_name:
                    region_clean = str(region_name).strip()
                    visible_regions.append(region_clean)
                    if region_clean:
                        visible_regions_global.add(region_clean)
                rendered_schedule_lines.extend(region_lines)

        if visible_regions:
            visible_regions = list(dict.fromkeys([x for x in visible_regions if x]))
            if len(visible_regions) == 1:
                lines.append(f"Ближайшее актуальное расписание сейчас есть в филиале: {visible_regions[0]}")
            else:
                lines.append(
                    "Ближайшее актуальное расписание сейчас есть по адресам: "
                    + ", ".join(visible_regions)
                )

        lines.extend(rendered_schedule_lines)

        if not has_free_slots:
            if date_from:
                lines.append("Свободных окон на выбранную дату не найдено.")
            else:
                lines.append("Свободных окон в ближайшие дни не найдено.")
        if has_free_slots:
            any_free_slots_global = True
        elif not has_any_content and not schedule:
            lines.append("Расписание по этому врачу пока недоступно.")
        lines.append("")

    if any_free_slots_global:
        need_doctor_clarify = rendered_doctors_count > 1 and not known_doctor
        need_branch_clarify = len(visible_regions_global) > 1 and not known_branch
        if need_doctor_clarify and need_branch_clarify:
            lines.append("Если нужно записаться — напишите удобное время или уточните врача/филиал.")
        elif need_doctor_clarify:
            lines.append("Если нужно записаться — напишите удобное время или уточните врача.")
        elif need_branch_clarify:
            lines.append("Если нужно записаться — напишите удобное время или уточните филиал.")
        else:
            lines.append("Если нужно записаться — напишите удобное время.")
    else:
        lines.append("Могу подобрать другого врача или передать диалог оператору.")
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

    single_selected = bool(doctor_hint) and len(docs) == 1

    shown_docs = docs[:DOCTORS_TOP_N]
    shown_count = len(shown_docs)

    lines: list[str] = []
    for i, doc in enumerate(shown_docs, 1):
        fio = str(doc.get("fio") or "Врач").strip()
        spec = str(doc.get("specialization") or "").strip()
        regions = doc.get("regions") or []
        lines.append(f"{i}. {fio}")
        if spec and not single_selected:
            # specialization уже сжат в services, оставляем человекочитаемый блок.
            lines.append(spec)
        if isinstance(regions, list) and regions:
            clean_regions = [str(r).strip() for r in regions if str(r).strip()]
            if clean_regions:
                lines.append(f"Адреса приема: {', '.join(clean_regions)}")
        lines.append("")

    if single_selected:
        lines.append("Хотите записаться к этому врачу? Напишите «расписание» или «запись».")
    else:
        if shown_count <= 1:
            lines.append("Если нужно — могу показать расписание этого врача или помочь с записью.")
        else:
            lines.append("Если нужно — могу показать расписание любого из этих врачей или помочь с записью.")
    return "\n".join([l for l in lines if l is not None]).strip()


def _format_slot_compact(slot_iso: str) -> str:
    raw = str(slot_iso or "").strip()
    if not raw:
        return ""
    try:
        dt = datetime.fromisoformat(raw)
        return dt.strftime("%d.%m %H:%M")
    except Exception:
        return raw


def _availability_text_for_doctor(doc: dict[str, Any]) -> str:
    note = str(doc.get("availability_note") or "").strip().lower()
    available = bool(doc.get("available"))
    nearest = _format_slot_compact(str(doc.get("nearest_slot") or ""))

    if note == "availability_source_unavailable":
        return "расписание временно недоступно"
    if note == "availability_missing_surname":
        return "расписание не проверено"
    if note in {"availability_empty", "availability_unmatched"}:
        return "расписание не найдено"
    if note == "availability_checked":
        if available and nearest:
            return f"доступен, ближайшее окно {nearest}"
        if available:
            return "доступен"
        return "сейчас без свободных окон"

    if available and nearest:
        return f"доступен, ближайшее окно {nearest}"
    if available:
        return "доступен"
    return "статус расписания уточняется"


def _format_price_family_variants(payload: dict[str, Any], fallback_service_name: str = "") -> str:
    """
    Формирует patient-facing текст для family-query price-выдачи.

    :param payload: payload семейства price-вариантов
    :param fallback_service_name: запасное название запроса
    :return: отформатированный текст
    """

    variants_raw = payload.get("family_variants")
    variants = variants_raw if isinstance(variants_raw, list) else []
    if not variants:
        return ""

    service_name = str(payload.get("service_name") or fallback_service_name or "услуга").strip()
    showing_all = bool(payload.get("showing_all"))
    visible_limit = max(1, int(payload.get("visible_limit") or 10))
    shown = variants if showing_all else variants[:visible_limit]

    lines = [f"По запросу «{service_name}» нашёл варианты стоимости:"]
    grouped = _group_family_variants_by_care_context(shown)
    has_care_groups = any(group["label"] or group["address"] for group in grouped)

    if has_care_groups:
        for group in grouped:
            rows = group["rows"]
            if not isinstance(rows, list) or not rows:
                continue
            lines.append("")
            heading = _format_family_care_group_heading(
                str(group["label"] or ""),
                str(group["address"] or ""),
            )
            if heading:
                lines.append(heading)
            for i, row in enumerate(rows, 1):
                if not isinstance(row, dict):
                    continue
                name = str(row.get("serviceName") or row.get("name") or service_name).strip()
                amount = _format_rub(_extract_price_amount(row))
                lines.append(f"{i}. {name} — {amount}.")
    else:
        for i, row in enumerate(shown, 1):
            if not isinstance(row, dict):
                continue
            name = str(row.get("serviceName") or row.get("name") or service_name).strip()
            amount = _format_rub(_extract_price_amount(row))
            care_suffix = _format_care_setting_suffix(row)
            line = f"{i}. {name} — {amount}."
            if care_suffix:
                line = f"{line} {care_suffix}"
            lines.append(line)

    hint = str(payload.get("show_all_hint") or "").strip()
    if hint:
        lines.append("")
        lines.append(hint)
    lines.append("Если нужно, помогу выбрать подходящий вариант или подскажу подготовку.")
    return "\n".join(lines).strip()


def format_service_bundle_for_patient(payload: dict[str, Any], entities: dict[str, Any]) -> str:
    clarify_text = str(payload.get("clarify_text") or "").strip()
    if clarify_text:
        return clarify_text
    service_name = str(payload.get("service_name") or entities.get("service_name") or entities.get("test_name") or "").strip()
    retail_prices_raw = payload.get("retail_prices")
    doctors_raw = payload.get("doctors")
    service_kind = str(payload.get("service_kind") or "").strip().lower()
    prepare_text = str(payload.get("prepare") or "").strip()
    show_prepare = bool(payload.get("show_prepare"))
    is_consult = bool(re.search(r"\b(при[её]м\w*|консультац\w*)\b", service_name.lower()))
    doctors = doctors_raw if isinstance(doctors_raw, list) else []
    retail_prices = retail_prices_raw if isinstance(retail_prices_raw, list) else []

    if not service_name:
        service_name = "услуга"
    if service_kind == "family_query":
        return _format_price_family_variants(payload, service_name)

    lines: list[str] = [f"По услуге «{service_name}» нашёл следующее:"]

    if retail_prices:
        if service_kind == "lab" and len(retail_prices) > 1:
            lines.append("1) Розничные варианты:")
            for price_row in retail_prices[:5]:
                if not isinstance(price_row, dict):
                    continue
                amount = _format_rub(_extract_price_amount(price_row))
                price_name = str(price_row.get("serviceName") or price_row.get("name") or service_name).strip()
                care_suffix = _format_care_setting_suffix(price_row)
                line = f"- {price_name} — {amount}."
                if care_suffix:
                    line = f"{line} {care_suffix}"
                lines.append(line)
        else:
            top_price = retail_prices[0] if isinstance(retail_prices[0], dict) else {}
            amount = _format_rub(_extract_price_amount(top_price))
            price_name = str(top_price.get("serviceName") or top_price.get("name") or service_name).strip()
            care_suffix = _format_care_setting_suffix(top_price)
            line = f"1) Розничная цена: {price_name} — {amount}."
            if care_suffix:
                line = f"{line} {care_suffix}"
            lines.append(line)
    else:
        lines.append("1) Розничную цену сейчас точно определить не удалось.")

    if service_kind == "lab":
        lines.append("2) Для этого лабораторного анализа запись к конкретному врачу обычно не требуется.")
    elif service_kind == "diagnostic_no_doctor":
        lines.append("2) Для этой диагностической услуги список врачей автоматически не показываю.")
    else:
        if doctors:
            lines.append("2) Врачи (по приоритету):")
            for i, doc in enumerate(doctors, 1):
                if not isinstance(doc, dict):
                    continue
                fio = str(doc.get("fio") or "Врач").strip()
                price_txt = _format_rub(_extract_price_amount(doc))
                avail_txt = _availability_text_for_doctor(doc)
                lines.append(f"{i}. {fio} — {price_txt}; {avail_txt}.")
        else:
            lines.append("2) Подходящих врачей по этой услуге сейчас не нашёл.")

    if not is_consult and show_prepare:
        if prepare_text:
            lines.append(f"3) Подготовка: {prepare_text}")
        else:
            lines.append("3) Подготовку по этой услуге сейчас не удалось получить автоматически.")

    if service_kind == "lab":
        lines.append("Если нужно, подскажу подготовку к анализу или подходящие филиалы для сдачи.")
    elif service_kind == "diagnostic_no_doctor":
        lines.append("Если нужно, подскажу подходящие филиалы для прохождения исследования.")
    elif not doctors:
        lines.append("Если нужно, передам запрос оператору для уточнения по этой услуге.")
    elif len(doctors) == 1:
        lines.append("Если нужно, покажу подробное расписание этого врача.")
    else:
        lines.append("Если нужно, покажу подробное расписание любого из этих врачей.")
    return "\n".join(lines).strip()


def _extract_price_amount(row: dict[str, Any]) -> int | None:
    for k in ("servicePrice", "price", "service_price", "amount", "cost"):
        v = row.get(k)
        if isinstance(v, (int, float)):
            return int(round(float(v)))
        if isinstance(v, str):
            raw = v.replace(" ", "").replace("\u00a0", "").replace(",", ".")
            m = re.search(r"\d+(?:\.\d+)?", raw)
            if m:
                try:
                    return int(round(float(m.group(0))))
                except Exception:
                    continue
    return None


def _format_rub(amount: int | None) -> str:
    if amount is None:
        return "цена по запросу"
    return f"{amount:,}".replace(",", " ") + " руб."


def _format_care_setting_suffix(row: dict[str, Any]) -> str:
    label = str(row.get("care_setting_label") or "").strip()
    address = str(row.get("care_setting_address") or "").strip()
    if label and address:
        return f"Формат: {label}. Адрес: {address}."
    if label:
        return f"Формат: {label}."
    if address:
        return f"Адрес: {address}."
    return ""


def _group_family_variants_by_care_context(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Группирует family-варианты по формату оказания и адресу.

    :param rows: строки family-вариантов
    :return: список групп с сохранением исходного порядка появления
    """

    groups: list[dict[str, Any]] = []
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        label = str(row.get("care_setting_label") or "").strip()
        address = str(row.get("care_setting_address") or "").strip()
        key = (label, address)
        group = by_key.get(key)
        if group is None:
            group = {"label": label, "address": address, "rows": []}
            by_key[key] = group
            groups.append(group)
        group["rows"].append(row)
    return groups


def _format_family_care_group_heading(label: str, address: str) -> str:
    """
    Формирует заголовок блока family-вариантов по формату оказания услуги.

    :param label: формат оказания услуги
    :param address: адрес оказания услуги
    :return: текст заголовка блока
    """

    norm_label = str(label or "").strip().lower()
    heading_by_label = {
        "поликлиника": "В поликлинике",
        "дневной стационар": "В дневном стационаре",
        "круглосуточный стационар": "В круглосуточном стационаре",
    }
    heading = heading_by_label.get(norm_label)
    if heading and address:
        return f"{heading} по адресу: {address}:"
    if heading:
        return f"{heading}:"
    if label and address:
        return f"Формат: {label}. Адрес: {address}:"
    if label:
        return f"Формат: {label}:"
    if address:
        return f"По адресу: {address}:"
    return "Другие варианты, формат и адрес нужно уточнить:"


def _price_source_marker(payload: dict[str, Any]) -> str | None:
    note = str(payload.get("note") or "").lower()
    if "doctorservicepricesbyregion" in note:
        return "врачебный прайс"
    if "pricebyregion(" in note:
        return "розничный прайс Самары"
    return None


def format_price_for_patient(payload: dict[str, Any], entities: dict[str, Any]) -> str:
    prices_raw = payload.get("prices")
    prices = prices_raw if isinstance(prices_raw, list) else []
    service_hint = str(entities.get("service_name") or entities.get("test_name") or "").strip()
    clarify_text = str(payload.get("clarify_text") or "").strip()
    source_marker = _price_source_marker(payload)

    def _finish(text: str) -> str:
        txt = str(text or "").strip()
        if not txt:
            return txt
        if source_marker:
            return f"{txt}\nИсточник цены: {source_marker}."
        return txt

    if clarify_text:
        return _finish(clarify_text)

    if str(payload.get("service_kind") or "").strip().lower() == "family_query":
        family_text = _format_price_family_variants(payload, service_hint)
        if family_text:
            return _finish(family_text)

    if not prices:
        if service_hint:
            return _finish(
                f"Не нашёл актуальную стоимость для «{service_hint}». "
                "Уточните название услуги или ФИО врача, и я проверю снова."
            )
        return _finish("Уточните, пожалуйста, название услуги или анализа — подскажу стоимость.")

    rows: list[dict[str, str | int | None]] = []
    seen: set[tuple[str, int | None, str, str]] = set()
    for row in prices:
        if not isinstance(row, dict):
            continue
        name = str(row.get("serviceName") or row.get("name") or service_hint or "Услуга").strip()
        fio = str(row.get("fio") or row.get("doctorFio") or "").strip()
        branch = str(row.get("regionName") or row.get("branchName") or "").strip()
        amount = _extract_price_amount(row)
        key = (name.lower(), amount, fio.lower(), branch.lower())
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "name": name,
                "fio": fio,
                "branch": branch,
                "amount": amount,
                "care_setting_label": str(row.get("care_setting_label") or "").strip(),
                "care_setting_address": str(row.get("care_setting_address") or "").strip(),
            }
        )
        if len(rows) >= 5:
            break

    if not rows:
        return _finish("Не удалось разобрать стоимость услуги. Уточните, пожалуйста, формулировку запроса.")

    doctor_context = bool(entities.get("doctor_id") or entities.get("doctor_name"))
    doctor_display = str(entities.get("doctor_name") or "").strip()
    if not doctor_display:
        first_fio = str(rows[0].get("fio") or "").strip()
        if first_fio:
            doctor_display = first_fio

    def _line(item: dict[str, str | int | None]) -> str:
        name = str(item.get("name") or "Услуга")
        fio = str(item.get("fio") or "").strip()
        branch = str(item.get("branch") or "").strip()
        amount = item.get("amount")
        amount_txt = _format_rub(amount if isinstance(amount, int) else None)
        parts = [f"{name} — {amount_txt}"]
        if doctor_context and fio:
            parts.insert(0, f"{fio}:")
        if branch:
            parts.append(f"({branch})")
        care_suffix = _format_care_setting_suffix(item)
        if care_suffix:
            parts.append(care_suffix)
        return " ".join(parts)

    if doctor_context:
        if len(rows) == 1:
            one = rows[0]
            name = str(one.get("name") or "Услуга")
            branch = str(one.get("branch") or "").strip()
            amount = one.get("amount")
            amount_txt = _format_rub(amount if isinstance(amount, int) else None)
            if doctor_display:
                base = f"У врача {doctor_display} услуга «{name}» стоит {amount_txt}."
            else:
                base = f"Услуга «{name}» стоит {amount_txt}."
            if branch:
                base = f"{base} ({branch})"
            care_suffix = _format_care_setting_suffix(one)
            if care_suffix:
                base = f"{base} {care_suffix}"
            return _finish(base)
        if doctor_display:
            lines = [f"По врачу {doctor_display} нашёл такие варианты стоимости:"]
        else:
            lines = ["Нашёл такие варианты стоимости по выбранному врачу:"]
        for i, item in enumerate(rows, 1):
            lines.append(f"{i}. {_line(item)}")
        lines.append("Если нужен точный вариант, уточните филиал.")
        return _finish("\n".join(lines))

    if len(rows) == 1:
        return _finish(_line(rows[0]))

    lines = ["Нашёл варианты по стоимости:"]
    for i, item in enumerate(rows, 1):
        lines.append(f"{i}. {_line(item)}")
    lines.append("Если нужен точный вариант, уточните врача или филиал.")
    return _finish("\n".join(lines))


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
