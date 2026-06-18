"""Deterministic rendering helpers for FreeTalk tool payloads."""

from __future__ import annotations

from datetime import datetime
import re
from typing import Any

_MONTHS_RU = {
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

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_HHMM_RE = re.compile(r"^\d{1,2}:\d{2}$")


def top_list(values: list[Any], limit: int = 5) -> list[Any]:
    return list(values[: max(1, limit)])


def format_iso_date_short(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        parsed = datetime.fromisoformat(raw)
        return parsed.strftime("%d.%m")
    except Exception:
        pass
    if len(raw) >= 10:
        return raw[:10]
    return raw


def format_iso_date_human(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        parsed = datetime.fromisoformat(raw)
        return f"{parsed.day} {_MONTHS_RU.get(parsed.month, parsed.strftime('%m'))}"
    except Exception:
        pass
    return raw


def schedule_day_line(day: dict[str, Any]) -> tuple[str, bool]:
    if not isinstance(day, dict):
        return "", False

    date_label = format_iso_date_human(str(day.get("date") or ""))
    slots_raw = day.get("slots")
    slots: list[str] = []
    seen_slots: set[str] = set()
    if isinstance(slots_raw, list):
        for item in slots_raw:
            text = str(item or "").strip()
            if not text:
                continue
            value = text[:5] if len(text) >= 5 else text
            if value in seen_slots:
                continue
            seen_slots.add(value)
            slots.append(value)

    if slots:
        title = date_label or "Ближайшая дата"
        return f"{title}: {', '.join(top_list(slots, limit=6))}", True

    start = str(day.get("start") or "").strip()
    end = str(day.get("end") or "").strip()
    if start or end:
        title = date_label or "Ближайшая дата"
        interval = f"{start[:5] if start else ''}-{end[:5] if end else ''}".strip("-")
        if interval:
            return f"{title}: {interval}", False

    return "", False


def render_schedule_details(payload: dict[str, Any]) -> str:
    schedule = payload.get("schedule")
    if not isinstance(schedule, list) or not schedule:
        return ""

    lines: list[str] = ["Нашел расписание:"]
    rendered = 0
    has_free_slots = False
    has_filter = _has_schedule_filter(payload)
    for row in top_list(schedule, 3):
        if not isinstance(row, dict):
            continue

        rendered += 1
        fio = str(row.get("fio") or "").strip() or "Врач"
        lines.append(f"{rendered}. {fio}")

        row_schedule = row.get("schedule")
        row_lines = 0
        if isinstance(row_schedule, dict):
            for region_name, days in list(row_schedule.items())[:2]:
                if not _schedule_region_matches_filter(str(region_name or ""), payload):
                    continue
                day_lines: list[str] = []
                filtered_days = _filtered_schedule_days(days, payload)
                if filtered_days:
                    for day in filtered_days[:4]:
                        preview, day_has_free = schedule_day_line(day)
                        if not preview:
                            continue
                        day_lines.append(preview)
                        if day_has_free:
                            has_free_slots = True
                if not day_lines:
                    continue

                region = str(region_name or "").strip()
                if region:
                    lines.append(f"   {region}")
                for preview in day_lines:
                    lines.append(f"   • {preview}")
                row_lines += len(day_lines)

        if row_lines == 0:
            if has_filter:
                filter_label = _schedule_filter_label(payload)
                suffix = f" ({filter_label})" if filter_label else ""
                lines.append(f"   По выбранным фильтрам{suffix} свободные окна по этому врачу не найдены.")
            else:
                lines.append("   Свободные окна по этому врачу не найдены, уточните дату или филиал.")
        lines.append("")

    if rendered == 0:
        return ""

    reason = str(payload.get("schedule_unavailable_reason") or "").strip().lower()
    if not has_free_slots and reason == "no_free_slots_2_weeks":
        lines.append("На ближайшие две недели свободных слотов по этому запросу нет.")
    elif has_free_slots:
        lines.append("Если хотите записаться, выберите дату и время из предложенных, и я продолжу запись.")

    return "\n".join([line for line in lines if str(line).strip()]).strip()


def _schedule_filter_entities(payload: dict[str, Any]) -> dict[str, str]:
    entities = payload.get("entities_used_ft") if isinstance(payload, dict) else {}
    if not isinstance(entities, dict):
        return {}
    out: dict[str, str] = {}
    for key in ("branch_name", "date", "date_from", "date_to", "time", "time_from", "time_to"):
        value = str(entities.get(key) or "").strip()
        if value:
            out[key] = value
    return out


def _has_schedule_filter(payload: dict[str, Any]) -> bool:
    return bool(_schedule_filter_entities(payload))


def _schedule_filter_label(payload: dict[str, Any]) -> str:
    filters = _schedule_filter_entities(payload)
    parts: list[str] = []
    branch = str(filters.get("branch_name") or "").strip()
    if branch:
        parts.append(f"филиал: {branch}")

    date = str(filters.get("date") or "").strip()
    date_from = str(filters.get("date_from") or "").strip()
    date_to = str(filters.get("date_to") or "").strip()
    if date and date not in {date_from, date_to}:
        parts.append(f"дата: {date}")
    elif date_from and date_to and date_from != date_to:
        parts.append(f"даты: {date_from} - {date_to}")
    elif date_from or date_to:
        parts.append(f"дата: {date_from or date_to}")

    time_value = str(filters.get("time") or "").strip()
    time_from = str(filters.get("time_from") or "").strip()
    time_to = str(filters.get("time_to") or "").strip()
    if time_value:
        parts.append(f"время: {time_value}")
    elif time_from and time_to:
        parts.append(f"время: {time_from}-{time_to}")
    elif time_from:
        parts.append(f"время: с {time_from}")
    elif time_to:
        parts.append(f"время: до {time_to}")
    return ", ".join(parts)


def _schedule_region_matches_filter(region_name: str, payload: dict[str, Any]) -> bool:
    branch_name = _schedule_filter_entities(payload).get("branch_name", "")
    if not branch_name:
        return True
    region_norm = _normalize_filter_text(region_name)
    branch_norm = _normalize_filter_text(branch_name)
    return bool(region_norm == branch_norm or branch_norm in region_norm or region_norm in branch_norm)


def _filtered_schedule_days(days: Any, payload: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(days, list):
        return []
    filters = _schedule_filter_entities(payload)
    grouped: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for day in days:
        if not isinstance(day, dict):
            continue
        date_value = str(day.get("date") or "").strip()
        if not _schedule_date_matches_filter(date_value, filters):
            continue
        filtered = _schedule_day_with_time_filter(day, filters)
        if not filtered:
            continue
        key = str(filtered.get("date") or date_value or len(order))
        if key not in grouped:
            grouped[key] = dict(filtered)
            order.append(key)
            grouped[key]["slots"] = _dedupe_slots(grouped[key].get("slots"))
            continue
        existing = grouped[key]
        merged_slots = _dedupe_slots(list(existing.get("slots") or []) + list(filtered.get("slots") or []))
        if merged_slots:
            existing["slots"] = merged_slots
        if not str(existing.get("start") or "").strip() and str(filtered.get("start") or "").strip():
            existing["start"] = filtered.get("start")
        if not str(existing.get("end") or "").strip() and str(filtered.get("end") or "").strip():
            existing["end"] = filtered.get("end")
    return [grouped[key] for key in order]


def _schedule_day_with_time_filter(day: dict[str, Any], filters: dict[str, str]) -> dict[str, Any]:
    out = dict(day)
    slots_raw = day.get("slots")
    has_time_filter = bool(filters.get("time") or filters.get("time_from") or filters.get("time_to"))
    if isinstance(slots_raw, list):
        slots = _dedupe_slots(slots_raw)
        if has_time_filter:
            slots = [slot for slot in slots if _schedule_time_matches_filter(slot, filters)]
        if not slots:
            return {}
        out["slots"] = slots
        return out
    if has_time_filter and not _schedule_interval_matches_filter(day, filters):
        return {}
    return out


def _schedule_date_matches_filter(date_value: str, filters: dict[str, str]) -> bool:
    if not filters.get("date") and not filters.get("date_from") and not filters.get("date_to"):
        return True
    if not _ISO_DATE_RE.match(date_value):
        return False
    exact = filters.get("date", "")
    if exact and exact not in {"weekend", "next_week", "this_week"} and _ISO_DATE_RE.match(exact):
        return date_value == exact
    date_from = filters.get("date_from", "")
    date_to = filters.get("date_to", "") or date_from
    if date_from and date_to and _ISO_DATE_RE.match(date_from) and _ISO_DATE_RE.match(date_to):
        return date_from <= date_value <= date_to
    if exact == "weekend":
        try:
            return datetime.fromisoformat(date_value).weekday() >= 5
        except Exception:
            return False
    return True


def _schedule_time_matches_filter(time_value: str, filters: dict[str, str]) -> bool:
    value = str(time_value or "").strip()[:5]
    if not _HHMM_RE.match(value):
        return False
    requested = filters.get("time", "")
    time_from = filters.get("time_from", "")
    time_to = filters.get("time_to", "")
    if requested and _HHMM_RE.match(requested) and not time_to:
        return value == requested[:5]
    if time_from and value < time_from[:5]:
        return False
    if time_to and value > time_to[:5]:
        return False
    return True


def _schedule_interval_matches_filter(day: dict[str, Any], filters: dict[str, str]) -> bool:
    start = str(day.get("start") or "").strip()[:5]
    end = str(day.get("end") or "").strip()[:5]
    if not (start or end):
        return False
    requested = filters.get("time", "")
    time_from = filters.get("time_from", "")
    time_to = filters.get("time_to", "")
    if requested and _HHMM_RE.match(requested) and not time_to:
        return (not start or start <= requested[:5]) and (not end or requested[:5] <= end)
    lower = time_from[:5] if time_from else start
    upper = time_to[:5] if time_to else end
    if start and upper and upper < start:
        return False
    if end and lower and lower > end:
        return False
    return True


def _dedupe_slots(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in values:
        value = str(item or "").strip()
        if not value:
            continue
        value = value[:5] if len(value) >= 5 else value
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _normalize_filter_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower().replace("ё", "е"))


def schedule_payload_stats(payload: dict[str, Any]) -> dict[str, Any]:
    schedule = payload.get("schedule")
    doctors_count = 0
    regions_count = 0
    days_count = 0
    slots_count = 0
    if isinstance(schedule, list):
        for row in schedule:
            if not isinstance(row, dict):
                continue
            doctors_count += 1
            row_schedule = row.get("schedule")
            if not isinstance(row_schedule, dict):
                continue
            for _, days in row_schedule.items():
                regions_count += 1
                if not isinstance(days, list):
                    continue
                for day in days:
                    if not isinstance(day, dict):
                        continue
                    days_count += 1
                    day_slots = day.get("slots")
                    if isinstance(day_slots, list):
                        slots_count += sum(1 for slot in day_slots if str(slot or "").strip())
    return {
        "doctors_count": doctors_count,
        "regions_count": regions_count,
        "days_count": days_count,
        "slots_count": slots_count,
        "schedule_unavailable_reason": str(payload.get("schedule_unavailable_reason") or "").strip(),
    }


def fallback_render(tool_name: str, payload: dict[str, Any]) -> str:
    clarify_text = str(payload.get("clarify_text") or "").strip()
    if clarify_text:
        return clarify_text

    handoff_text = str(payload.get("handoff_message") or "").strip()
    if handoff_text:
        return handoff_text

    if tool_name == "test_result_status":
        missing = payload.get("missing_fields") or []
        if missing:
            return "Чтобы проверить результат, уточните: " + ", ".join(str(x) for x in missing)
        links = [str(x).strip() for x in (payload.get("result_links") or []) if str(x).strip()]
        preview = str(payload.get("result_preview") or "").strip()
        if preview:
            if links:
                return f"{preview}\n\nОткрыть результат: {links[0]}"
            return preview
        if payload.get("ready") and links:
            return f"Результат готов. Ссылка: {links[0]}"
        return "По указанным данным результат пока не найден или еще не готов."

    if tool_name == "test_prepare":
        text = str(payload.get("prepare") or "").strip()
        if text:
            return text

    if tool_name == "price_info":
        prices = [x for x in (payload.get("prices") or []) if isinstance(x, dict)]
        if prices:
            lines = ["Нашел цены по запросу:"]
            for row in top_list(prices, 5):
                name = str(row.get("serviceName") or row.get("name") or "").strip()
                cost = row.get("cost")
                if name and cost:
                    lines.append(f"- {name}: {cost} ₽")
                elif name:
                    lines.append(f"- {name}")
            return "\n".join(lines)
        return "По вашему запросу цены в данных клиники не найдены."

    if tool_name == "test_assist":
        tests = [x for x in (payload.get("tests") or []) if isinstance(x, dict)]
        if tests:
            lines = ["Подходящие анализы:"]
            for row in top_list(tests, 5):
                name = str(row.get("serviceName") or row.get("name") or "").strip()
                cost = row.get("cost")
                if name and cost:
                    lines.append(f"- {name}: {cost} ₽")
                elif name:
                    lines.append(f"- {name}")
            return "\n".join(lines)
        return "Подходящие анализы в данных клиники не найдены."

    if tool_name == "address_info":
        addresses = [str(x).strip() for x in (payload.get("addresses") or []) if str(x).strip()]
        branches = [x for x in (payload.get("branches") or []) if isinstance(x, dict)]
        if branches:
            has_contact = any(
                str(row.get("phone") or row.get("work_time") or "").strip()
                for row in branches
            )
            lines = ["Нашел контакты филиалов:" if has_contact else "Нашел адреса филиалов:"]
            seen: set[str] = set()
            for row in top_list(branches, 6):
                addr = str(row.get("address") or row.get("name") or "").strip()
                if not addr or addr in seen:
                    continue
                seen.add(addr)
                lines.append(f"- {addr}")
                phone = str(row.get("phone") or "").strip()
                work_time = str(row.get("work_time") or "").strip()
                if phone:
                    lines.append(f"  Телефон: {phone}")
                if work_time:
                    lines.append(f"  График: {work_time}")
            if len(lines) > 1:
                return "\n".join(lines)
        if addresses:
            lines = ["Нашел адреса филиалов:"]
            lines.extend(f"- {addr}" for addr in top_list(addresses, 6))
            return "\n".join(lines)
        return "Адреса по вашему запросу в данных клиники не найдены."

    if tool_name == "doctors_info":
        doctors = [x for x in (payload.get("doctors") or []) if isinstance(x, dict)]
        if doctors:
            lines = ["Нашел врачей по запросу:"]
            for doc in top_list(doctors, 6):
                fio = str(doc.get("fio") or "").strip()
                spec = str(doc.get("specialization") or "").strip()
                if fio and spec:
                    lines.append(f"- {fio} ({spec})")
                elif fio:
                    lines.append(f"- {fio}")
            return "\n".join(lines)
        return "Врачей по вашему запросу в данных клиники не найдено."

    if tool_name == "doctors_schedule_week":
        detailed = render_schedule_details(payload)
        if detailed:
            return detailed
        reason = str(payload.get("schedule_unavailable_reason") or "").strip().lower()
        if reason == "no_free_slots_2_weeks":
            return "На ближайшие две недели свободных слотов по этому запросу нет."
        return "Расписание по вашему запросу в данных клиники не найдено."

    if tool_name == "service_bundle_info":
        prices = [x for x in (payload.get("retail_prices") or []) if isinstance(x, dict)]
        doctors = [x for x in (payload.get("doctors") or []) if isinstance(x, dict)]
        lines: list[str] = []
        if prices:
            top = prices[0]
            name = str(top.get("serviceName") or top.get("name") or "").strip()
            cost = top.get("cost")
            if name and cost:
                lines.append(f"Услуга: {name}. Цена от {cost} ₽.")
        if doctors:
            lines.append("Врачи по услуге:")
            for doc in top_list(doctors, 4):
                fio = str(doc.get("fio") or "").strip()
                if fio:
                    lines.append(f"- {fio}")
        prepare = str(payload.get("prepare") or "").strip()
        if prepare:
            lines.append("")
            lines.append("Подготовка:")
            lines.append(prepare)
        if lines:
            return "\n".join(lines)
        return "По этой услуге в данных клиники сейчас нет релевантной информации."

    if tool_name == "main_index_info":
        content = str(payload.get("content") or "").strip()
        if content:
            return content
        return "По вашему запросу в базе знаний клиники ничего не найдено."

    if tool_name == "news_info":
        news = [x for x in (payload.get("news") or []) if isinstance(x, dict)]
        if news:
            lines = ["Нашел новости клиники:"]
            for item in top_list(news, 5):
                title = str(item.get("title") or item.get("name") or "").strip()
                url = str(item.get("url") or item.get("link") or "").strip()
                if title and url:
                    lines.append(f"- {title} ({url})")
                elif title:
                    lines.append(f"- {title}")
                elif url:
                    lines.append(f"- {url}")
            return "\n".join(lines)
        return "Новости по вашему запросу не найдены."

    if tool_name == "web_search":
        results = [x for x in (payload.get("results") or []) if isinstance(x, dict)]
        if results:
            lines = ["Нашел в интернете:"]
            for row in top_list(results, 5):
                title = str(row.get("title") or "").strip()
                url = str(row.get("url") or "").strip()
                snippet = str(row.get("snippet") or "").strip()
                source = str(row.get("source") or "").strip()
                if title and url:
                    lines.append(f"- {title} ({url})")
                elif title:
                    lines.append(f"- {title}")
                elif url:
                    lines.append(f"- {url}")
                if snippet:
                    lines.append(f"  {snippet}")
                if source:
                    lines.append(f"  Источник поиска: {source}")
            lines.append("")
            lines.append("Это общая информация из интернет-поиска, не из данных клиники.")
            return "\n".join(lines)
        return "В интернет-поиске по этому запросу не найдено релевантных результатов."

    content = str(payload.get("content") or "").strip()
    if content:
        return content
    return "Нашел данные, но не удалось корректно сформировать ответ. Уточните вопрос, и я попробую точнее."
