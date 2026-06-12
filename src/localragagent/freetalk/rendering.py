"""Deterministic rendering helpers for FreeTalk tool payloads."""

from __future__ import annotations

from datetime import datetime
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
    if isinstance(slots_raw, list):
        for item in slots_raw:
            text = str(item or "").strip()
            if not text:
                continue
            slots.append(text[:5] if len(text) >= 5 else text)

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
                day_lines: list[str] = []
                if isinstance(days, list):
                    for day in days[:4]:
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
