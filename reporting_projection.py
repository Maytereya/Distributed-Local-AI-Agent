"""Strict boundary for aggregate reporting: fixed fields and numeric values."""
from datetime import datetime
import math
import re
from zoneinfo import ZoneInfo


def _number(value,percent=False):
    if percent and value is None: return
    if type(value) not in (int,float) or not math.isfinite(value) or not 0 <= value <= (100 if percent else 2**63-1):
        raise ValueError("invalid_reporting_number")
    if not percent and type(value) is not int: raise ValueError("invalid_reporting_count")


def _date(value):
    if not isinstance(value,str) or len(value)>40 or datetime.fromisoformat(value).tzinfo is None:
        raise ValueError("invalid_reporting_time")


def validate_counts(value):
    fields={"total","outcomes","routes","completed_known","success_percent","partial_percent","patients","unit_unknown","refusal_events","success_kinds"}
    if not isinstance(value,dict) or set(value)!=fields: raise ValueError("invalid_reporting_fields")
    if not isinstance(value["outcomes"],dict) or not isinstance(value["routes"],dict) or set(value["outcomes"])!=set("SPURHWX") or set(value["routes"])!={"bot","operator","unknown"}:
        raise ValueError("invalid_reporting_fields")
    for k in fields-{"outcomes","routes","success_kinds"}: _number(value[k],k.endswith("percent"))
    if not isinstance(value["success_kinds"],dict) or set(value["success_kinds"])!={"tax_link","results_link","structured_answer"}:
        raise ValueError("invalid_reporting_fields")
    for count in value["success_kinds"].values():
        _number(count)
        if count>value["outcomes"]["S"]: raise ValueError("reporting_balance_mismatch")
    for counts in (value["outcomes"],value["routes"]):
        for count in counts.values(): _number(count)
    if sum(value["outcomes"].values())!=value["total"] or sum(value["routes"].values())!=value["total"] or value["routes"]["operator"]!=value["outcomes"]["H"]:
        raise ValueError("reporting_balance_mismatch")
    if value["completed_known"] != sum(value["outcomes"][key] for key in "SPURH"):
        raise ValueError("reporting_balance_mismatch")
    for key,outcome in (("success_percent","S"),("partial_percent","P")):
        expected = round(100*value["outcomes"][outcome]/value["completed_known"],1) if value["completed_known"] else None
        if value[key] != expected: raise ValueError("reporting_balance_mismatch")
    return value


def validate_snapshot(value):
    if not isinstance(value,dict) or type(value.get("version")) is not int or value.get("version") != 1 or type(value.get("enabled")) is not bool:
        raise ValueError("invalid_reporting_snapshot")
    if not value["enabled"]:
        if set(value)!={"version","enabled","status"} or value["status"]!="not_started": raise ValueError("invalid_reporting_snapshot")
        return value
    fields={"version","enabled","status","activation","generated_at","worker_at","start","end","partial_period","period","cumulative","previous","operator_queue","patient_delivery","analyzer_quality_passed"}
    if set(value)!=fields or value["status"] not in {"fresh","stale"}: raise ValueError("invalid_reporting_snapshot")
    for key in ("activation","generated_at","start","end"): _date(value[key])
    if value["worker_at"] is not None: _date(value["worker_at"])
    for key in ("partial_period","analyzer_quality_passed"):
        if type(value[key]) is not bool: raise ValueError("invalid_reporting_flag")
    for key in ("period","cumulative"): validate_counts(value[key])
    if value["previous"] is not None: validate_counts(value["previous"])
    _number(value["operator_queue"])
    delivery=value["patient_delivery"]
    if not isinstance(delivery,dict) or set(delivery)!={"last_accepted_at","unconfirmed_period"}: raise ValueError("invalid_reporting_fields")
    if delivery["last_accepted_at"] is not None: _date(delivery["last_accepted_at"])
    _number(delivery["unconfirmed_period"])
    return value


def corporate_card(snapshot):
    s=validate_snapshot(snapshot)
    if not s["enabled"]: return "Корпоративный чатбот\nНовый учёт обращений ещё не включён."
    tz=ZoneInfo("Europe/Samara")
    stamp=lambda v:datetime.fromisoformat(v).astimezone(tz).strftime("%d.%m.%Y %H:%M")
    p,c=s["period"],s["cumulative"]
    o,r=p["outcomes"],p["routes"]
    pct=lambda v:"нет данных" if v is None else f"{v:g}%"
    lines=["Корпоративный чатбот",f"Период: {stamp(s['start'])} — {stamp(s['end'])}, Самара",
        f"Данные обновлены: {stamp(s['generated_at'])}"]
    if s["partial_period"]: lines.append(f"Неполный период — учёт включён {stamp(s['activation'])}.")
    if s["status"]!="fresh": lines.append("! Нет свежего подтверждения работы сборщика; итоги могут быть неполными.")
    lines += ["","──── Обращения за период ────",f"Обращений: {p['total']}",f"Пациентов обратились: {p['patients']}",
        f"Передано оператору: {r['operator']}; доля от всех обращений: {pct(round(100*r['operator']/p['total'],1) if p['total'] else None)}",f"Без передачи оператору: {r['bot']}",f"Маршрут не определён: {r['unknown']}",
        "","──── Результаты обращений ────",f"Бот дал нужный ответ или ссылку: {o['S']}",f"Частично решено: {o['P']}",
        f"Вопрос остался без ответа по существу: {o['U']}",f"Пациент отказался от оператора, вопрос не решён: {o['R']}",
        f"Передано оператору: {o['H']}",f"Ещё в работе: {o['W']}",f"Результат пока не определён: {o['X']}",
        f"Полностью решено ботом: {pct(p['success_percent'])}",f"Частично решено: {pct(p['partial_percent'])}",
        f"Доли среди завершённых с известным итогом: {p['completed_known']}.",
        f"Событий отказа от оператора: {p['refusal_events']}"]
    kinds=p["success_kinds"]
    lines += [f"Среди полных решений — ссылка на справку: {kinds['tax_link']}; результаты анализов: {kinds['results_link']}; другие проверяемые ответы: {kinds['structured_answer']}.",
              "В составном обращении виды успешного ответа могут пересекаться."]
    if p["unit_unknown"]: lines.append(f"Границы обращения требуют проверки: {p['unit_unknown']}.")
    lines += ["","──── С начала учёта ────",f"Учёт с {stamp(s['activation'])}",f"Обращений всего: {c['total']}",
        f"Без передачи оператору: {c['routes']['bot']}",f"Передано оператору: {c['routes']['operator']}",
        f"Маршрут не определён: {c['routes']['unknown']}",f"Полностью решено: {c['outcomes']['S']}",
        f"Частично решено: {c['outcomes']['P']}",f"Ожидают оператора сейчас: {s['operator_queue']}"]
    prev=s["previous"]
    if prev is None: lines.append("Сравнение с предыдущим периодом пока недоступно.")
    elif p["success_percent"] is not None and prev["success_percent"] is not None:
        lines += [f"К предыдущему периоду: обращений {p['total']-prev['total']:+d}; доля полного решения {p['success_percent']-prev['success_percent']:+.1f} п. п."]
    delivery=s["patient_delivery"]
    lines += ["","──── Доставка пациентских ответов ────",
        "Последнее подтверждение Telegram: "+(stamp(delivery["last_accepted_at"]) if delivery["last_accepted_at"] else "нет данных с начала учёта"),
        f"Отправок без подтверждения за период: {delivery['unconfirmed_period']}."]
    return "\n".join(lines)


CAUSES = {"requested_operator":"Пациент запросил оператора","missing_data":"Нет данных для ответа",
    "clarification_abandoned":"Уточнение осталось без ответа","refused_operator":"Пациент отказался от оператора",
    "partial_answer":"Получен ответ на часть вопросов","irrelevant_answer":"Ответ не соответствует вопросу",
    "technical_error":"Техническая ошибка","delivery_unconfirmed":"Доставка не подтверждена",
    "unsupported_scenario":"Сценарий не поддерживается","unknown":"Недостаточно данных для вывода"}
ACTIONS = {"none":"Не требуется","check_data":"Проверить источник данных","check_scenario":"Проверить сценарий бота",
    "check_delivery":"Проверить доставку","review_manually":"Проверить вручную"}
CASE_STATES = {"pending","typed_evidence","model_preview","model_checked","unavailable"}
CASE_BASIS={"pending":"Разбор ещё не выполнен","typed_evidence":"События системы",
    "model_preview":"Локальный ИИ; требует проверки","model_checked":"Локальный ИИ после проверки качества",
    "unavailable":"Данных для разбора нет или анализатор недоступен"}
TASK_LABELS={"PRICE":"Стоимость","DOCTOR_SCHEDULE":"Расписание врача","DOCTOR_INFO":"Информация о враче",
    "TEST_RESULT":"Результаты анализов","TAX_DOCUMENT":"Справка для вычета","APPOINTMENT":"Запись на приём",
    "ADDRESS":"Адрес филиала","PREPARE":"Подготовка","TEST_ASSIST":"Выбор анализа","NEWS":"Новости и акции",
    "OTHER":"Другой вопрос","URGENT":"Срочное обращение","COMPLAINT":"Жалоба","MEDICAL_ADVICE":"Медицинский вопрос"}
EVENT_LABELS={"patient_received":"Входящие сообщения","scenario_evidence":"Признаки сценария",
    "delivery_accepted":"Подтверждённые отправки","delivery_uncertain":"Отправки без подтверждения",
    "delivery_failed":"Отклонённые отправки","actual_handoff":"Фактическая передача оператору",
    "operator_offered":"Предложение оператора","operator_declined":"Отказ от оператора",
    "operator_accepted":"Согласие на оператора","inactivity_closed":"Закрытие после 6 часов"}
OUTCOME_LABELS={"P":"Частично решено","U":"Без ответа по существу","R":"Отказ от оператора, вопрос не решён",
    "H":"Передано оператору","X":"Результат не определён"}


def validate_cases(value):
    if not isinstance(value,dict) or set(value)!={"version","enabled","quality_passed","cases","total","truncated"} or type(value["version"]) is not int or value["version"]!=1:
        raise ValueError("invalid_failure_projection")
    for k in ("enabled","quality_passed","truncated"):
        if type(value[k]) is not bool: raise ValueError("invalid_failure_projection")
    _number(value["total"])
    if not isinstance(value["cases"],list) or len(value["cases"])>100: raise ValueError("invalid_failure_projection")
    if not value["enabled"] and (value["cases"] or value["total"] or value["quality_passed"] or value["truncated"]):
        raise ValueError("invalid_failure_projection")
    for case in value["cases"]:
        if not isinstance(case,dict) or set(case)!={"id","outcome","cause","action","status","confidence","tasks","events"}:
            raise ValueError("invalid_failure_projection")
        _number(case["id"])
        if case["outcome"] not in set("PURHX") or case["cause"] not in CAUSES or case["action"] not in ACTIONS or case["status"] not in CASE_STATES:
            raise ValueError("invalid_failure_projection")
        _number(case["confidence"],True)
        if not isinstance(case["tasks"],dict) or len(case["tasks"])>8: raise ValueError("invalid_failure_projection")
        for key,task in case["tasks"].items():
            if not isinstance(key,str) or not re.fullmatch(r"[A-Z_]+(?::[1-8])?",key) or key.split(":")[0] not in TASK_LABELS: raise ValueError("invalid_failure_projection")
            if not isinstance(task,dict) or set(task)!={"state","kind"} or task["state"] not in {"solved","pending","unknown"} or task["kind"] not in {"tax_link","results_link","structured_answer","none"}: raise ValueError("invalid_failure_projection")
        if not isinstance(case["events"],dict) or set(case["events"])!=set(EVENT_LABELS): raise ValueError("invalid_failure_projection")
        for count in case["events"].values(): _number(count)
    return value


def maintenance_priority(tasks,cause,delivery_issue=False):
    # This never changes medical urgency, queue order or the appeal's outcome.
    if any(k.split(":")[0]=="URGENT" for k in tasks):
        return "Высокий: проверить срочный сценарий вручную"
    if delivery_issue or cause in {"technical_error","delivery_unconfirmed"}:
        return "Высокий: технический сбой или доставка"
    if cause in {"missing_data","irrelevant_answer","unsupported_scenario","partial_answer"}:
        return "Обычный: проверить данные или сценарий"
    if cause=="requested_operator":
        return "Ошибка бота не установлена: пациент выбрал оператора"
    return "Требуется ручная оценка"


def observed_causes(case):
    causes={case["cause"]}
    if case["events"]["delivery_failed"]: causes.add("technical_error")
    if case["events"]["delivery_uncertain"]: causes.add("delivery_unconfirmed")
    if case["events"]["operator_declined"] and case["outcome"]=="R": causes.add("refused_operator")
    return [key for key in CAUSES if key in causes]


def failure_card(data,detail=False):
    d=validate_cases(data)
    if not d["enabled"]: return "Разбор проблемных обращений\nНовый учёт обращений ещё не включён."
    lines=["Разбор проблемных обращений",f"Обращений в выборке: {d['total']}"]
    if not d["quality_passed"]: lines.append("Пробный режим: качество ИИ-анализатора ещё не подтверждено вручную.")
    if d["truncated"]: lines.append("Показаны последние 100 обращений; сузьте период.")
    if detail:
        for c in d["cases"]:
            solved=[TASK_LABELS[k.split(":")[0]] for k,t in c["tasks"].items() if t["state"]=="solved"]
            pending=[TASK_LABELS[k.split(":")[0]] for k,t in c["tasks"].items() if t["state"]!="solved"]
            lines += ["",f"Обращение №{c['id']}",f"Итог: {OUTCOME_LABELS[c['outcome']]}",
                "Решено: "+(", ".join(solved) or "нет подтверждённого решения"),
                "Не решено или не подтверждено: "+(", ".join(pending) or "тип вопроса не определён"),
                "Ожидаемый результат: ответ по всем запрошенным вопросам или нужная ссылка, с подтверждением отправки.",
                "Подтверждённые события:"]
            lines += [f"{EVENT_LABELS[k]}: {v}" for k,v in c["events"].items() if v]
            lines += [f"Причина: {CAUSES[c['cause']]}",f"Действие: {ACTIONS[c['action']]}",
                f"Основание: {CASE_BASIS[c['status']]}",
                "Уверенность: "+({25:"низкая",50:"средняя",75:"высокая"}.get(c['confidence'],"нет данных"))+
                    (" — самооценка ИИ, требует проверки." if c['confidence'] is not None else ".")]
            additional=[CAUSES[k] for k in observed_causes(c) if k!=c["cause"]]
            if additional: lines.append("Дополнительные признаки по событиям: "+"; ".join(additional)+".")
            lines += ["Приоритет проверки исправлений: "+maintenance_priority(c["tasks"],c["cause"],bool(c["events"]["delivery_failed"] or c["events"]["delivery_uncertain"])),
                "Временная шкала, оригиналы и статус ручной проверки — в панели, вход действующим оператором через VPN:",
                f"http://172.16.0.28:8080/reporting/appeals/{c['id']}/"]
    else:
        lines += [f"В показанной выборке обращений: {len(d['cases'])}",
            f"Неуспешных или с неизвестным итогом: {sum(c['outcome'] in set('URHX') for c in d['cases'])}",
            f"Частично решённых: {sum(c['outcome']=='P' for c in d['cases'])}",
            f"Разобрано по событиям или ИИ: {sum(c['status'] in {'typed_evidence','model_preview','model_checked'} for c in d['cases'])}",
            f"Требуют ручной проверки: {sum(c['status']!='typed_evidence' for c in d['cases'])}",
            "Показано текущее состояние обращений, поступивших за выбранный период."]
        counts={}
        for c in d["cases"]:
            for cause in observed_causes(c): counts[cause]=counts.get(cause,0)+1
        lines += [f"{CAUSES[k]}: {v}" for k,v in sorted(counts.items(),key=lambda x:-x[1])]
        if counts: lines.append("Категории могут пересекаться: основная причина плюс подтверждённые события доставки/отказа.")
        priorities={}
        for c in d["cases"]:
            priority=maintenance_priority(c["tasks"],c["cause"],bool(c["events"]["delivery_failed"] or c["events"]["delivery_uncertain"]))
            priorities[priority]=priorities.get(priority,0)+1
        if priorities:
            lines.append("──── Приоритет проверки исправлений ────")
            lines += [f"{k}: {v}" for k,v in sorted(priorities.items(),key=lambda x:(0 if x[0].startswith('Высокий') else 1 if x[0].startswith('Обычный') else 2,-x[1]))]
            lines.append("Приоритет относится к проверке сценария бота.")
        if d["cases"]: lines.append("Карточки: "+", ".join('/failure '+str(c['id']) for c in d['cases'][:10]))
    lines.append("Тексты пациентов и ответы бота остаются в защищённой панели операторов.")
    return "\n".join(lines)
