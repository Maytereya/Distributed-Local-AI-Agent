"""Успех внутри меню-кнопок: доля решённых вопросов по листу меню и в целом (метрика «рабочий бот», 25.10).

Решение владельца 25.10: ~90% пациентов идут через меню шлюза, главная метрика — успех при живой LLM внутри кнопок,
цель — 100%. Набор — вопросы эталона с листом, который пациент нажал бы перед вопросом
(`~/Dev/dla-private/bench/sets/buttons_v1.jsonl`; разметка листа — `labels/button_leaf/`). Ход набора — как у пациента:
нажатие листа, затем вопрос свободным текстом (`"press": true`, `evalbench.run`).

    ./venv/bin/python -m evalbench.card <прогон_было> <прогон_стало>          # раздел «Успех по кнопкам»
    ./venv/bin/python -m evalbench.buttons judge-export <прогон>... --out DIR   # пачки ответов без вердикта — судье
    ./venv/bin/python -m evalbench.buttons sample <прогон> --n 30 > FILE       # выборка вердиктов на проверку владельцу

Вердикт хода:
- `success` — ответ решает вопрос по данным мира;
- `handoff_ok` — честный перевод или оффер оператора там, где данных у бота нет по дизайну (существующая запись,
  ДМС, срок готовности); в успех входит — цель «никогда не ошибаться и не молчать», а не «всё самому»; карточка
  печатает его отдельно, чтобы рост переводов был виден;
- `fail` — с классом провала (`FAIL_CLASSES`);
- `unjudged` — нужен судья, вердикта для ЭТОГО ответа ещё нет (ответ изменился — судить заново).
Вне знаменателя: `excluded` (другой город — перевод по ТЗ; без предыдущих реплик не понять), `by_design` (лист-перевод:
кнопка сама отдаёт оператору, второго хода нет).

Цена — без судьи, по разметке допустимых serviceId (`prices.price_verdict`): успех — `correct` / `list_hit`, на услугу,
которой нет в прайсе (`absent`), — ответ без цены. Технический сбой и служебный текст — провал без судьи. Остальное —
судья-LLM по ответу и данным мира (`JUDGE_INSTRUCTIONS`); вердикт привязан к паре (id, отпечаток ответа) в
`labels/button_judge/*.jsonl`, поэтому неизменный ответ второй раз не судится. Против Гудхарта: набор — мерило; промпты
и правила под его вопросы не подгонять, критерий судьи — смысл ответа, а не совпадение с формулировкой.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .detectors import service_text
from .prices import PricePair, price_verdict
from .world import PRIVATE_ROOT

# «Таблица экранов, версия 1» — контракт со шлюзом: лист → подпись, которая приходит в `text` при нажатии.
GATEWAY_CAPTIONS: dict[str, str] = {
    "menu.address": "Адреса и часы работы",
    "menu.operator": "Оператор",
    "menu.appointment.book": "Записаться на приём",
    "menu.appointment.schedule": "Когда принимает врач",
    "menu.appointment.change": "Перенести или отменить",
    "menu.appointment.prepare": "Подготовка к процедуре",
    "menu.tests.where": "Где и до скольки принимают",
    "menu.tests.booking_needed": "Нужно ли записываться",
    "menu.tests.prepare": "Как подготовиться к анализу",
    "menu.tests.suggest": "Подобрать анализы",
    "menu.tests.price": "Сколько стоит анализ",
    "menu.results.lookup": "Ввести данные",
    "menu.results.missing": "Результат не пришёл",
    "menu.results.eta": "Когда будет готов",
    "menu.price.test": "Цена анализа",
    "menu.price.doctor": "Цена приёма врача",
    "menu.price.promo": "Акции и скидки",
    "menu.docs.tax": "Справка для вычета",
    "menu.docs.contract": "Копия договора, выписка",
    "menu.docs.sick_leave": "Больничный лист",
}

# Тема листа — по ней судья берёт критерий; цена судится по разметке цен.
LEAF_TOPIC: dict[str, str] = {
    "menu.appointment.book": "booking",
    "menu.appointment.schedule": "schedule",
    "menu.appointment.change": "change",
    "menu.appointment.prepare": "prepare",
    "menu.tests.where": "address",
    "menu.tests.booking_needed": "booking_needed",
    "menu.tests.prepare": "prepare",
    "menu.tests.suggest": "assist",
    "menu.tests.price": "price",
    "menu.results.lookup": "results",
    "menu.results.missing": "results",
    "menu.results.eta": "handoff",
    "menu.price.test": "price",
    "menu.price.doctor": "price",
    "menu.price.promo": "promo",
    "menu.address": "address",
    "menu.docs.tax": "docs",
    "menu.docs.contract": "handoff",
    "menu.docs.sick_leave": "handoff",
    "menu.operator": "handoff",
}

# Классы провала: цена — по разметке, сбой и служебный текст — по ответу, остальное называет судья (одно из списка).
FAIL_CLASSES: tuple[str, ...] = (
    "цена: чужая цена",
    "цена: одна цена вместо списка",
    "цена: уточнение вместо цены",
    "цена: перевод вместо цены",
    "цена: «не нашёл»",
    "цена: ответ без цены",
    "сбой",
    "служебный текст",
    "перевод на нажатии",
    # судья
    "шаблон вместо ответа",
    "чужая тема",
    "неверные данные",
    "выдумка",
    "лишний перевод",
    "нет перевода",
    "не понял / лишнее уточнение",
    "неполный ответ",
    "другое",
)
JUDGE_CLASSES = FAIL_CLASSES[FAIL_CLASSES.index("шаблон вместо ответа") :]
VERDICTS = ("success", "handoff_ok", "fail")

_PRICE_FAIL = {
    "foreign": "цена: чужая цена",
    "list_miss": "цена: чужая цена",
    "single_for_ambiguous": "цена: одна цена вместо списка",
}
_NO_PRICE_FAIL = {
    "clarify": "цена: уточнение вместо цены",
    "not_understood": "цена: уточнение вместо цены",
    "operator_offer": "цена: перевод вместо цены",
    "handoff": "цена: перевод вместо цены",
    "no_info": "цена: «не нашёл»",
}

JUDGE_DIR = PRIVATE_ROOT / "labels" / "button_judge"


def answer_sha(text: str) -> str:
    return hashlib.sha256(str(text or "").strip().encode("utf-8")).hexdigest()[:12]


def load_cases(header: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Строки набора прогона по id — из ТЕКУЩИХ файлов наборов: правка разметки листа не требует перепрогона."""

    out: dict[str, dict[str, Any]] = {}
    for ref in header.get("sets") or []:
        path = Path(ref).expanduser()
        if path.is_file():
            for line in path.open(encoding="utf-8"):
                if line.strip():
                    row = json.loads(line)
                    if row.get("press") and row.get("button"):
                        out[row["id"]] = row
    return out


def load_judge(folder: Path = JUDGE_DIR) -> dict[tuple[str, str], dict[str, Any]]:
    """Вердикты судьи: строка `{"id", "answer_sha", "verdict", "klass", "note", "by"}`; поздний файл перекрывает ранний."""

    out: dict[tuple[str, str], dict[str, Any]] = {}
    if not folder.is_dir():
        return out
    for path in sorted(folder.glob("*.jsonl")):
        for line in path.open(encoding="utf-8"):
            if line.strip():
                row = json.loads(line)
                if row.get("verdict") in VERDICTS:
                    out[(row["id"], row["answer_sha"])] = row
    return out


def turn_verdict(turn: dict[str, Any], case: dict[str, Any], price_label: dict[str, Any] | None, judged: dict[tuple[str, str], dict[str, Any]]) -> tuple[str, str | None]:
    """(вердикт, класс провала или причина исключения) одного хода набора кнопок."""

    if case.get("exclude"):
        return "excluded", str(case["exclude"])
    leaf = str(case["button"])
    topic = LEAF_TOPIC.get(leaf, "other")
    press = turn.get("press") or {}
    if press and not press.get("question_sent", True):
        if topic == "handoff" and press.get("handoff"):
            return "by_design", None
        return "fail", "сбой" if press.get("error") else "перевод на нажатии"
    if turn.get("harness_error") or turn.get("outcome") in ("tech_error", "handoff_tech"):
        return "fail", "сбой"
    answer = str(turn.get("answer") or "")
    if service_text(answer):
        return "fail", "служебный текст"
    kind = (price_label or {}).get("kind")
    if topic == "price" and kind == "context_dependent":
        return "excluded", "без контекста"
    pairs = [PricePair(**p) for p in turn.get("price_pairs") or []]
    # Цена приёма строкой врача («ФИО (Специальность) — N руб.») к строке прайса не привязана: разметка serviceId её не
    # видит — чья специальность, решает судья.
    by_doctor = any(p.status == "doctor" for p in pairs) and not any(p.status == "matched" for p in pairs)
    if topic == "price" and kind in ("single", "family", "multi", "absent") and not by_doctor:
        verdict = price_verdict(pairs, price_label)
        if verdict in ("correct", "list_hit") or (verdict == "no_price" and kind == "absent"):
            return "success", None
        if verdict == "no_price":
            return "fail", _NO_PRICE_FAIL.get(str(turn.get("outcome")), "цена: ответ без цены")
        return "fail", _PRICE_FAIL.get(str(verdict), "цена: чужая цена")
    hit = judged.get((turn["id"], answer_sha(answer)))
    if hit is None:
        return "unjudged", None
    return hit["verdict"], hit.get("klass") if hit["verdict"] == "fail" else None


def button_metrics(turns: list[dict[str, Any]], cases: dict[str, dict[str, Any]], price_labels: dict[str, dict[str, Any]], judged: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any] | None:
    """Счётчики раздела по листам и в целом; None — прогон не по набору кнопок."""

    per_leaf: dict[str, Counter] = defaultdict(Counter)
    classes: dict[str, list[str]] = defaultdict(list)
    excluded: Counter = Counter()
    verdicts: dict[str, tuple[str, str | None]] = {}
    for t in turns:
        case = cases.get(t["id"])
        if case is None:
            continue
        verdict, klass = turn_verdict(t, case, price_labels.get(t["id"]), judged)
        verdicts[t["id"]] = (verdict, klass)
        if verdict == "excluded":
            excluded[klass or "—"] += 1
            continue
        per_leaf[str(case["button"])][verdict] += 1
        if verdict == "fail":
            classes[klass or "другое"].append(t["id"])
    if not verdicts:
        return None
    total: Counter = Counter()
    for counts in per_leaf.values():
        total.update(counts)
    return {"per_leaf": {k: dict(v) for k, v in per_leaf.items()}, "total": dict(total), "classes": {k: sorted(v) for k, v in classes.items()}, "excluded": dict(excluded), "verdicts": verdicts}


def _denominator(counts: dict[str, int]) -> int:
    return sum(counts.get(v, 0) for v in (*VERDICTS, "unjudged"))


def _rate(counts: dict[str, int]) -> str:
    """«успех %» по судимым: success + handoff_ok из success + handoff_ok + fail; несудимые — отдельно."""

    judged = sum(counts.get(v, 0) for v in VERDICTS)
    if not judged:
        return "—"
    ok = counts.get("success", 0) + counts.get("handoff_ok", 0)
    return f"{round(100 * ok / judged)}% ({ok}/{judged})"


def render_buttons(runs: list[tuple[dict[str, Any], dict[str, Any] | None]]) -> list[str]:
    """Раздел карточки «Успех по кнопкам» — только счётчики и id, без реплик."""

    present = [(h, b) for h, b in runs if b]
    if not present:
        return []
    names = [h["name"] for h, _ in present]
    lines = ["", "## Успех по кнопкам", "", "Успех = решено ботом (success) + честный перевод там, где данных нет по дизайну (handoff_ok); % — от судимых.", ""]
    lines.append("| Лист | n | " + " | ".join(f"{n}: успех" for n in names) + " | " + " | ".join(f"{n}: перевод ok / провал / не судимо" for n in names) + " |")
    lines.append("|---|---|" + "---|" * (2 * len(present)))
    leaves = sorted({leaf for _, b in present for leaf in b["per_leaf"]}, key=lambda leaf: list(GATEWAY_CAPTIONS).index(leaf) if leaf in GATEWAY_CAPTIONS else 99)
    for leaf in [*leaves, None]:
        rows = [b["total"] if leaf is None else b["per_leaf"].get(leaf, {}) for _, b in present]
        title = "**в целом**" if leaf is None else f"{GATEWAY_CAPTIONS.get(leaf, leaf)} (`{leaf}`)"
        n = max(_denominator(r) + r.get("by_design", 0) for r in rows)
        rate = [("**" + _rate(r) + "**") if leaf is None else _rate(r) for r in rows]
        split = [f"{r.get('handoff_ok', 0)} / {r.get('fail', 0)} / {r.get('unjudged', 0)}" + (f" (+{r['by_design']} кнопка-перевод)" if r.get("by_design") else "") for r in rows]
        lines.append(f"| {title} | {n} | " + " | ".join(rate) + " | " + " | ".join(split) + " |")
    for h, b in present:
        if b["excluded"]:
            lines.append("")
            lines.append(f"Вне знаменателя ({h['name']}): " + ", ".join(f"{k} {v}" for k, v in sorted(b["excluded"].items())))
    for h, b in present:
        lines += ["", f"**Классы провалов {h['name']}** (по убыванию вклада):", ""]
        for klass, ids in sorted(b["classes"].items(), key=lambda kv: (-len(kv[1]), kv[0])):
            lines.append(f"- {klass}: {len(ids)} — " + ", ".join(ids[:12]) + (" …" if len(ids) > 12 else ""))
    if len(present) == 2:
        (_, before), (_, after) = present
        worse = sorted(i for i, (v, _) in after["verdicts"].items() if v == "fail" and before["verdicts"].get(i, ("",))[0] in ("success", "handoff_ok"))
        better = sorted(i for i, (v, _) in after["verdicts"].items() if v in ("success", "handoff_ok") and before["verdicts"].get(i, ("",))[0] == "fail")
        lines += ["", f"**Стало хуже:** {len(worse)}" + (f" ({', '.join(worse[:15])})" if worse else "") + f"; **стало лучше:** {len(better)}" + (f" ({', '.join(better[:15])})" if better else "")]
    return lines


# --- судья ---------------------------------------------------------------------------------------------------------

JUDGE_INSTRUCTIONS = """\
# Судья «успех внутри кнопки» (evalbench.buttons, v1)

Медицинский чат-бот клиники «Наука» (Самара). Пациент нажал в меню лист `button` (подпись `caption`), бот ответил на
нажатие (`press_answer`), затем пациент написал вопрос (`question`), бот ответил (`answer`). Оцени ответ на ВОПРОС.
Данные бота (замороженный мир, на них и надо проверять факты) — каталог `{world}`:
`apidata/samara_branches_snapshot.json` (филиалы: адрес, телефон, часы будни/сб/вс, флаги analysis/ecg/usi/doctorService),
`apidata/price_by_region/` (прайс), `apidata/service_info/` (подготовка, биоматериал, сроки, синонимы),
`apidata/doctors_*.jsonl` (врачи и специальности), `captured/promotions.json` (акции), `captured/kb_main_index.json`
(памятки). В мире НЕТ расписания (`schedule`) и поиска по базе знаний: честное «расписание сейчас не вижу — уточнит
оператор» там — правильное поведение.

Вердикт (одно):
- `success` — ответ решает вопрос пациента и верен по данным: нужный филиал / часы, памятка нужной услуги, верная
  инструкция по результатам, запись идёт к нужной специальности или услуге (спрашивает филиал / дату, показывает
  врачей), названы нужные врачи. Список вариантов при неоднозначности — успех. Ответ может быть длиннее нужного, но
  искомое в нём должно быть, и в нём не должно быть неверного.
- `handoff_ok` — бот честно переводит или предлагает оператора, потому что нужных данных у бота нет по дизайну
  (существующая запись, ДМС, детский приём, онлайн-консультация, срок готовности, медицинский совет, документы,
  то, чего нет в данных мира). Перевод при данных, которые есть в мире, — НЕ handoff_ok, а `fail` / «лишний перевод».
- `fail` — всё остальное; класс — одно из: «шаблон вместо ответа» (общий текст, вопрос не услышан: все филиалы
  на вопрос про один, общая инструкция результатов на конкретный вопрос), «чужая тема», «неверные данные» (не тот
  филиал / часы / памятка чужой услуги / не та специальность), «выдумка» (факт, которого нет в данных), «лишний перевод»,
  «нет перевода» (данных нет, а бот не перевёл и не предложил оператора — пациент без ответа), «не понял / лишнее
  уточнение» (переспросил понятное), «неполный ответ» (из двух вопросов ответил на один), «другое».

Критерий темы (`topic`):
- address — адреса и часы: вопрос про филиал / время / выходные / что делают в филиале — ответ по данным филиалов.
- booking_needed — анализы без записи, в порядке живой очереди; услуги врача / УЗИ — по записи.
- prepare — памятка нужной услуги (или список памяток на выбор при неоднозначности); общая памятка вместо имеющейся
  своей — «неверные данные»; памятки нет в данных — честно сказать / перевести (handoff_ok).
- results — как получить результат (данные для поиска, ссылка) или ответ на конкретный вопрос о результатах; общая
  инструкция на конкретный вопрос («пришлёте на почту?», «сразу выдаёте?») — «шаблон вместо ответа».
- booking — запись дошла до нужной специальности / услуги / врача; ДМС, дети, онлайн — оператор (handoff_ok).
- schedule — названы нужные врачи (специальность / ФИО); расписания в мире нет — честно сказать (не выдумывать дни).
- change — перенос: бот берёт врача и пожелание и передаёт оператору; отмена — сразу оператор (оба handoff_ok).
- assist — подбор анализов: уместный список услуг из прайса или честное «подберёт врач / оператор»; без диагнозов.
- promo — действующие акции из данных или ответ про конкретную акцию / скидку.
- docs — что нужно для справки для вычета и куда обратиться.
- price — (не-ценовой вопрос внутри ценового листа) по смыслу вопроса.

Верни строку JSONL на КАЖДЫЙ пакет входа: `{"id", "answer_sha", "verdict", "klass" (только для fail), "note" (≤15 слов,
без цитат пациента и без ПДн), "by": "<кто судил>"}` — `answer_sha` переписать из пакета без изменений.
"""


def judge_packets(runs: list[tuple[dict[str, Any], list[dict[str, Any]]]], price_labels: dict[str, dict[str, Any]], judged: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    """Ходы, которым нужен судья и вердикта ещё нет; одинаковые (id, ответ) разных прогонов — один пакет."""

    packets: dict[tuple[str, str], dict[str, Any]] = {}
    for header, turns in runs:
        cases = load_cases(header)
        for t in turns:
            case = cases.get(t["id"])
            if case is None:
                continue
            verdict, _ = turn_verdict(t, case, price_labels.get(t["id"]), judged)
            if verdict != "unjudged":
                continue
            key = (t["id"], answer_sha(t.get("answer") or ""))
            packets.setdefault(key, {
                "id": t["id"],
                "answer_sha": key[1],
                "button": case["button"],
                "caption": GATEWAY_CAPTIONS.get(case["button"]),
                "topic": LEAF_TOPIC.get(case["button"], "other"),
                "question": case["question"],
                "expected": case.get("expected"),
                "press_answer": (t.get("press") or {}).get("answer"),
                "answer": t.get("answer") or "",
                "outcome": t.get("outcome"),
                "handoff": t.get("handoff"),
            })
    return list(packets.values())


def judge_sample(runs: list[tuple[dict[str, Any], list[dict[str, Any]]]], judged: dict[tuple[str, str], dict[str, Any]], n: int, seed: int = 25) -> list[dict[str, Any]]:
    """Выборка вердиктов судьи владельцу: по темам поровну, провалы и переводы — первыми (спорные первыми, решение (б)6)."""

    by_topic: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen = set()
    for header, turns in runs:
        cases = load_cases(header)
        for t in turns:
            case = cases.get(t["id"])
            hit = judged.get((t["id"], answer_sha(t.get("answer") or ""))) if case else None
            if hit is None or (t["id"], hit["answer_sha"]) in seen:
                continue
            seen.add((t["id"], hit["answer_sha"]))
            by_topic[LEAF_TOPIC.get(case["button"], "other")].append({**hit, "button": case["button"], "question": case["question"], "answer": t.get("answer"), "press_answer": (t.get("press") or {}).get("answer")})
    rng = random.Random(seed)
    for rows in by_topic.values():
        rng.shuffle(rows)
        rows.sort(key=lambda r: {"fail": 0, "handoff_ok": 1, "success": 2}[r["verdict"]])
    picked: list[dict[str, Any]] = []
    while len(picked) < n and any(by_topic.values()):
        for topic in sorted(by_topic):
            if by_topic[topic] and len(picked) < n:
                picked.append(by_topic[topic].pop(0))
    return picked


def main(argv: list[str] | None = None) -> int:
    from .card import load_labels, load_run

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    exp = sub.add_parser("judge-export", help="пакеты ходов без вердикта судьи")
    exp.add_argument("runs", nargs="+")
    exp.add_argument("--out", type=Path, required=True, help="каталог пакетов (вне репо: в них реплики пациентов)")
    exp.add_argument("--per-file", type=int, default=60)
    smp = sub.add_parser("sample", help="выборка вердиктов на проверку владельцу (Markdown, с репликами — вне репо)")
    smp.add_argument("runs", nargs="+")
    smp.add_argument("--n", type=int, default=30)
    for p in (exp, smp):
        p.add_argument("--labels", type=Path, default=PRIVATE_ROOT / "labels" / "price_allowed.jsonl")
    args = ap.parse_args(argv)
    runs = [load_run(ref) for ref in args.runs]
    judged = load_judge()
    if args.cmd == "judge-export":
        packets = judge_packets(runs, load_labels(args.labels), judged)
        world = PRIVATE_ROOT / "worlds" / str(runs[0][0].get("world", {}).get("date"))
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "INSTRUCTIONS.md").write_text(JUDGE_INSTRUCTIONS.replace("{world}", str(world)), encoding="utf-8")
        for k in range(0, len(packets), args.per_file):
            with open(args.out / f"packets_{k // args.per_file:02d}.jsonl", "w", encoding="utf-8") as fh:
                for row in packets[k : k + args.per_file]:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"пакетов {len(packets)} → {args.out}; вердикты класть в {JUDGE_DIR}/<имя>.jsonl")
        return 0
    lines = ["# Выборка вердиктов судьи кнопок — на проверку владельцу", "", "Отметьте у каждого: согласен / не согласен (и как правильно).", ""]
    for k, row in enumerate(judge_sample(runs, judged, args.n), 1):
        lines += [
            f"## {k}. {row['id']} · {GATEWAY_CAPTIONS.get(row['button'], row['button'])} · **{row['verdict']}**" + (f" — {row.get('klass')}" if row.get("klass") else ""),
            "",
            f"- Вопрос: {row['question']}",
            f"- Ответ бота: {str(row.get('answer') or '').strip()[:700]}",
            f"- Судья: {row.get('note') or '—'}",
            "- Владелец: ☐ согласен ☐ не согласен —",
            "",
        ]
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
