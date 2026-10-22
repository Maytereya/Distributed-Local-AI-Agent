"""Карточка замера «было / стало» по метрикам решения 08.10 (мастер-хэндофф §7.1).

    ./venv/bin/python -m evalbench.card <прогон_было> [<прогон_стало>] [--labels prices.jsonl]

Главные на эталоне: чужая цена (по разметке допустимых serviceId), срочное — полнота 100%
на синтетике, цена без источника. Ограничители: переводы и уточнения — раздельно
(день и ночь — разными прогонами), «нет данных» без перевода (класс 4), задержка p50/p95,
сбои LLM. Правило решения ≤3: одна предотвращённая чужая цена стоит не больше трёх
лишних уточнений или переводов.

Карточка печатается в Markdown; в ней только счётчики — без реплик пациентов, её можно
вставлять в журнал и хэндофф.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .detectors import ClinicSynonyms, PoolRecall, pool_recall, service_text
from .prices import PriceCatalog, PricePair, multi_coverage, price_verdict
from .specialties import SpecialtyMatcher, specialty_verdict
from .world import PRIVATE_ROOT

OUTCOME_ORDER = (
    "price", "memo", "answer", "clarify", "not_understood", "operator_offer",
    "handoff", "handoff_tech", "urgent", "no_info", "tech_error",
)


def load_run(ref: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = Path(ref)
    if not path.exists():
        path = PRIVATE_ROOT / "runs" / ref
    header = json.loads((path / "run.json").read_text(encoding="utf-8"))
    turns = [json.loads(line) for line in (path / "turns.jsonl").open(encoding="utf-8") if line.strip()]
    return header, turns


def load_labels(path: Path | None) -> dict[str, dict[str, Any]]:
    """Разметка ценовых вопросов: строка `{"id", "kind", "core", "ok", "ambiguous", ...}`."""

    if path is None or not path.exists():
        return {}
    out = {}
    for line in path.open(encoding="utf-8"):
        if line.strip():
            row = json.loads(line)
            out[row["id"]] = row
    return out


def _pct(values: list[int], q: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def refresh_expectations(header: dict[str, Any], turns: list[dict[str, Any]]) -> None:
    """Ожидания (срочно / нет) — из текущих файлов наборов: решение по эталону применяется и к
    старым прогонам без перепрогона (09.10: u031 перешёл в двойники)."""

    expected: dict[tuple[str, str], Any] = {}
    for ref in header.get("sets") or []:
        path = Path(ref).expanduser()
        if not path.is_absolute():
            path = Path(__file__).resolve().parent.parent / path
        if path.is_file():
            for line in path.open(encoding="utf-8"):
                if line.strip():
                    row = json.loads(line)
                    expected[(path.stem, row["id"])] = row.get("expect_urgent")
    for t in turns:
        key = (t.get("set"), t["id"])
        if key in expected:
            t["expect_urgent"] = expected[key]


# Роли МИС, к которым с жалобой не направляют: назвать их пациенту — ошибка при любой разметке.
NOT_FOR_PATIENTS = {
    "узи", "ультразвуковой диагностики", "функциональной диагностики", "медицинская сестра по физиотерапии",
    "специалист по физиотерапии", "анестезиолог-реаниматолог", "эндоскопист", "просвиров -прием",
}


# Шаблон «Не совсем понял…» перечисляет примеры вопросов («Запиши к кардиологу», «УЗИ брюшной
# полости») — это не рекомендация врача; вырезаем до разбора специальностей.
_CANNED_EXAMPLES_RE = re.compile(r"Не совсем понял ваш запрос\..*?(?:\n\n|$)", re.S)


def load_symptom_labels(header: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    """Разметка «симптом → врач»: синтетика — прямо в файле набора, трафик — в приватных метках."""

    out: dict[tuple[str, str], dict[str, Any]] = {}
    for ref in header.get("sets") or []:
        path = Path(ref).expanduser()
        if not path.is_absolute():
            path = Path(__file__).resolve().parent.parent / path
        if not path.is_file():
            continue
        for line in path.open(encoding="utf-8"):
            if line.strip():
                row = json.loads(line)
                if "core" in row:
                    out[(path.stem, row["id"])] = row
                elif path.stem.startswith("golden"):
                    out.setdefault((path.stem, row["id"]), None)  # кандидат: метка — в приватном файле
    private = PRIVATE_ROOT / "labels" / "symptom_allowed.jsonl"
    if private.is_file():
        by_id = {json.loads(line)["id"]: json.loads(line) for line in private.open(encoding="utf-8") if line.strip()}
        for key in list(out):
            if out[key] is None and key[1] in by_id:
                out[key] = by_id[key[1]]
    return {k: v for k, v in out.items() if v}


def score_specialties(header: dict[str, Any], turns: list[dict[str, Any]]) -> None:
    labels = load_symptom_labels(header)
    vocab_file = PRIVATE_ROOT / "labels" / f"specialties_{header.get('world', {}).get('date')}.json"
    if not labels or not vocab_file.is_file():
        return
    roles = json.loads(vocab_file.read_text(encoding="utf-8"))
    matcher = SpecialtyMatcher(roles)
    for t in turns:
        label = labels.get((t.get("set"), t["id"]))
        if label is None:
            continue
        named = matcher.named(_CANNED_EXAMPLES_RE.sub(" ", t.get("answer") or ""))
        t["specialties_named"] = named
        t["specialty_verdict"] = specialty_verdict(named, label.get("core") or [], label.get("ok") or [], roles)
        t["specialty_not_for_patients"] = sorted(set(named) & NOT_FOR_PATIENTS)


def rescore_prices(header: dict[str, Any], turns: list[dict[str, Any]]) -> None:
    """Пары «услуга — цена» заново по сохранённому ответу: правка детектора не требует перепрогона."""

    apidata = PRIVATE_ROOT / "worlds" / str(header.get("world", {}).get("date")) / "apidata"
    if not apidata.is_dir():
        return
    catalog = PriceCatalog.from_world(apidata)
    for t in turns:
        pairs = catalog.pairs(t.get("answer") or "")
        t["price_pairs"] = [p.__dict__ for p in pairs]
        t["unsourced_prices"] = sum(1 for p in pairs if p.status == "unsourced")


def metrics(turns: list[dict[str, Any]], labels: dict[str, dict[str, Any]], synonyms: ClinicSynonyms | None = None) -> dict[str, Any]:
    n = len(turns)
    outcomes = Counter(t["outcome"] for t in turns)
    verdicts = Counter()
    coverage: list[tuple[int, int, int, int]] = []
    for t in turns:
        if t["id"] in labels:
            pairs = [PricePair(**p) for p in t.get("price_pairs") or []]
            verdict = price_verdict(pairs, labels[t["id"]])
            if verdict is not None:
                verdicts[verdict] += 1
            covered = multi_coverage(pairs, labels[t["id"]])
            if covered is not None:
                coverage.append(covered)
    urgent_cases = [t for t in turns if t.get("expect_urgent") is True]
    twins = [t for t in turns if t.get("expect_urgent") is False]
    spec = Counter(t["specialty_verdict"] for t in turns if t.get("specialty_verdict"))
    # Решения модуля выбора (ход (в)) — из телеметрии хода; SLO: UNAVAILABLE <2%, p95 выбора <4 с.
    # Решения тени (`shadow: true`, L-15b шаг 4) на ответ не влияли — в решения хода не входят.
    choices = [c for t in turns for c in (t.get("telemetry") or {}).get("choose") or [] if not c.get("shadow")]
    choose_status = Counter(c["status"] for c in choices)
    llm = Counter(c["outcome"] for t in turns for c in t.get("llm_calls") or [])
    sites = Counter(c["site"] for t in turns for c in t.get("llm_calls") or [])
    ms = [t["ms"] for t in turns]
    # Служебный текст — по сохранённому ответу (как цены): правка детектора не требует перепрогона.
    leaked = [service_text(t.get("answer") or "") for t in turns]
    # Recall@pool — по пулу, который записал сам ход (`price_pool`), срез «в словаре» — по данным мира.
    pool_q = pool_by_question(turns, labels, synonyms)
    return {
        "n": n,
        "outcomes": dict(outcomes),
        "handoff": sum(1 for t in turns if t["handoff"] and not t["urgent"]),
        "handoff_tech": outcomes.get("handoff_tech", 0),
        "operator_offer": outcomes.get("operator_offer", 0),
        "clarify": outcomes.get("clarify", 0) + outcomes.get("not_understood", 0),
        "no_info": outcomes.get("no_info", 0),
        "unsourced_turns": sum(1 for t in turns if t.get("unsourced_prices")),
        "unattributed_turns": sum(
            1 for t in turns if any(p["status"] == "unattributed" for p in t.get("price_pairs") or [])
        ),
        "leak": sum(1 for t in turns if t.get("leak")),
        "service_text": sum(1 for kinds in leaked if kinds),
        "service_text_kinds": dict(Counter(kind for kinds in leaked for kind in kinds)),
        "long_1500": sum(1 for t in turns if t.get("chars", 0) > 1500),
        "price_labeled": sum(verdicts.values()),
        "price_verdicts": dict(verdicts),
        "foreign": verdicts.get("foreign", 0) + verdicts.get("list_miss", 0),
        **{f"price_{k}": verdicts.get(k, 0) for k in ("correct", "list_hit", "single_for_ambiguous", "no_price")},
        "multi_n": len(coverage),
        "multi_two_plus": sum(1 for named, _, _, _ in coverage if named >= 2),
        "multi_core_pct": round(100 * sum(hit / total for _, hit, total, _ in coverage if total) / len(coverage)) if coverage else None,
        "multi_named": sum(named for named, _, _, _ in coverage),
        "multi_outside": sum(outside for _, _, _, outside in coverage),
        "urgent_recall": (sum(1 for t in urgent_cases if t["urgent"]), len(urgent_cases)),
        "urgent_false": (sum(1 for t in twins if t["urgent"]), len(twins)),
        "symptom_n": sum(spec.values()),
        **{f"symptom_{k}": spec.get(k, 0) for k in ("right", "acceptable", "partial", "wrong", "none")},
        "symptom_not_for_patients": sum(1 for t in turns if t.get("specialty_not_for_patients")),
        "choose_n": len(choices),
        "choose_unavailable": choose_status.get("unavailable", 0),
        "choose_invalid": choose_status.get("invalid", 0),
        "choose_p95_ms": _pct([c["ms"] for c in choices if not c.get("cached")], 0.95),
        "p50_ms": _pct(ms, 0.5),
        "p95_ms": _pct(ms, 0.95),
        "llm_calls": sum(llm.values()),
        "llm_outcomes": dict(llm),
        "llm_sites": dict(sites),
        "pool": pool_metrics(pool_q),
        "pool_q": pool_q,
    }


def _fmt(value: Any) -> str:
    if isinstance(value, tuple):
        hit, total = value
        return f"{hit}/{total}" if total else "—"
    return "—" if value is None else str(value)


def answer_diff(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> tuple[int, int, list[str]]:
    """Сколько общих вопросов получили разный текст ответа: (отличаются, общих, первые id).

    Для A/A и доказательства переноса без смены поведения (ход (в): `legacy` против `off` на
    записанных ответах должно быть 0).
    """

    by_key = {(t.get("set"), t["id"]): str(t.get("answer") or "").strip() for t in left}
    shared = [(t.get("set"), t["id"], str(t.get("answer") or "").strip()) for t in right if (t.get("set"), t["id"]) in by_key]
    differ = [qid for qset, qid, text in shared if by_key[(qset, qid)] != text]
    return len(differ), len(shared), differ[:10]


# Срезы Recall@pool (замер 22.10, L-31): слово пациента в словаре клиники или нет — для одной услуги; список — отдельно.
POOL_SLICES = (("dict", "слово в словаре клиники"), ("nodict", "вне словаря"), ("list", "список услуг"), ("all", "всего"))


def load_synonyms(header: dict[str, Any]) -> ClinicSynonyms | None:
    apidata = PRIVATE_ROOT / "worlds" / str(header.get("world", {}).get("date")) / "apidata"
    return ClinicSynonyms.from_world(apidata) if apidata.is_dir() else None


def pool_slice(turn: dict[str, Any], kind: str, synonyms: ClinicSynonyms | None) -> str:
    if kind == "multi":
        return "list"
    return "dict" if synonyms is not None and synonyms.named(turn.get("question") or "") else "nodict"


def pool_by_question(turns: list[dict[str, Any]], labels: dict[str, dict[str, Any]], synonyms: ClinicSynonyms | None) -> dict[str, tuple[str, PoolRecall]]:
    """id вопроса → (срез, Recall@pool хода) — по размеченным ценовым вопросам, где эталон есть в прайсе."""

    out: dict[str, tuple[str, PoolRecall]] = {}
    for t in turns:
        recall = pool_recall(t.get("telemetry"), labels.get(t["id"]))
        if recall is not None:
            out[t["id"]] = (pool_slice(t, recall.kind, synonyms), recall)
    return out


def pool_metrics(by_question: dict[str, tuple[str, PoolRecall]]) -> dict[str, dict[str, Any]]:
    """По срезам: размечено, с записанным пулом, Recall@pool первого круга и обоих кругов (%, среднее по вопросам с
    пулом), размер пула первого круга (среднее и p90 по вызовам выбора), вызовов второго круга."""

    acc: dict[str, dict[str, Any]] = {key: {"n": 0, "recorded": 0, "first": 0.0, "both": 0.0, "sizes": [], "second": 0} for key, _ in POOL_SLICES}
    for slice_key, recall in by_question.values():
        for key in (slice_key, "all"):
            s = acc[key]
            s["n"] += 1
            if recall.recorded:
                s["recorded"] += 1
                s["first"] += recall.first or 0.0
                s["both"] += recall.both or 0.0
                s["sizes"].extend(recall.sizes_first)
                s["second"] += len(recall.sizes_second)
    out = {}
    for key, s in acc.items():
        k = s["recorded"]
        out[key] = {
            "n": s["n"],
            "recorded": k,
            "first_pct": round(100 * s["first"] / k) if k else None,
            "both_pct": round(100 * s["both"] / k) if k else None,
            "size_mean": round(sum(s["sizes"]) / len(s["sizes"]), 1) if s["sizes"] else None,
            "size_p90": _pct(s["sizes"], 0.9),
            "second_calls": s["second"],
        }
    return out


def pool_changes(before: dict[str, tuple[str, PoolRecall]], after: dict[str, tuple[str, PoolRecall]]) -> dict[str, list[str]]:
    """Поимённо: какие вопросы выпали из пула и какие вошли (по первому кругу и по обоим), у каких пул появился или
    пропал. Список — по доле услуг: меньше — «выпал», больше — «вошёл». Только id, без реплик."""

    out: dict[str, list[str]] = {k: [] for k in ("first_lost", "first_gained", "both_lost", "both_gained", "pool_gone", "pool_new")}
    for qid in sorted(set(before) & set(after)):
        old, new = before[qid][1], after[qid][1]
        if old.recorded != new.recorded:
            out["pool_gone" if old.recorded else "pool_new"].append(qid)
            continue
        if not old.recorded:
            continue
        for stage in ("first", "both"):
            was, now = getattr(old, stage), getattr(new, stage)
            if now < was:
                out[f"{stage}_lost"].append(qid)
            elif now > was:
                out[f"{stage}_gained"].append(qid)
    return out


def _pool_cell(s: dict[str, Any]) -> str:
    if not s["n"]:
        return "—"
    if not s["recorded"]:
        return f"пул не записан (0/{s['n']})"
    return (
        f"{s['first_pct']}% / {s['both_pct']}% ({s['recorded']}/{s['n']}); "
        f"пул {_fmt(s['size_mean'])} / p90 {_fmt(s['size_p90'])}; 2-й круг {s['second_calls']}"
    )


def render_pool(runs: list[tuple[dict[str, Any], dict[str, Any]]], changes: dict[str, list[str]] | None = None) -> list[str]:
    """Раздел карточки «Recall@pool (цена)»: строки — срезы, столбцы — прогоны; при двух прогонах — поимённые
    переходы вопросов."""

    lines = [
        "",
        "**Recall@pool (цена)** — есть ли строка эталона (core ∪ ok; у списка — доля услуг core) среди кандидатов, ушедших "
        "в выбор строки LLM (`price_pool` хода; у корзины — варианты LLM корзины и строки сильных позиций по данным). "
        "Ячейка: 1-й круг / с 2-м кругом (вопросов с пулом / размечено); размер пула 1-го круга — среднее / p90 по вызовам "
        "LLM; вызовов 2-го круга.",
        "",
        "| Срез | " + " | ".join(h["name"] for h, _ in runs) + " |",
        "|---|" + "---|" * len(runs),
    ]
    for key, title in POOL_SLICES:
        lines.append(f"| {title} | " + " | ".join(_pool_cell(m["pool"][key]) for _, m in runs) + " |")
    if changes is not None:
        titles = (
            ("first_lost", "выпали из пула 1-го круга"),
            ("first_gained", "вошли в пул 1-го круга"),
            ("both_lost", "выпали из пула (оба круга)"),
            ("both_gained", "вошли в пул (оба круга)"),
            ("pool_gone", "пул пропал (LLM строку не выбирала)"),
            ("pool_new", "пул появился"),
        )
        lines.append("")
        for key, title in titles:
            ids = changes[key]
            lines.append(f"- {title}: {len(ids)}" + (f" — {', '.join(ids)}" if ids else ""))
    return lines


def render(
    runs: list[tuple[dict[str, Any], dict[str, Any]]],
    diff: tuple[int, int, list[str]] | None = None,
    changes: dict[str, list[str]] | None = None,
) -> str:
    lines = ["# Карточка замера", ""]
    for header, _m in runs:
        world = header.get("world", {})
        lines.append(
            f"- **{header['name']}**: код `{header['code_sha']}`, LLM `{header['mode']}` ({header.get('model')}), "
            f"мир {world.get('date')} {world.get('profile')}, вопросов {header['done']}/{header['cases']}, "
            f"синонимов {header.get('synonyms')}, valid={header.get('valid')}"
            + (f", **остановлен: {header['stopped']}**" if header.get("stopped") else "")
            + (
                ", настройки " + ", ".join(f"{k}={v.get('value')}" + ("" if v.get("known") else " (нет в коде)") for k, v in header["cfg"].items())
                if header.get("cfg")
                else ""
            )
        )
        if not header.get("valid"):
            lines.append(
                f"  - недействителен: сеть {header.get('net_attempts')}, очистки {header.get('cleanup_calls')}, "
                f"мир {header.get('world_problems')}, неопознанные промпты {header.get('llm_outcomes', {}).get('unknown_site', 0)}, "
                f"неоднозначные {header.get('ambiguous_sites')}"
            )
        lines.append(f"  - недоступно в мире: {', '.join(world.get('unavailable', []))}")
    lines.append("")
    names = [h["name"] for h, _ in runs]
    lines.append("| Метрика | " + " | ".join(names) + (" | Δ |" if len(runs) == 2 else " |"))
    lines.append("|---|" + "---|" * len(runs) + ("---|" if len(runs) == 2 else ""))
    rows = [
        ("Чужая цена (разметка)", "foreign"),
        ("Размечено ценовых", "price_labeled"),
        ("  верная одна цена", "price_correct"),
        ("  список с нужной", "price_list_hit"),
        ("  одна цена вместо списка", "price_single_for_ambiguous"),
        ("  без цены", "price_no_price"),
        ("Вопрос о нескольких услугах (multi)", "multi_n"),
        ("  названо ≥2 услуг списка", "multi_two_plus"),
        ("  покрытие core, % (среднее)", "multi_core_pct"),
        ("  допустимых услуг названо (сумма)", "multi_named"),
        ("  услуг вне разметки показано (сумма)", "multi_outside"),
        ("Срочное: полнота", "urgent_recall"),
        ("Срочное: ложное на двойниках", "urgent_false"),
        ("Симптом → врач: размечено", "symptom_n"),
        ("  верно (первая очередь)", "symptom_right"),
        ("  допустимо", "symptom_acceptable"),
        ("  частично (есть лишние)", "symptom_partial"),
        ("  неверно", "symptom_wrong"),
        ("  врач не назван", "symptom_none"),
        ("  назвал роль не для пациента", "symptom_not_for_patients"),
        ("Выбор (choose): решений", "choose_n"),
        ("  LLM недоступна (UNAVAILABLE)", "choose_unavailable"),
        ("  ответ не разобран (INVALID)", "choose_invalid"),
        ("  p95 выбора, мс", "choose_p95_ms"),
        ("Цена без источника (ходов)", "unsourced_turns"),
        ("Цена без привязки к услуге (ходов)", "unattributed_turns"),
        ("Перевод на оператора", "handoff"),
        ("  из них технический", "handoff_tech"),
        ("Оффер оператора", "operator_offer"),
        ("Уточнение / не понял", "clarify"),
        ("Нет данных без перевода (класс 4)", "no_info"),
        ("Служебный текст в ответе", "service_text"),
        ("Утечка внутренностей", "leak"),
        ("Длиннее 1500 знаков", "long_1500"),
        ("p50, мс", "p50_ms"),
        ("p95, мс", "p95_ms"),
        ("Вызовов LLM", "llm_calls"),
    ]
    for title, key in rows:
        vals = [m[key] for _, m in runs]
        delta = ""
        if len(runs) == 2 and all(isinstance(v, int) for v in vals):
            delta = f" {vals[1] - vals[0]:+d} |"
        elif len(runs) == 2:
            delta = " |"
        lines.append(f"| {title} | " + " | ".join(_fmt(v) for v in vals) + (" |" + delta if delta else " |"))
    if any(m["service_text"] for _, m in runs):
        lines.append("")
        lines.append(
            "**Служебный текст по видам** (ходов): "
            + "; ".join(f"{h['name']}: " + (", ".join(f"{k} {v}" for k, v in sorted(m["service_text_kinds"].items())) or "—") for h, m in runs)
        )
    if diff is not None:
        lines.append("")
        lines.append(f"**Ответ отличается:** {diff[0]} из {diff[1]} общих вопросов" + (f" (например: {', '.join(diff[2])})" if diff[2] else ""))
    lines.extend(render_pool(runs, changes))
    lines.append("")
    lines.append("**Исходы ходов:**")
    lines.append("")
    lines.append("| Исход | " + " | ".join(names) + " |")
    lines.append("|---|" + "---|" * len(runs))
    for key in OUTCOME_ORDER:
        lines.append(f"| {key} | " + " | ".join(str(m["outcomes"].get(key, 0)) for _, m in runs) + " |")
    comparable = len(runs) == 2 and all(
        runs[0][0].get(k) == runs[1][0].get(k) for k in ("mode", "llm_mode", "cases")
    ) and runs[0][0].get("world", {}).get("date") == runs[1][0].get("world", {}).get("date") and runs[0][0].get(
        "world", {}
    ).get("profile") == runs[1][0].get("world", {}).get("profile")
    if len(runs) == 2 and not comparable:
        lines.append("")
        lines.append("**Правило ≤3 не применимо:** прогоны в разных режимах LLM, мирах или наборах — это срез, а не «было / стало».")
    if comparable:
        _, before = runs[0]
        _, after = runs[1]
        prevented = before["foreign"] - after["foreign"]
        extra = (after["handoff"] + after["operator_offer"] + after["clarify"]) - (
            before["handoff"] + before["operator_offer"] + before["clarify"]
        )
        lines.append("")
        lines.append(
            f"**Правило ≤3:** предотвращено чужих цен {prevented}, лишних уточнений и переводов {extra} → "
            + ("принять" if extra <= 3 * max(prevented, 0) else "НЕ проходит")
            + (" (чужие цены не размечены — правило не применимо)" if not before["price_labeled"] else "")
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="имя прогона или путь к каталогу (1 или 2)")
    ap.add_argument("--labels", type=Path, default=PRIVATE_ROOT / "labels" / "price_allowed.jsonl")
    ap.add_argument("--common", action="store_true", help="сравнивать только вопросы, которые есть в обоих прогонах")
    args = ap.parse_args(argv)
    labels = load_labels(args.labels)
    runs = [load_run(ref) for ref in args.runs[:2]]
    if args.common and len(runs) == 2:
        keys = [{(t.get("set"), t["id"]) for t in turns} for _, turns in runs]
        shared = keys[0] & keys[1]
        runs = [(dict(h, cases=len(shared), done=len(shared)), [t for t in turns if (t.get("set"), t["id"]) in shared]) for h, turns in runs]
    diff = answer_diff(runs[0][1], runs[1][1]) if len(runs) == 2 else None
    loaded = []
    for header, turns in runs:
        refresh_expectations(header, turns)
        rescore_prices(header, turns)
        score_specialties(header, turns)
        loaded.append((header, metrics(turns, labels, load_synonyms(header))))
    changes = pool_changes(loaded[0][1]["pool_q"], loaded[1][1]["pool_q"]) if len(loaded) == 2 else None
    print(render(loaded, diff, changes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
