"""П8: ОФЛАЙН-валидация embedding-слоя синонимов услуг (только отчёт, НЕ прод).

Слой 3 воронки распознавания услуг (карта синонимов → LLM-нормализация П5 →
embeddings): проверяем, решают ли эмбеддинги класс BUG-2026-06-02-09
(сахар=глюкоза, коагулограмма=гемостазиограмма, забор=взятие, ЭМГ и т.д.)
без ложных срабатываний на не-услугах (парковка, запись, ФИО).

Механика: индексируем уникальные serviceName реального прайса (region 3)
эмбеддером из agent_logic_1/embedding_filtration.py (rubert-tiny2, L2-норма,
косинус = dot) — численно в памяти, без Chroma. Прогоняем позитив-пары из
bug-log + негативы → recall@1/@5, разделимость порогом, FP-rate по порогам.

Запуск:  ./venv/bin/python scripts/validate_embedding_synonyms.py
Выход:   docs/embedding_synonym_validation_<date>.md + сводка в stdout.

РЕЗУЛЬТАТ = отчёт владельцу; включение в прод — отдельное решение.
"""

from __future__ import annotations

import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_logic_1 import embedding_filtration as ef  # noqa: E402
from agent_logic_2.nayka_api import api_price  # noqa: E402

# config.ini может задавать прод-путь cache_dir (/app_data/…), недоступный на
# dev-Mac. Для офлайн-валидации хватает дефолтного HF-кэша (~/.cache/huggingface),
# где rubert-tiny2 уже лежит.
ef.FILTRATION_CACHE = None

# (запрос пациента, [подстроки-матчеры целевого serviceName, lowercase])
# Пары — из bug-log: BUG-2026-06-23-01, BUG-2026-06-02-09, quirk «анти-тг».
POSITIVE_CASES: list[tuple[str, list[str]]] = [
    # --- ядро класса BUG-2026-06-02-09 (синонимы/аббревиатуры) ---
    ("забор крови из вены", ["взятие крови из вены"]),
    ("электромиография", ["эмг"]),
    ("сахар в крови", ["глюкоза"]),
    ("сахар", ["глюкоза"]),
    ("коагулограмма", ["гемостазиограмма"]),
    ("са-125", ["ca - 125"]),
    ("са 125", ["ca - 125"]),
    ("анти-тг", ["ат - тг", "тиреоглобулин"]),
    ("антитела к тиреоглобулину", ["ат - тг", "тиреоглобулин"]),
    # --- sanity: точные термины каталога обязаны находиться ---
    ("взятие крови из вены", ["взятие крови из вены"]),
    ("гемостазиограмма", ["гемостазиограмма"]),
    ("глюкоза", ["глюкоза"]),
    ("общий анализ крови", ["общий анализ крови"]),
    ("глюкозотолерантный тест", ["глюкозотолерантный"]),
]

# Exploratory: прямой цели в каталоге НЕТ (панельные названия) — в recall не
# считаем, но топ-3 показываем владельцу (что embeddings предложили бы).
EXPLORATORY_CASES: list[str] = [
    "печёночные пробы",
    "почечные пробы",
]

# Негативы: НЕ услуги — матчить не должны (иначе слой 3 будет продавать
# «Свеклу сахарную» на «парковку»). ФИО/запись/small-talk/адрес.
NEGATIVE_CASES: list[str] = [
    "парковка",
    "где припарковаться у клиники",
    "запишите к врачу",
    "запишите меня к врачу на завтра",
    "Иванов Иван Иванович",
    "добрый день",
    "сайт не работает",
    "как доехать до филиала на Победы 83",
]

THRESHOLDS = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
TOP_K = 5


def _load_catalog_names() -> list[str]:
    rows = [r for r in api_price.load_price_by_region(3) if isinstance(r, dict)]
    names = sorted({
        str(r.get("serviceName") or "").strip()
        for r in rows
        if str(r.get("serviceName") or "").strip()
    })
    if len(names) < 3000:
        raise SystemExit(f"Каталог подозрительно мал: {len(names)} имён (ожидалось ~3.5к)")
    return names


def _top_k(query_emb, name_embs, names: list[str], k: int) -> list[tuple[str, float]]:
    sims = (query_emb @ name_embs.T).squeeze(0)
    top = sims.topk(min(k, len(names)))
    return [(names[i], float(s)) for s, i in zip(top.values.tolist(), top.indices.tolist())]


def _matches(name: str, matchers: list[str]) -> bool:
    low = name.lower()
    return any(m in low for m in matchers)


def main() -> None:
    t0 = time.monotonic()
    names = _load_catalog_names()
    print(f"Каталог: {len(names)} уникальных serviceName (region 3)")

    t_idx = time.monotonic()
    name_embs = ef._embed_texts(names)
    idx_s = time.monotonic() - t_idx
    print(f"Индексация: {idx_s:.1f}s, dim={name_embs.shape[1]}, device={ef._DEVICE}")

    pos_rows = []
    hits_at_1 = hits_at_5 = 0
    pos_top1_sims: list[float] = []
    for query, matchers in POSITIVE_CASES:
        q_emb = ef._embed_texts([query])
        top = _top_k(q_emb, name_embs, names, TOP_K)
        rank = next((i + 1 for i, (n, _) in enumerate(top) if _matches(n, matchers)), None)
        hits_at_1 += 1 if rank == 1 else 0
        hits_at_5 += 1 if rank is not None else 0
        pos_top1_sims.append(top[0][1])
        pos_rows.append((query, rank, top))

    neg_rows = []
    neg_top1_sims: list[float] = []
    for query in NEGATIVE_CASES:
        q_emb = ef._embed_texts([query])
        top = _top_k(q_emb, name_embs, names, 3)
        neg_top1_sims.append(top[0][1])
        neg_rows.append((query, top))

    exp_rows = []
    for query in EXPLORATORY_CASES:
        q_emb = ef._embed_texts([query])
        exp_rows.append((query, _top_k(q_emb, name_embs, names, 3)))

    n_pos = len(POSITIVE_CASES)
    recall1 = hits_at_1 / n_pos
    recall5 = hits_at_5 / n_pos

    # Разделимость: сколько позитивов теряем / негативов пропускаем на пороге.
    thr_rows = []
    for thr in THRESHOLDS:
        pos_kept = sum(1 for s in pos_top1_sims if s >= thr)
        neg_fp = sum(1 for s in neg_top1_sims if s >= thr)
        thr_rows.append((thr, pos_kept, n_pos, neg_fp, len(NEGATIVE_CASES)))

    total_s = time.monotonic() - t0
    today = date.today().isoformat()
    report_path = Path(__file__).resolve().parent.parent / "docs" / f"embedding_synonym_validation_{today}.md"

    def fmt_top(top: list[tuple[str, float]]) -> str:
        return "<br>".join(f"{s:.3f} · {n[:70]}" for n, s in top)

    lines = [
        f"# П8: офлайн-валидация embedding-слоя синонимов ({today})",
        "",
        "> Слой 3 воронки распознавания услуг (карта → LLM-П5 → embeddings).",
        "> ТОЛЬКО отчёт: в прод НЕ включено, решение за владельцем.",
        "",
        f"- Модель: `{ef.FILTRATION_MODEL}` (device={ef._DEVICE}, dim={name_embs.shape[1]})",
        f"- Индекс: {len(names)} уникальных serviceName прайса region 3, в памяти (без Chroma)",
        f"- Индексация: {idx_s:.1f}s; полный прогон: {total_s:.1f}s",
        "",
        "## Итог",
        "",
        f"- **recall@1 = {hits_at_1}/{n_pos} ({recall1:.0%})**, **recall@5 = {hits_at_5}/{n_pos} ({recall5:.0%})** на позитив-парах из bug-log",
        f"- top-1 sim: позитивы min={min(pos_top1_sims):.3f} / медиана={sorted(pos_top1_sims)[n_pos // 2]:.3f}; негативы max={max(neg_top1_sims):.3f}",
        "",
        "## Позитив-пары (синонимы из bug-log)",
        "",
        "| Запрос пациента | Ранг цели | Топ-5 (sim · serviceName) |",
        "|---|---|---|",
    ]
    for query, rank, top in pos_rows:
        rank_s = str(rank) if rank is not None else "— (мимо топ-5)"
        lines.append(f"| {query} | {rank_s} | {fmt_top(top)} |")

    lines += [
        "",
        "## Негативы (не-услуги: парковка/запись/ФИО)",
        "",
        "| Запрос | Топ-3 (sim · serviceName) |",
        "|---|---|",
    ]
    for query, top in neg_rows:
        lines.append(f"| {query} | {fmt_top(top)} |")

    lines += [
        "",
        "## Разделимость порогом (top-1 sim ≥ порога)",
        "",
        "| Порог | Позитивов прошло | Негативов прошло (FP) |",
        "|---|---|---|",
    ]
    for thr, pos_kept, pos_n, neg_fp, neg_n in thr_rows:
        lines.append(f"| {thr:.2f} | {pos_kept}/{pos_n} | {neg_fp}/{neg_n} |")

    lines += [
        "",
        "## Exploratory (без прямой цели в каталоге — что предложил бы слой)",
        "",
        "| Запрос | Топ-3 (sim · serviceName) |",
        "|---|---|",
    ]
    for query, top in exp_rows:
        lines.append(f"| {query} | {fmt_top(top)} |")

    lines += [
        "",
        "## Чтение результатов",
        "",
        "- Если позитивы стабильно в топ-5, а негативы отсекаются порогом без потери",
        "  позитивов — слой пригоден как кандидат-генератор ПЕРЕД confirm-флоу",
        "  (как П5: кандидат → «Вы имели в виду …?», без прямой выдачи).",
        "- Если разделимости нет (негативы выше порога позитивов) — в текущем виде",
        "  слой в прод не готов: нужен другой эмбеддер или гейт по типу запроса.",
        "",
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nrecall@1={recall1:.0%} recall@5={recall5:.0%}; отчёт: {report_path}")


if __name__ == "__main__":
    main()
