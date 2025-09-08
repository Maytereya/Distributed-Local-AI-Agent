#
# Компоненты UI
#
import time
import uuid
from typing import Optional, Any, List, Tuple

import pandas as pd


def generate_new_id():
    return str(uuid.uuid4())


TABLE_HEADERS = ["Колонка 1", "Колонка 2"]  # держим в одном месте


def _now_utc_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _split_paragraphs(text: str):
    # простой и предсказуемый сплит: по пустой строке
    paras = [p.strip() for p in (text or "").split("\n\n") if p.strip()]
    return paras if paras else []


def _normalize_table(table_value: Any, headers: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
    headers = headers or ["Column 1", "Column 2"]
    width = len(headers)

    rows = table_value or []
    # аккуратно выравниваем строки по ширине
    norm_rows = [(list(r) + [""] * width)[:width] for r in rows]

    df = pd.DataFrame(norm_rows, columns=headers)
    # None -> "", str(), strip()
    df = df.applymap(lambda x: ("" if x is None else str(x)).strip())
    # удаляем только полностью пустые строки
    mask_nonempty = df.apply(lambda r: any(bool(cell) for cell in r), axis=1)
    df = df[mask_nonempty]

    if df.empty:
        return None
    return df


def build_blocks(
        doc_id: str,
        title: str,
        content: str,
        keywords_csv: str,
        table_data: Any,
        table_headers: Optional[List[str]] = None  # например: ["Колонка 1", "Колонка 2"]
) -> Tuple[str, List[dict]]:
    """
    Возвращает (doc_id, blocks) для индексации.
    Таблица добавляется отдельным блоком с полями content/html/csv.
    """
    if not doc_id or not doc_id.strip():
        doc_id = generate_new_id()
    if not title or not title.strip():
        raise ValueError("Не введён заголовок")
    if not content or not content.strip():
        raise ValueError("Поле 'Основной текст' пусто")

    created = _now_utc_iso()
    keywords = [kw.strip() for kw in (keywords_csv or "").split(",") if kw and kw.strip()]

    blocks: List[dict] = []
    block_id = 0

    # текст → параграфы (каждый абзац — отдельный блок)
    for para in _split_paragraphs(content):
        block_id += 1
        blocks.append({
            "id": f"{doc_id}_p1_b{block_id}",
            "doc_id": doc_id,
            "page": 1,
            "block_id": block_id,
            "type": "text",
            "title": title if block_id == 1 else None,  # заголовок только в первом блоке
            "content": para,  # <-- ключ 'content'
            "html": None,
            "csv": None,
            "keywords": keywords,
            "created_at": created
        })

    # таблица (если есть данные)
    if table_data is not None:
        df = _normalize_table(table_data, headers=TABLE_HEADERS)
        if df is not None:
            df2 = df.fillna("").astype(str)
            html = df2.to_html(index=False)
            csv_ = df2.to_csv(index=False)
            content_flat = "\n".join([",".join(row) for row in df2.values.tolist()])

            # красивый заголовок (вариант A)
            first_preview = " | ".join(df2.values.tolist()[0]) if len(df2) else ""
            table_title = f"Таблица: {first_preview[:60]}" if first_preview else f"Таблица ({len(df2)}×{len(df2.columns)})"

            block_id += 1
            blocks.append({
                "id": f"{doc_id}_p1_b{block_id}",
                "doc_id": doc_id,
                "page": 1,
                "block_id": block_id,
                "type": "table",
                "title": table_title,  # ← теперь не «Без заголовка»
                "content": content_flat,
                "html": html,
                "csv": csv_,
                "keywords": keywords,
                "created_at": created
            })

    return doc_id, blocks
