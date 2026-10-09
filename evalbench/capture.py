"""Разовая запись живых источников в мир стенда: `/regions`, акции, памятки базы знаний.

    ./venv/bin/python -m evalbench.capture --world 20261008

Только чтение: GET к МИС и выборка документов `main_index` из Meilisearch. Индексы не
создаются и не настраиваются (`ensure_default_indexes_once` пишет в индекс — здесь он не
вызывается). Нужен VPN. Запись ложится в `<мир>/captured/` и в манифест мира не входит:
это отдельный слой, его наличие печатается в шапке прогона.

Расписание не записывается: оно живое и считается от «сегодня» — в мире стенда его нет.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .world import PRIVATE_ROOT

REPO_ROOT = Path(__file__).resolve().parent.parent


def _save(folder: Path, name: str, data) -> str:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name}.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return f"{name}: {len(data)}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world", default="20261008")
    args = ap.parse_args(argv)
    folder = PRIVATE_ROOT / "worlds" / args.world / "captured"
    sys.path.insert(0, str(REPO_ROOT))

    from agent_logic_1 import meilisearch_client
    from agent_logic_2.nayka_api import api_nayka

    report = []
    regions = api_nayka.site_regions()
    report.append(_save(folder, "regions", regions) if regions else "regions: ПУСТО (МИС недоступна?) — не записано")
    try:
        promos = api_nayka.site_promotions()
        report.append(_save(folder, "promotions", promos))
    except Exception as exc:
        report.append(f"promotions: ошибка {type(exc).__name__}")
    try:
        client = meilisearch_client.get_meilisearch_client()
        docs = client.index("main_index").get_documents({"limit": 1000, "fields": ["title", "content"]}).results
        docs = [dict(d) if not isinstance(d, dict) else d for d in docs]
        report.append(_save(folder, "kb_main_index", [{"title": d.get("title"), "content": d.get("content")} for d in docs]))
    except Exception as exc:
        report.append(f"kb_main_index: ошибка {type(exc).__name__}")
    print("\n".join(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
