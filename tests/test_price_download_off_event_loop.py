"""Скачивание срезов прайса не идёт в потоке цикла событий — ход (д), шаг 9б, DATA-7 (17.10).

Было: uvicorn — один процесс и один цикл событий. Ценовой ход, у которого нет среза «на сегодня» (первый вопрос нового
дня по часам сервера, свежий том), синхронно звал `resolve_price_service_name_from_catalog` / `query_has_unsatisfiable_
qualifier` без строк и `_annotate_price_rows_with_care_context` → `load_price_by_region` / `load_price_units` →
скачивание у МИС (60 с на попытку, три повтора) — прямо в цикле событий: все пациенты ждали.

Стало: в начале `price_info` и `service_bundle_info` шаг «скачать сегодняшний срез, если его нет» (`api_price.
prefetch_price_today`) идёт в потоке (`asyncio.to_thread`). Дальнейшие синхронные чтения хода уже не качают: срез есть,
а после сбоя действует пауза повтора (`snapshots.RETRY_AFTER_S`). Данные и ответ те же.

Инвариант класса: в ценовом ходе скачивание среза МИС не выполняется в потоке цикла событий. Проверка — через эндпоинт,
два хода, мир мини-прода, «сегодня» загрузчиков прайса сдвинуто вперёд (среза на сегодня нет — загрузчик идёт качать).
"""

from __future__ import annotations

import threading
from pathlib import Path

from evalbench import world as ev_world
from tests.test_price_turns_golden import mini_prod_world, run_dialog  # noqa: F401 — фикстура

_DIALOG = ["сколько стоит ферритин", "а цена?"]


def _play(world_dir: Path, monkeypatch, session: str, *, shift_to: str | None):
    from agent_logic_2.nayka_api import api_price

    downloads: list[tuple[str, bool]] = []
    with ev_world.FrozenWorld(world_dir, "day") as world, monkeypatch.context() as patch:
        if shift_to is not None:
            for attr in ("price_by_region_path", "price_units_path"):
                real = getattr(api_price, attr).__wrapped__
                if attr == "price_by_region_path":
                    patch.setattr(api_price, attr, lambda region_id, date=None, _real=real: _real(region_id, date or shift_to))
                else:
                    patch.setattr(api_price, attr, lambda date=None, _real=real: _real(date or shift_to))
        for attr in ("update_price_by_region", "update_price_units"):
            real_update = getattr(api_price, attr)

            def spy(*args, _real=real_update, _name=attr, **kwargs):
                downloads.append((_name, threading.current_thread() is threading.main_thread()))
                return _real(*args, **kwargs)

            patch.setattr(api_price, attr, spy)
        records = run_dialog(_DIALOG, session)
        attempts = list(world.net.attempts)
    assert attempts == []
    return records, downloads


def test_price_snapshot_download_runs_off_the_event_loop(mini_prod_world, monkeypatch):  # noqa: F811
    records, downloads = _play(mini_prod_world, monkeypatch, "download-off-loop", shift_to="20261012")

    assert {name for name, _ in downloads} == {"update_price_by_region", "update_price_units"}, downloads
    on_loop = [name for name, in_main in downloads if in_main]  # asyncio.run держит цикл в главном потоке
    assert on_loop == [], downloads
    assert "Ферритин" in records[0]["text"], records[0]["text"]


def test_answers_are_the_same_as_before(mini_prod_world, monkeypatch):  # noqa: F811
    fresh, downloads = _play(mini_prod_world, monkeypatch, "download-same", shift_to=None)
    shifted, _ = _play(mini_prod_world, monkeypatch, "download-same", shift_to="20261012")

    assert downloads == []  # срез «на сегодня» есть — качать нечего
    # Ответ тот же; срез 4 дней под `mis_honest_degradation = on` (прод с 18.10) — с припиской даты данных (`services/_freshness`).
    assert shifted[0]["text"].rstrip().endswith("по данным на 08.10._"), shifted[0]["text"]
    assert [r["text"].replace(", по данным на 08.10._", "._") for r in shifted] == [r["text"] for r in fresh]
