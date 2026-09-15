"""Добор `serviceSynonyms` поштучно: массовый метод МИС их не отдаёт.

Замер 10.09 на живом API: `site/serviceInfoAll` возвращает 8472 услуги и
`serviceSynonyms: null` у ВСЕХ, тогда как `site/serviceInfo/{id}` по той же
услуге отдаёт заполненное поле. У обоих ответов совпадает набор полей (16 из 16)
и побайтово совпадают `serviceName`/`description` — это одна запись, наполняемая
двумя разными путями на стороне клиники (сообщено их разработчику).

Пока это не починено, синонимы добираются поштучно после суточной выгрузки:
1622 услуги за ~6 минут, вне пути пациента. Когда массовый метод начнёт отдавать
поле, добор просто перестанет находить новое — снимать его не понадобится.
"""

from __future__ import annotations

import json
from pathlib import Path

from agent_logic_2.nayka_api import api_service_info


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")


def test_backfill_fills_missing_synonyms_from_per_service_method(monkeypatch, tmp_path: Path):
    """Пустое поле в дневном срезе заполняется значением поштучного метода."""
    cache = tmp_path / "service_info_20260910.jsonl"
    _write(cache, [
        {"serviceId": 462, "serviceName": "Общий анализ крови (полный)", "serviceSynonyms": None},
        {"serviceId": 999, "serviceName": "Глюкоза", "serviceSynonyms": None},
    ])
    monkeypatch.setattr(api_service_info, "fetch_service_info_one",
                        lambda sid: {"serviceId": sid,
                                     "serviceSynonyms": "оак, кровь с лейкоформулой" if sid == 462 else None})

    filled = api_service_info.backfill_service_synonyms(cache)

    rows = {r["serviceId"]: r for r in api_service_info.jsonl_read(cache)}
    assert filled == 1
    assert rows[462]["serviceSynonyms"] == "оак, кровь с лейкоформулой"
    assert rows[999]["serviceSynonyms"] is None
    assert rows[462]["serviceName"] == "Общий анализ крови (полный)", "остальные поля не трогаем"


def test_backfill_keeps_value_already_present_in_bulk(monkeypatch, tmp_path: Path):
    """Когда клиника починит массовый метод, добор не должен перезатирать данные."""
    cache = tmp_path / "service_info_20260910.jsonl"
    _write(cache, [{"serviceId": 1, "serviceName": "X", "serviceSynonyms": "из массива"}])

    def _boom(sid):  # noqa: ARG001
        raise AssertionError("поштучный метод не должен вызываться для заполненной строки")

    monkeypatch.setattr(api_service_info, "fetch_service_info_one", _boom)
    assert api_service_info.backfill_service_synonyms(cache) == 0
    assert api_service_info.jsonl_read(cache)[0]["serviceSynonyms"] == "из массива"


def test_backfill_survives_per_service_failures(monkeypatch, tmp_path: Path):
    """Сбой на одной услуге не должен ронять весь утренний прогон."""
    cache = tmp_path / "service_info_20260910.jsonl"
    _write(cache, [
        {"serviceId": 1, "serviceName": "A", "serviceSynonyms": None},
        {"serviceId": 2, "serviceName": "B", "serviceSynonyms": None},
    ])

    def _flaky(sid):
        if sid == 1:
            raise TimeoutError("сеть моргнула")
        return {"serviceId": sid, "serviceSynonyms": "бэ"}

    monkeypatch.setattr(api_service_info, "fetch_service_info_one", _flaky)
    assert api_service_info.backfill_service_synonyms(cache) == 1
    rows = {r["serviceId"]: r for r in api_service_info.jsonl_read(cache)}
    assert rows[1]["serviceSynonyms"] is None
    assert rows[2]["serviceSynonyms"] == "бэ"


def test_backfill_skips_sweep_when_bulk_already_delivered(monkeypatch, tmp_path: Path):
    """Клиника починила массовый метод 15.09 — поштучный обход стал лишним.

    Пока `serviceInfoAll` отдавал `serviceSynonyms: null`, добор был единственным
    способом получить словарь. Теперь выгрузка несёт синонимы сама, и обход 1344
    услуг каждое утро — бессмысленные запросы к чужому API.

    Условие простое: если в срезе есть ХОТЬ ОДИН синоним, значит метод работает,
    и доверяем ему. Если массовая выгрузка снова отдаст пусто — добор включится
    сам и останется страховкой.

    Факт вызова считаем счётчиком, а НЕ исключением из мока: `backfill` ловит
    `Exception` целиком, поэтому проброшенный AssertionError был бы проглочен и
    тест позеленел бы впустую (наблюдалось при написании).
    """
    cache = tmp_path / "service_info_20260915.jsonl"
    _write(cache, [
        {"serviceId": 1, "serviceName": "A", "serviceSynonyms": "из массовой выгрузки"},
        {"serviceId": 2, "serviceName": "B", "serviceSynonyms": None},
    ])
    calls: list[object] = []

    def _record(sid):
        calls.append(sid)
        return {"serviceId": sid, "serviceSynonyms": "не должно попасть"}

    monkeypatch.setattr(api_service_info, "fetch_service_info_one", _record)

    assert api_service_info.backfill_service_synonyms(cache) == 0
    assert calls == [], f"поштучный обход не нужен, а был вызван: {calls}"


def test_backfill_still_runs_when_bulk_returned_nothing(monkeypatch, tmp_path: Path):
    """Страховка: массовый метод снова сломался — добор обязан включиться."""
    cache = tmp_path / "service_info_20260915.jsonl"
    _write(cache, [{"serviceId": 7, "serviceName": "C", "serviceSynonyms": None}])
    calls: list[object] = []

    def _record(sid):
        calls.append(sid)
        return {"serviceId": sid, "serviceSynonyms": "добрано поштучно"}

    monkeypatch.setattr(api_service_info, "fetch_service_info_one", _record)
    assert api_service_info.backfill_service_synonyms(cache) == 1
    assert calls == [7]
