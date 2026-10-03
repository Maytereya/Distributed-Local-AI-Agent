"""Класс `empty_snapshot_poisons_day`: пустой или недоступный ответ МИС не оставляет бота
без справочника на весь день (03.10).

## Что было

Загрузчики прайса региона, priceUnits и serviceInfoAll проверяли только, ЕСТЬ ли файл
«на сегодня». Ответ МИС «200 и пустой список» ложился файлом 0 байт — и до следующего дня
бот работал без прайса или без памяток подготовки и синонимов. Живое свидетельство —
`price_region_1_20260602.jsonl`, 0 байт; та же беда с doctor_prices 30.09
(BUG-2026-09-25-NO-DOCTORS-FOUND-FALSE). Обновление serviceInfoAll в 08:20 перезаписывает
файл принудительно — пустой ответ затёр бы и скачанный ночью хороший срез.

Сбой МИС: пока она не отвечает, каждый запрос пациента заново ждал тайм-аут скачивания
(60 с с повторами; замер 03.10 — 272 с на список врачей) ради того же вчерашнего кэша.

## Инварианты

- пустой ответ МИС не записывается и не затирает срез;
- нет сегодняшнего среза и МИС недоступна — последний непустой срез, а не пусто/ошибка;
- после сбоя следующая попытка скачать — не раньше чем через `RETRY_AFTER_S`;
- запись атомарна: читатель видит прежний файл или новый целиком.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_logic_2.nayka_api import api_nayka, api_price, api_service_info, snapshots

_YESTERDAY_ROW = {"serviceName": "Общий анализ крови", "cost": 390}


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path: Path):
    """Каждый тест — с чистой паузой после сбоев и БЕЗ доступа к настоящему кэшу.

    Очистка старых срезов работает по папкам модулей, а не по пути среза: на коде до
    фикса тест, дошедший до записи, удалял настоящие срезы разработчика (03.10 так
    пропали прайс и serviceInfoAll, и гейт упал на пустом каталоге). Папки — во
    временную, очистка — пустышка.
    """

    monkeypatch.setattr(snapshots, "_failed_at", {})
    monkeypatch.setattr(api_price, "cleanup_old", lambda *args, **kwargs: None)
    monkeypatch.setattr(api_service_info, "_cleanup_old", lambda: None)
    for module, attr in (
        (api_price, "PRICE_BY_REGION_DIR"),
        (api_price, "PRICE_UNITS_DIR"),
        (api_service_info, "SERVICE_INFO_DIR"),
    ):
        monkeypatch.setattr(module, attr, tmp_path)


def _snapshot(path: Path, rows: list[dict]) -> Path:
    snapshots.write_rows(path, rows)
    return path


# --- Модуль срезов ----------------------------------------------------------------


def test_write_is_atomic_and_leaves_no_temp_file(tmp_path: Path):
    target = _snapshot(tmp_path / "price_region_3_20261003.jsonl", [_YESTERDAY_ROW, {"serviceName": "ТТГ"}])
    assert api_price.jsonl_read(target) == [_YESTERDAY_ROW, {"serviceName": "ТТГ"}]
    assert [p.name for p in tmp_path.iterdir()] == [target.name]


def test_latest_nonempty_skips_empty_and_future_snapshots(tmp_path: Path):
    good = _snapshot(tmp_path / "price_region_3_20261001.jsonl", [_YESTERDAY_ROW])
    (tmp_path / "price_region_3_20261002.jsonl").write_text("", encoding="utf-8")
    _snapshot(tmp_path / "price_region_3_20261005.jsonl", [_YESTERDAY_ROW])
    today = tmp_path / "price_region_3_20261003.jsonl"

    assert snapshots.latest_nonempty(today) == good
    assert snapshots.latest_nonempty(tmp_path / "price_region_3_20260930.jsonl") is None


def test_empty_answer_is_not_a_snapshot(tmp_path: Path):
    yesterday = _snapshot(tmp_path / "price_region_3_20261002.jsonl", [_YESTERDAY_ROW])
    today = tmp_path / "price_region_3_20261003.jsonl"

    def mis_returns_nothing():
        snapshots.require_rows([], "priceByRegion(3)")

    rows = snapshots.load_daily(today, mis_returns_nothing, api_price.jsonl_read)

    assert rows == [_YESTERDAY_ROW]
    assert not today.exists(), "пустой ответ МИС не должен лечь срезом на весь день"
    assert yesterday.exists()


def test_after_a_failure_patients_do_not_wait_for_mis_again(tmp_path: Path):
    _snapshot(tmp_path / "service_info_20261002.jsonl", [_YESTERDAY_ROW])
    today = tmp_path / "service_info_20261003.jsonl"
    calls: list[int] = []

    def mis_down():
        calls.append(1)
        raise ConnectionError("МИС не отвечает")

    for _ in range(3):
        assert snapshots.load_daily(today, mis_down, api_service_info.jsonl_read) == [_YESTERDAY_ROW]
    assert len(calls) == 1


def test_legacy_empty_today_file_is_downloaded_again(tmp_path: Path):
    today = tmp_path / "price_units_20261003.jsonl"
    today.write_text("", encoding="utf-8")  # такие файлы мог оставить прежний код

    def download():
        snapshots.write_rows(today, [{"id": 146}])

    assert snapshots.load_daily(today, download, api_price.jsonl_read) == [{"id": 146}]


# --- Загрузчики справочников ------------------------------------------------------


def test_price_by_region_survives_empty_mis_answer(monkeypatch, tmp_path: Path):
    _snapshot(tmp_path / "price_region_3_20261002.jsonl", [_YESTERDAY_ROW])
    today = tmp_path / "price_region_3_20261003.jsonl"
    monkeypatch.setattr(api_price, "price_by_region_path", lambda region_id, date=None: today)
    monkeypatch.setattr(api_price, "fetch_price_by_region", lambda region_id: [])

    assert api_price.load_price_by_region(3) == [_YESTERDAY_ROW]
    assert not today.exists()


def test_price_by_region_survives_mis_error(monkeypatch, tmp_path: Path):
    # 25.09: priceByRegion/3 отдавал 400 — раньше исключение уходило в ответ пациенту.
    _snapshot(tmp_path / "price_region_3_20261002.jsonl", [_YESTERDAY_ROW])
    today = tmp_path / "price_region_3_20261003.jsonl"
    monkeypatch.setattr(api_price, "price_by_region_path", lambda region_id, date=None: today)

    def http_400(region_id):
        raise RuntimeError("400 Client Error")

    monkeypatch.setattr(api_price, "fetch_price_by_region", http_400)

    assert api_price.load_price_by_region(3) == [_YESTERDAY_ROW]


def test_morning_refresh_with_empty_answer_keeps_service_info(monkeypatch, tmp_path: Path):
    # Обновление в 08:20 перезаписывает срез принудительно; пустой ответ его не затирает.
    good = {"serviceName": "Ферритин", "preparation": "Кровь сдают утром натощак."}
    today = _snapshot(tmp_path / "service_info_20261003.jsonl", [good])
    monkeypatch.setattr(api_service_info, "service_info_path", lambda date=None: today)
    monkeypatch.setattr(api_service_info, "fetch_service_info_all", lambda: [])

    with pytest.raises(snapshots.EmptySnapshotError):
        api_service_info.update_service_info(force=True)
    assert api_service_info.load_service_info() == [good]


def test_doctors_loader_does_not_retry_mis_on_every_request(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(api_nayka, "DATA_DIR", tmp_path)
    monkeypatch.setattr(api_nayka, "get_active_date_str", lambda: "20261003")
    monkeypatch.setattr(api_nayka, "find_existing_doctors_file", lambda: None)
    calls: list[int] = []

    def mis_down():
        calls.append(1)
        raise ConnectionError("Read timed out")

    monkeypatch.setattr(api_nayka, "get_all_doctors", mis_down)

    for _ in range(3):
        assert api_nayka.get_cached_doctors_data() == []
    assert len(calls) == 1


def test_failed_write_leaves_no_temp_file(tmp_path: Path):
    target = tmp_path / "price_units_20261003.jsonl"

    def broken_rows():
        yield {"id": 1}
        raise OSError("диск переполнен")

    with pytest.raises(OSError):
        snapshots.write_rows(target, broken_rows())
    assert list(tmp_path.iterdir()) == []
