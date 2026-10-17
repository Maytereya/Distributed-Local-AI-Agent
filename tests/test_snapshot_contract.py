"""Контракт среза МИС — ход (д), шаг 1: наблюдатель без смены ответов (17.10).

Было (DATA-6, DATA-9): `snapshots.load_daily` при сбое МИС отдаёт последний непустой срез любой давности, а если
срезов нет — пустой список; ни дата, ни признак «вчерашнее» или «ничего» до потребителя и в лог не доходят — бот
отвечает по недельному прайсу как по свежему, а пустоту выдаёт за «не нашёл».

Стало (решение владельца 08.10, контракт `{строки, дата, статус fresh | stale | unavailable}`): `load_snapshot`
возвращает `Snapshot`; `load_daily` — прежний список строк (`.rows`), потребители не меняются. Статус и давность
уходят в лог (один раз на срез и статус) и в строку телеметрии хода (`src`). Ответ пациенту на этом шаге не
меняется — поведение по давности (приписка, оператор) будет отдельным шагом под выключателем.

Инварианты класса:
- статус следует из диска: сегодняшний непустой — `fresh`, давность 0; сегодняшнего нет или он пуст и скачать не
  вышло — последний непустой, `stale`, давность — разница дат в именах файлов (от «сегодня» самого загрузчика,
  новых часов нет — DATA-8); непустых нет — `unavailable`, строк нет;
- `load_daily` отдаёт ровно строки `load_snapshot`;
- наблюдатель не ломает чтение среза;
- через эндпоинт, два хода: ход, прочитавший прайс, пишет его статус в телеметрию; устаревший прайс даёт тот же
  ответ, что свежий, и статус `stale` с давностью; пустой каталог — `unavailable`.

Тесты загрузчиков — только во временной папке (ловушка `CLAUDE.md`, `test_snapshot_files.py::_isolated`): здесь
срезы пишутся в `tmp_path`, мир стенда — копия.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pytest

from agent_logic_2.nayka_api import snapshots
from evalbench import world as ev_world
from tests.test_price_turns_golden import mini_prod_world, run_dialog  # noqa: F401 — фикстура


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    monkeypatch.setattr(snapshots, "_failed_at", {})
    monkeypatch.setattr(snapshots, "_warned", set())


def _write(path: Path, rows: list[dict]) -> Path:
    snapshots.write_rows(path, rows)
    return path


def _read(path: Path) -> list[dict]:
    import json

    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _failing_download():
    raise ConnectionError("МИС недоступна")


def test_today_snapshot_is_fresh(tmp_path):
    today = _write(tmp_path / "price_region_3_20261017.jsonl", [{"id": 1}])
    _write(tmp_path / "price_region_3_20261016.jsonl", [{"id": 0}])

    snap = snapshots.load_snapshot(today, _failing_download, _read)

    assert (snap.name, snap.rows, snap.date, snap.status, snap.age_days) == ("price_region_3", [{"id": 1}], date(2026, 10, 17), "fresh", 0)


@pytest.mark.parametrize("today_file", ["missing", "empty"])
def test_failed_download_gives_the_last_nonempty_as_stale(today_file, tmp_path, caplog):
    today = tmp_path / "price_region_3_20261017.jsonl"
    if today_file == "empty":
        today.write_text("", encoding="utf-8")
    _write(tmp_path / "price_region_3_20261013.jsonl", [{"id": 13}])
    _write(tmp_path / "price_region_3_20261010.jsonl", [{"id": 10}])

    with caplog.at_level(logging.WARNING, logger=snapshots.log.name):
        first = snapshots.load_snapshot(today, _failing_download, _read)
        again = snapshots.load_snapshot(today, _failing_download, _read)

    assert (first.rows, first.date, first.status, first.age_days) == ([{"id": 13}], date(2026, 10, 13), "stale", 4)
    assert again == first
    stale_warnings = [r for r in caplog.records if "устарел" in r.getMessage()]
    assert len(stale_warnings) == 1, [r.getMessage() for r in caplog.records]  # один раз на срез и статус
    assert "price_region_3" in stale_warnings[0].getMessage() and "4" in stale_warnings[0].getMessage()


def test_no_snapshot_at_all_is_unavailable(tmp_path, caplog):
    today = tmp_path / "service_info_20261017.jsonl"

    with caplog.at_level(logging.WARNING, logger=snapshots.log.name):
        snap = snapshots.load_snapshot(today, _failing_download, _read)

    assert (snap.name, snap.rows, snap.date, snap.status, snap.age_days) == ("service_info", [], None, "unavailable", None)
    assert any("недоступен" in r.getMessage() for r in caplog.records)


def test_successful_download_is_fresh(tmp_path):
    today = tmp_path / "price_units_20261017.jsonl"
    _write(tmp_path / "price_units_20261001.jsonl", [{"id": "old"}])

    snap = snapshots.load_snapshot(today, lambda: _write(today, [{"id": "new"}]), _read)

    assert (snap.rows, snap.status, snap.age_days) == ([{"id": "new"}], "fresh", 0)


@pytest.mark.parametrize("files", [["20261017"], ["20261012"], []])
def test_load_daily_is_the_rows_of_load_snapshot(files, tmp_path):
    for day in files:
        _write(tmp_path / f"price_region_3_{day}.jsonl", [{"day": day}])
    today = tmp_path / "price_region_3_20261017.jsonl"

    rows = snapshots.load_daily(today, _failing_download, _read)

    snapshots._failed_at.clear()
    assert rows == snapshots.load_snapshot(today, _failing_download, _read).rows


def test_observer_sees_each_read_and_cannot_break_it(tmp_path, monkeypatch):
    today = _write(tmp_path / "price_region_3_20261017.jsonl", [{"id": 1}])
    seen: list[snapshots.Snapshot] = []
    monkeypatch.setattr(snapshots, "_observer", seen.append)

    snapshots.load_snapshot(today, _failing_download, _read)
    assert [(s.name, s.status) for s in seen] == [("price_region_3", "fresh")]

    def broken(_snapshot):
        raise RuntimeError("наблюдатель упал")

    monkeypatch.setattr(snapshots, "_observer", broken)
    assert snapshots.load_daily(today, _failing_download, _read) == [{"id": 1}]


def test_the_router_registers_turn_telemetry_as_the_observer():
    from messengers_router import turn_telemetry
    from messengers_router.services import core  # noqa: F401 — регистрация при импорте

    assert snapshots._observer is turn_telemetry.note_snapshot


# --------------------------------------------------------------------------- через эндпоинт, два хода

_DIALOG = ["сколько стоит ферритин", "а цена?"]


def _price_src(rows: list[dict]) -> list[dict | None]:
    return [(row.get("src") or {}).get("price_region_3") for row in rows]


def _play(world_dir: Path, monkeypatch, session: str, *, shift_to: str | None = None, empty_price: Path | None = None):
    """Два хода в мире мини-прода (день, сеть закрыта); `shift_to` — «сегодня» загрузчика прайса (срез мира
    остаётся датой мира — прайс устарел), `empty_price` — пустая папка прайса (непустых срезов нет)."""

    from agent_logic_2.nayka_api import api_price

    with ev_world.FrozenWorld(world_dir, "day") as world, monkeypatch.context() as patch:
        if shift_to is not None:
            real = api_price.price_by_region_path.__wrapped__
            patch.setattr(api_price, "price_by_region_path", lambda region_id, date=None: real(region_id, date or shift_to))
        if empty_price is not None:
            patch.setattr(api_price, "PRICE_BY_REGION_DIR", empty_price)
        records = run_dialog(_DIALOG, session)
        rows = list(world.telemetry)
        attempts = list(world.net.attempts)
    assert attempts == []
    assert len(rows) == len(_DIALOG)
    return records, rows


def test_price_turn_reports_a_fresh_snapshot(mini_prod_world, monkeypatch):  # noqa: F811
    records, rows = _play(mini_prod_world, monkeypatch, "snapshot-fresh")

    assert "Ферритин" in records[0]["text"], records[0]["text"]
    assert _price_src(rows)[0] == {"status": "fresh", "age_days": 0}
    assert all(entry in (None, {"status": "fresh", "age_days": 0}) for entry in _price_src(rows))


def test_stale_price_answers_the_same_and_reports_its_age(mini_prod_world, monkeypatch):  # noqa: F811
    fresh, _rows = _play(mini_prod_world, monkeypatch, "snapshot-same")
    stale, rows = _play(mini_prod_world, monkeypatch, "snapshot-same", shift_to="20261012")

    assert [r["text"] for r in stale] == [r["text"] for r in fresh]  # наблюдатель ответа не меняет
    assert _price_src(rows)[0] == {"status": "stale", "age_days": 4}


def test_no_price_snapshot_is_reported_unavailable(mini_prod_world, monkeypatch, tmp_path):  # noqa: F811
    empty = tmp_path / "price_by_region"
    empty.mkdir()

    _records, rows = _play(mini_prod_world, monkeypatch, "snapshot-none", empty_price=empty)

    assert _price_src(rows)[0] == {"status": "unavailable", "age_days": None}
    assert list(empty.iterdir()) == []  # мир не пишет срезов


# --------------------------------------------------------------------------- шаг 1б: срезы со своим выбором файла
#
# Врачи (`api_nayka.get_cached_doctors_data`), цены врачей (`api_price.load_doctor_prices`) и словарь синонимов
# (`_biomaterial._vocab_source`) выбирают файл сами — статус сообщает `snapshots.report`. Возвращаемые строки — те же
# (тот же объект), что и до шага.


def test_report_statuses(tmp_path):
    today = tmp_path / "doctors_20261017.jsonl"
    rows = [{"id": 1}]

    assert snapshots.report(today, today, rows).rows is rows
    assert (snapshots.report(today, tmp_path / "doctors_20261015.jsonl", rows).status, snapshots.report(today, tmp_path / "doctors_20261015.jsonl", rows).age_days) == ("stale", 2)
    # Врачи до 06:00 по Самаре: активная дата — вчера, а читается сегодняшний файл — он не устарел.
    newer = snapshots.report(tmp_path / "doctors_20261016.jsonl", today, rows)
    assert (newer.status, newer.age_days) == ("fresh", 0)
    assert (snapshots.report(today, None).status, snapshots.report(today, None).rows) == ("unavailable", [])


@pytest.fixture
def seen(monkeypatch) -> list[snapshots.Snapshot]:
    out: list[snapshots.Snapshot] = []
    monkeypatch.setattr(snapshots, "_observer", out.append)
    return out


@pytest.fixture
def doctor_prices_dir(monkeypatch, tmp_path) -> Path:
    from agent_logic_2.nayka_api import api_price

    folder = tmp_path / "doctor_prices"
    folder.mkdir()
    monkeypatch.setattr(api_price, "PRICES_DIR", folder)
    monkeypatch.setattr(api_price, "cleanup_old", lambda *args, **kwargs: None)
    monkeypatch.setattr(api_price, "doctor_prices_path", lambda date=None: folder / f"doctor_prices_{date or '20261017'}.jsonl")

    def no_build():
        raise ConnectionError("МИС недоступна")

    monkeypatch.setattr(api_price, "update_doctor_prices", no_build)
    return folder


@pytest.mark.parametrize(
    ("files", "expected"),
    [
        ({"20261017": [{"id": 1}]}, ("fresh", 0, [{"id": 1}])),
        ({"20261009": [{"id": 9}]}, ("stale", 8, [{"id": 9}])),
        ({"20261017": [], "20261009": [{"id": 9}]}, ("stale", 8, [{"id": 9}])),
        ({"20261017": []}, ("unavailable", None, [])),
    ],
    ids=["today", "older", "empty_today_older", "empty_today_only"],
)
def test_doctor_prices_report_their_status(files, expected, doctor_prices_dir, seen):
    from agent_logic_2.nayka_api import api_price

    for day, rows in files.items():
        path = doctor_prices_dir / f"doctor_prices_{day}.jsonl"
        path.write_text("", encoding="utf-8") if not rows else snapshots.write_rows(path, rows)

    rows = api_price.load_doctor_prices()

    status, age, expected_rows = expected
    assert rows == expected_rows
    assert [(s.name, s.status, s.age_days) for s in seen] == [("doctor_prices", status, age)]


def test_doctor_prices_build_failure_is_unavailable_and_still_raises(doctor_prices_dir, seen):
    from agent_logic_2.nayka_api import api_price

    with pytest.raises(ConnectionError):
        api_price.load_doctor_prices()
    assert [(s.name, s.status) for s in seen] == [("doctor_prices", "unavailable")]


@pytest.fixture
def doctors_dir(monkeypatch, tmp_path) -> Path:
    from agent_logic_2.nayka_api import api_nayka

    monkeypatch.setattr(api_nayka, "DATA_DIR", tmp_path)
    monkeypatch.setattr(api_nayka, "get_active_date_str", lambda: "20261017")
    monkeypatch.setattr(api_nayka, "cleanup_old_doctors_files", lambda *args, **kwargs: None)
    monkeypatch.setattr(api_nayka, "_doctors_schema_outdated", lambda doctors: False)

    def no_download():
        raise ConnectionError("МИС недоступна")

    monkeypatch.setattr(api_nayka, "get_all_doctors", no_download)
    return tmp_path


@pytest.mark.parametrize(
    ("days", "expected"),
    [(["20261017"], ("fresh", 0)), (["20261014"], ("stale", 3)), ([], ("unavailable", None))],
    ids=["active", "older", "none"],
)
def test_doctors_report_their_status(days, expected, doctors_dir, seen):
    from agent_logic_2.nayka_api import api_nayka

    for day in days:
        snapshots.write_rows(doctors_dir / f"doctors_{day}.jsonl", [{"id": day}])

    rows = api_nayka.get_cached_doctors_data()

    assert rows == [{"id": day} for day in days]
    assert [(s.name, s.status, s.age_days) for s in seen] == [("doctors", *expected)]


@pytest.mark.parametrize(
    ("days", "expected"),
    [(["20261017"], ("fresh", 0)), (["20261016"], ("stale", 1)), ([], ("unavailable", None))],
    ids=["today", "night_yesterday", "none"],
)
def test_vocabulary_source_reports_its_status(days, expected, monkeypatch, tmp_path, seen):
    from agent_logic_2.nayka_api import api_service_info
    from messengers_router.services import _biomaterial

    monkeypatch.setattr(api_service_info, "service_info_path", lambda date=None: tmp_path / f"service_info_{date or '20261017'}.jsonl")
    for day in days:
        snapshots.write_rows(tmp_path / f"service_info_{day}.jsonl", [{"id": day}])

    source = _biomaterial._vocab_source()

    assert source == (tmp_path / f"service_info_{days[0]}.jsonl" if days else None)
    assert [(s.name, s.status, s.age_days) for s in seen] == [("service_info", *expected)]


def test_turn_reports_each_source_it_read(mini_prod_world, monkeypatch):  # noqa: F811
    """Health раздельно по источникам (решение 08.10): строка хода называет каждый прочитанный срез. В мире мини-прода
    прайс и priceUnits есть, а врачей, цен врачей и памяток нет — ход это и пишет."""

    _records, rows = _play(mini_prod_world, monkeypatch, "snapshot-sources")

    assert rows[0]["src"] == {
        "price_region_3": {"status": "fresh", "age_days": 0},
        "price_units": {"status": "fresh", "age_days": 0},
        "service_info": {"status": "unavailable", "age_days": None},
        "doctors": {"status": "unavailable", "age_days": None},
        "doctor_prices": {"status": "unavailable", "age_days": None},
    }
