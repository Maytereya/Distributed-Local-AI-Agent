"""Выдержка из среза врачей — запасной путь гейта и CI без локального `doctors_*.jsonl` (ход (ж), решение владельца 18.10).

Было: срез врачей не в git; без него ~68 тестов записи и врачей в 11 файлах красные (замер 18.10), в CI они не
попадают, а локальный гейт ломался, когда срез стирало обновление (ловушка CLAUDE.md, 04.10).

Стало: `tests/fixtures/doctors_excerpt/doctors_20261008.jsonl` — врачи, названные в тестах, и до 4 на специальность;
описание врача — только первая строка (должность), без биографии (решение владельца: выдержка без описаний).
`conftest.doctors_fallback` отдаёт её, только когда в папке срезов гейта нет ни одного среза врачей и тест эту папку не
подменял; локальный срез — всегда первым.

Инварианты: выдержка — в схеме загрузчика и без описаний; запасной путь не перебивает ни локальный срез, ни
подменённую тестом папку.
"""

from __future__ import annotations

import json
from pathlib import Path

from agent_logic_2.nayka_api import api_nayka
from tests.conftest import _DOCTORS_EXCERPT, doctors_fallback

_FIELDS = {"fio", "id", "main_specializations", "main_units", "ord", "region_ids", "regions", "specialization", "unit_links", "units"}


def _rows() -> list[dict]:
    return [json.loads(line) for line in _DOCTORS_EXCERPT.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_excerpt_is_small_in_loader_schema_and_without_biographies():
    rows = _rows()

    assert 20 <= len(rows) <= 100
    assert all(set(r) <= _FIELDS for r in rows)
    assert all("\n" not in str(r["specialization"]) and len(str(r["specialization"])) <= 160 for r in rows)
    assert not api_nayka._doctors_schema_outdated(api_nayka.load_doctors_data(_DOCTORS_EXCERPT))


def test_doctors_named_in_booking_tests_are_in_the_excerpt():
    surnames = {str(r["fio"]).split()[0] for r in _rows()}
    for name in ("Трубин", "Дразнин"):
        assert name in surnames


def test_fallback_only_when_the_gate_folder_has_no_doctors(monkeypatch, tmp_path):
    gate_dir = tmp_path / "gate"
    gate_dir.mkdir()
    monkeypatch.setattr(api_nayka, "DATA_DIR", gate_dir)
    local = gate_dir / "doctors_20261017.jsonl"

    find = doctors_fallback(lambda: None, gate_dir, _DOCTORS_EXCERPT)
    assert find() == _DOCTORS_EXCERPT  # своего среза нет — выдержка

    assert doctors_fallback(lambda: local, gate_dir, _DOCTORS_EXCERPT)() == local  # локальный срез — первым

    other = tmp_path / "world"
    other.mkdir()
    monkeypatch.setattr(api_nayka, "DATA_DIR", other)
    assert find() is None  # тест подменил папку (мир стенда) — без подстановки


def test_loader_reads_the_excerpt_when_nothing_else_exists(monkeypatch, tmp_path):
    monkeypatch.setattr(api_nayka, "DATA_DIR", tmp_path)
    monkeypatch.setattr(api_nayka, "get_active_date_str", lambda: "20261018")
    monkeypatch.setattr(api_nayka, "get_all_doctors", lambda: [])
    monkeypatch.setattr(api_nayka, "find_existing_doctors_file", doctors_fallback(lambda: None, Path(tmp_path), _DOCTORS_EXCERPT))
    from agent_logic_2.nayka_api import snapshots

    monkeypatch.setattr(snapshots, "_failed_at", {})

    rows = api_nayka.get_cached_doctors_data()

    assert len(rows) == len(_rows())
    assert list(tmp_path.iterdir()) == []  # выдержку не копируем в папку срезов
