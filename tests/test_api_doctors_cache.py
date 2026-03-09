import json

from agent_logic_2.nayka_api import api_nayka


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_find_existing_doctors_file_skips_empty_active(tmp_path, monkeypatch):
    active = tmp_path / "doctors_20260307.jsonl"
    prev = tmp_path / "doctors_20260306.jsonl"
    active.write_text("", encoding="utf-8")
    _write_jsonl(prev, [{"id": 1, "fio": "Трубин Алексей Юрьевич"}])

    monkeypatch.setattr(api_nayka, "DATA_DIR", tmp_path)
    monkeypatch.setattr(api_nayka, "get_active_date_str", lambda: "20260307")

    assert api_nayka.find_existing_doctors_file() == prev


def test_get_cached_doctors_data_falls_back_to_last_nonempty_when_refresh_empty(tmp_path, monkeypatch):
    active = tmp_path / "doctors_20260307.jsonl"
    prev = tmp_path / "doctors_20260306.jsonl"
    active.write_text("", encoding="utf-8")
    rows = [{"id": 1, "fio": "Трубин Алексей Юрьевич"}]
    _write_jsonl(prev, rows)

    monkeypatch.setattr(api_nayka, "DATA_DIR", tmp_path)
    monkeypatch.setattr(api_nayka, "get_active_date_str", lambda: "20260307")
    monkeypatch.setattr(api_nayka, "get_all_doctors", lambda: [])

    assert api_nayka.get_cached_doctors_data() == rows


def test_save_doctors_data_does_not_create_empty_active_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(api_nayka, "DATA_DIR", tmp_path)
    monkeypatch.setattr(api_nayka, "get_active_date_str", lambda: "20260307")
    monkeypatch.setattr(api_nayka, "cleanup_old_doctors_files", lambda keep_dates=None: None)

    api_nayka.save_doctors_data([])

    assert not (tmp_path / "doctors_20260307.jsonl").exists()
