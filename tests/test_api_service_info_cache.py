from pathlib import Path

from agent_logic_2.nayka_api import api_service_info


def test_service_info_base_url_prefers_no_site_setting(monkeypatch):
    monkeypatch.setattr(api_service_info.c, "nayka_base_url_no_site", "http://example.test/api/v1")
    monkeypatch.setattr(api_service_info.c, "nayka_base_url", "http://example.test/api/v1/site")

    assert api_service_info._service_info_base_url() == "http://example.test/api/v1"


def test_service_info_base_url_strips_trailing_site_when_no_site_missing(monkeypatch):
    monkeypatch.setattr(api_service_info.c, "nayka_base_url_no_site", "")
    monkeypatch.setattr(api_service_info.c, "nayka_base_url", "http://example.test/api/v1/site")

    assert api_service_info._service_info_base_url() == "http://example.test/api/v1"


def test_load_service_info_builds_today_cache_on_first_access(monkeypatch, tmp_path: Path):
    target = tmp_path / "service_info_20260331.jsonl"
    calls = {"update": 0}

    monkeypatch.setattr(api_service_info, "service_info_path", lambda date=None: target)

    def fake_update(force: bool = False):
        _ = force
        calls["update"] += 1
        target.write_text(
            '{"serviceName":"Анализ крови на холестерин","preparation":"Кровь сдаётся натощак."}\n',
            encoding="utf-8",
        )
        return target

    monkeypatch.setattr(api_service_info, "update_service_info", fake_update)

    rows = api_service_info.load_service_info()

    assert calls["update"] == 1
    assert rows and rows[0]["serviceName"] == "Анализ крови на холестерин"


def test_update_service_info_keeps_rows_with_useful_fields(monkeypatch, tmp_path: Path):
    target = tmp_path / "service_info_20260331.jsonl"

    monkeypatch.setattr(api_service_info, "service_info_path", lambda date=None: target)
    monkeypatch.setattr(api_service_info, "_cleanup_old", lambda: None)
    monkeypatch.setattr(
        api_service_info,
        "fetch_service_info_all",
        lambda: [
            {"serviceName": "Пустая запись", "description": "", "indication": "", "preparation": ""},
            {"serviceName": "ЭКГ", "description": "Описание ЭКГ", "preparation": ""},
            {"serviceName": "Холестерин", "preparation": "Кровь сдаётся натощак.", "extra": {"foo": "bar"}},
        ],
    )

    out = api_service_info.update_service_info(force=True)
    rows = api_service_info.jsonl_read(out)

    assert out == target
    assert len(rows) == 2
    assert rows[0]["serviceName"] == "ЭКГ"
    assert rows[1]["serviceName"] == "Холестерин"
    assert rows[1]["extra"] == {"foo": "bar"}


def test_ensure_daily_service_info_refresh_started_schedules_loop_and_warmup(monkeypatch, tmp_path: Path):
    target = tmp_path / "service_info_20260331.jsonl"

    class DummyTask:
        """Простой объект задачи для проверки запуска планировщика."""

        def __init__(self, coro):
            self.coro = coro

        def done(self) -> bool:
            return False

    class DummyLoop:
        """Фиктивный event loop, который собирает созданные coroutine."""

        def __init__(self):
            self.created = []

        def create_task(self, coro):
            self.created.append(coro)
            return DummyTask(coro)

    loop = DummyLoop()
    monkeypatch.setattr(api_service_info.asyncio, "get_running_loop", lambda: loop)
    monkeypatch.setattr(api_service_info, "_SERVICE_INFO_REFRESH_TASK", None)
    monkeypatch.setattr(api_service_info, "service_info_path", lambda date=None: target)

    started = api_service_info.ensure_daily_service_info_refresh_started()

    assert started is True
    assert len(loop.created) == 2
    for coro in loop.created:
        coro.close()


def test_ensure_daily_service_info_refresh_started_without_loop_returns_false(monkeypatch):
    def fail_loop():
        raise RuntimeError("no loop")

    monkeypatch.setattr(api_service_info.asyncio, "get_running_loop", fail_loop)

    assert api_service_info.ensure_daily_service_info_refresh_started() is False
