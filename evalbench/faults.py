"""Сбои МИС на сетевой границе — стенд хода (д), шаг 2 (17.10; TEST-3 ревью `dla/02_review.md`).

Тесты и стенд имитировали сбой МИС исключением, которого прод не бросает (`raise RuntimeError` из подменённого
загрузчика), — проверялась честная ветка, недостижимая на проде (DATA-6). Здесь сбой ставится туда же, где он случается
на проде, — на HTTP-границу клиентов МИС, а всё выше работает настоящим кодом:

- живые методы (`api_nayka._session_get`: /regions, /promotions, расписание и его «шапка» /doctors) и скачивание срезов
  (`api_price.SESSION.get`, `api_service_info.SESSION.get`) — один из видов `LIVE_KINDS`:
  `network` — `requests.ConnectionError`; `timeout` — `requests.ReadTimeout`; `empty` — 200 и `[]`;
  `http_5xx` — 503, `raise_for_status` бросает `requests.HTTPError`. «Строка-ошибка» — не вид сбоя границы, а соглашение
  уровнем выше (`find_doctor_schedule` превращает `RequestException` в «Не удалось получить …») — её дают те же виды;
- дневные срезы на диске — `SNAPSHOT_KINDS`: `stale` — «сегодня» загрузчиков сдвинуто на `stale_days` вперёд (срезы мира
  устарели; загрузчик идёт качать — граница отвечает сбоем); `none` — папки срезов пусты (непустых срезов нет).

`FrozenWorld` подменяет `site_regions` / `site_promotions` записью мира; под сбоем живых методов фабрика возвращает
настоящие функции (оригиналы — из журнала отката мира), чтобы запрос дошёл до границы. Входить внутри открытого мира;
откат — при выходе (до выхода из мира).
"""

from __future__ import annotations

import datetime as _dt
import tempfile
from pathlib import Path
from typing import Any, Callable

LIVE_KINDS = ("network", "timeout", "empty", "http_5xx")
SNAPSHOT_KINDS = ("stale", "none")

_DATED_PATHS = (
    ("agent_logic_2.nayka_api.api_price", "price_by_region_path"),
    ("agent_logic_2.nayka_api.api_price", "price_units_path"),
    ("agent_logic_2.nayka_api.api_price", "doctor_prices_path"),
    ("agent_logic_2.nayka_api.api_service_info", "service_info_path"),
)


class _Response:
    """Ответ границы: `status_code`, `json()`, `raise_for_status()` — то, что читают клиенты МИС."""

    def __init__(self, status: int, payload: Any) -> None:
        self.status_code = status
        self._payload = payload
        self.text = ""

    def json(self) -> Any:
        return self._payload

    def raise_for_status(self) -> None:
        import requests

        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} стенд: сбой МИС", response=self)  # type: ignore[arg-type]


def boundary(kind: str, calls: list[str]) -> Callable[..., Any]:
    """Подмена `get` клиента МИС: каждый вызов — в `calls` (хвост пути), ответ — по виду сбоя."""

    import requests

    if kind not in LIVE_KINDS:
        raise ValueError(f"вид сбоя: {kind!r}, ожидался один из {LIVE_KINDS}")

    def get(url: Any, *_args: Any, **_kwargs: Any) -> Any:
        calls.append(str(url).rsplit("/api/", 1)[-1].split("?")[0])
        if kind == "network":
            raise requests.ConnectionError("стенд: МИС недоступна (сеть)")
        if kind == "timeout":
            raise requests.ReadTimeout("стенд: МИС не ответила (таймаут)")
        if kind == "empty":
            return _Response(200, [])
        return _Response(503, None)

    return get


def _original(world: Any, obj: Any, attr: str) -> Any:
    """Значение атрибута до входа в мир (из журнала отката `FrozenWorld`)."""

    for target, name, value in getattr(world, "_undo", ()):
        if target is obj and name == attr:
            return value
    return getattr(obj, attr)


class Faults:
    """Сбой поверх открытого `FrozenWorld`.

    :param world: открытый мир стенда
    :param live: вид сбоя живых методов и скачивания срезов (`LIVE_KINDS`) или None
    :param snapshots: вид сбоя срезов на диске (`SNAPSHOT_KINDS`) или None
    :param stale_days: на сколько дней «сегодня» загрузчиков позже даты мира (для `stale`)
    """

    def __init__(self, world: Any, *, live: str | None = None, snapshots: str | None = None, stale_days: int = 0) -> None:
        if snapshots is not None and snapshots not in SNAPSHOT_KINDS:
            raise ValueError(f"сбой срезов: {snapshots!r}, ожидался один из {SNAPSHOT_KINDS}")
        if snapshots == "stale" and stale_days <= 0:
            raise ValueError("stale: нужен stale_days > 0")
        self.world = world
        self.live = live
        self.snapshots = snapshots
        self.stale_days = stale_days
        self.calls: list[str] = []
        self._undo: list[tuple[Any, str, Any]] = []
        self._tmp: tempfile.TemporaryDirectory | None = None

    def _set(self, obj: Any, attr: str, value: Any) -> None:
        self._undo.append((obj, attr, getattr(obj, attr)))
        setattr(obj, attr, value)

    def __enter__(self) -> "Faults":
        try:
            self._install()
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *exc: Any) -> None:
        while self._undo:
            obj, attr, value = self._undo.pop()
            setattr(obj, attr, value)
        if self._tmp is not None:
            self._tmp.cleanup()
            self._tmp = None

    def _install(self) -> None:
        from agent_logic_2.nayka_api import api_nayka, api_price, api_service_info

        # Кэш справочников МИС (/regions, /promotions) — пустой на время сбоя, прежний — после.
        self._set(api_nayka, "_SCHEDULE_REFS_CACHE", {})
        if self.live is not None:
            get = boundary(self.live, self.calls)
            self._set(api_nayka, "_session_get", lambda url, *a, **kw: get(url, *a, **kw))
            for module in (api_price, api_service_info):
                session = getattr(module, "SESSION", None)
                if session is not None:
                    self._set(session, "get", get)
            # Настоящие клиенты справочников — до границы (мир подменял их записью).
            self._set(api_nayka, "site_regions", _original(self.world, api_nayka, "site_regions"))
            self._set(api_nayka, "site_promotions", _original(self.world, api_nayka, "site_promotions"))
            from messengers_router.services import news

            if hasattr(news, "site_promotions"):
                self._set(news, "site_promotions", _original(self.world, news, "site_promotions"))
        if self.snapshots == "stale":
            self._shift_today(api_nayka)
        elif self.snapshots == "none":
            self._empty_folders(api_nayka, api_price, api_service_info)

    def _shift_today(self, api_nayka: Any) -> None:
        import importlib

        world_day = _dt.datetime.strptime(self.world.date, "%Y%m%d").date()
        today = (world_day + _dt.timedelta(days=self.stale_days)).strftime("%Y%m%d")
        for module_name, attr in _DATED_PATHS:
            module = importlib.import_module(module_name)
            real = getattr(getattr(module, attr), "__wrapped__", getattr(module, attr))
            if attr == "price_by_region_path":
                self._set(module, attr, lambda region_id, date=None, _real=real: _real(region_id, date or today))
            else:
                self._set(module, attr, lambda date=None, _real=real: _real(date or today))
        self._set(api_nayka, "get_active_date_str", lambda: today)

    def _empty_folders(self, api_nayka: Any, api_price: Any, api_service_info: Any) -> None:
        from agent_logic_2.nayka_api import cache_paths

        self._tmp = tempfile.TemporaryDirectory(prefix="evalbench-faults-")
        root = Path(self._tmp.name)
        for name in ("doctor_prices", "price_by_region", "price_units", "service_info"):
            (root / name).mkdir()
        self._set(api_price, "DATA_DIR", root)
        self._set(api_price, "DOCTORS_DIR", root)
        self._set(api_price, "PRICES_DIR", root / "doctor_prices")
        self._set(api_price, "PRICE_BY_REGION_DIR", root / "price_by_region")
        self._set(api_price, "PRICE_UNITS_DIR", root / "price_units")
        self._set(api_service_info, "DATA_DIR", root)
        self._set(api_service_info, "SERVICE_INFO_DIR", root / "service_info")
        self._set(api_nayka, "DATA_DIR", root)
        self._set(cache_paths, "resolve_cache_data_dir", lambda: root)

    def written(self) -> list[str]:
        """Что легло в пустые папки за время сбоя (мир не должен писать срезов)."""

        if self._tmp is None:
            return []
        root = Path(self._tmp.name)
        return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())
