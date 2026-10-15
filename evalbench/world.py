"""Замороженный мир стенда: срезы МИС на дату, закрытая сеть, часы, без побочных записей.

Мир — каталог вне репо: копия срезов (прайс, priceUnits, цены врачей, service_info,
врачи, самарские филиалы) на дату D плюс манифест sha256. Стенд перенаправляет
загрузчики МИС в копию, поэтому даже сработавшая очистка старых срезов удалит файлы
копии, а не настоящие (ловушка 03.10 из CLAUDE.md); в конце прогона манифест
сверяется — мир не изменился.

Что заморожено и чем:
- срезы — копия на дату, выбор «последний непустой не позже D», как у загрузчиков;
- часы — профиль `day` (D 12:00 по Самаре) или `night` (D+1 01:30 по Самаре). Часы
  машины считаются UTC, как в контейнере прода;
- сеть — закрыта, кроме явно разрешённых хостов (адрес LLM в режиме real); каждая
  попытка записывается, прогон с попытками недействителен;
- фоновые обновления, прогрев, скачивание врачей, очистка срезов, журнал синонимов,
  лог промптов — выключены или уведены во временный каталог прогона;
- живые источники без копии (расписание, `/regions`, акции, база знаний Meilisearch) —
  из записи `captured/*.json`, если она снята (`python -m evalbench.capture`), иначе
  «недоступны» так же, как при сбое сети. Что именно было — в отпечатке мира.
"""

from __future__ import annotations

import datetime as _dt
import errno
import hashlib
import ipaddress
import json
import re
import shutil
import socket
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

PRIVATE_ROOT = Path.home() / "Dev" / "dla-private" / "bench"
SAMARA = ZoneInfo("Europe/Samara")

# Семейства дневных срезов: подкаталог apidata → имя файла `<префикс>_<ГГГГММДД>.jsonl`.
_DAILY_DIRS = ("price_by_region", "price_units", "doctor_prices", "service_info", ".")
_DATED_RE = re.compile(r"^(?P<prefix>.+)_(?P<date>\d{8})\.jsonl$")
_STATIC_FILES = ("samara_branches_snapshot.json",)

PROFILES = ("day", "night")


# --------------------------------------------------------------------------- сборка


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(path: Path) -> int | None:
    if path.suffix != ".jsonl":
        return None
    with path.open("rb") as fh:
        return sum(1 for line in fh if line.strip())


def select_snapshots(source: Path, date: str) -> list[Path]:
    """Срезы, которые увидит бот в день `date`: по каждому префиксу — последний непустой ≤ D.

    :param source: настоящий каталог apidata
    :param date: дата мира ГГГГММДД
    :return: пути к файлам-источникам
    """

    chosen: list[Path] = []
    for sub in _DAILY_DIRS:
        folder = source / sub
        if not folder.is_dir():
            continue
        best: dict[str, Path] = {}
        for path in folder.glob("*.jsonl"):
            match = _DATED_RE.match(path.name)
            if not match or match["date"] > date or path.stat().st_size == 0:
                continue
            prefix = match["prefix"]
            if prefix not in best or path.name > best[prefix].name:
                best[prefix] = path
        chosen.extend(best[k] for k in sorted(best))
    chosen.extend(source / name for name in _STATIC_FILES if (source / name).is_file())
    return chosen


def build_world(source: Path, date: str, dest: Path | None = None) -> Path:
    """Копирует срезы на дату в каталог мира и пишет манифест `world.json`.

    :param source: настоящий каталог apidata (только читается)
    :param date: дата мира ГГГГММДД
    :param dest: корень мира (по умолчанию `~/Dev/dla-private/bench/worlds/<D>`)
    :return: корень мира
    """

    root = dest or PRIVATE_ROOT / "worlds" / date
    apidata = root / "apidata"
    apidata.mkdir(parents=True, exist_ok=True)
    files: dict[str, Any] = {}
    for src in select_snapshots(source, date):
        rel = src.relative_to(source)
        target = apidata / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        files[str(rel)] = {"sha256": _sha256(target), "bytes": target.stat().st_size, "rows": _rows(target)}
    manifest = {"date": date, "files": files}
    (root / "world.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    return root


def verify_world(root: Path) -> list[str]:
    """Расхождения каталога мира с манифестом: удалённые, изменённые и лишние файлы."""

    manifest = json.loads((root / "world.json").read_text(encoding="utf-8"))
    apidata = root / "apidata"
    problems = []
    for rel, meta in manifest["files"].items():
        path = apidata / rel
        if not path.is_file():
            problems.append(f"удалён: {rel}")
        elif _sha256(path) != meta["sha256"]:
            problems.append(f"изменён: {rel}")
    known = set(manifest["files"])
    for path in apidata.rglob("*"):
        if path.is_file() and str(path.relative_to(apidata)) not in known:
            problems.append(f"лишний: {path.relative_to(apidata)}")
    return problems


# --------------------------------------------------------------------------- срезы «на сегодня»

# Пути к дневным срезам, которые прикрепляются к последнему существующему (гейт и стенд — одно правило).
TODAY_SNAPSHOT_PATHS = (
    ("agent_logic_2.nayka_api.api_service_info", "service_info_path"),
    ("agent_logic_2.nayka_api.api_price", "price_by_region_path"),
    ("agent_logic_2.nayka_api.api_price", "price_units_path"),
    ("agent_logic_2.nayka_api.api_price", "doctor_prices_path"),
)


def latest_if_missing(real: Callable[..., Path]) -> Callable[..., Path]:
    """Путь к срезу «на сегодня», которого нет, — последний непустой не позже; явная дата уважается.

    Утром (Самара) и в копии мира файла «на сегодня» нет — загрузчик пошёл бы качать его у МИС.
    Дата «передана явно», если аргумент `date` есть в вызове (позиционно или по имени).
    """

    import inspect

    from agent_logic_2.nayka_api import snapshots

    signature = inspect.signature(real)

    def path(*args: Any, **kwargs: Any) -> Path:
        target = real(*args, **kwargs)
        try:
            explicit = signature.bind_partial(*args, **kwargs).arguments.get("date") is not None
        except TypeError:
            explicit = False
        if explicit or target.exists():
            return target
        return snapshots.latest_nonempty(target) or target

    path.__wrapped__ = real  # type: ignore[attr-defined]
    return path


def pin_today_snapshots(set_attr: Callable[[Any, str, Any], None]) -> None:
    """Прикрепить все `TODAY_SNAPSHOT_PATHS`; `set_attr(obj, name, value)` — с откатом у вызывающего."""

    import importlib

    for module_name, attr in TODAY_SNAPSHOT_PATHS:
        module = importlib.import_module(module_name)
        set_attr(module, attr, latest_if_missing(getattr(module, attr)))


# --------------------------------------------------------------------------- часы


def frozen_now_utc(date: str, profile: str) -> _dt.datetime:
    """Момент мира в UTC: день — D 12:00 по Самаре, ночь — D+1 01:30 по Самаре."""

    day = _dt.datetime.strptime(date, "%Y%m%d")
    if profile == "day":
        local = day.replace(hour=12, tzinfo=SAMARA)
    elif profile == "night":
        local = (day + _dt.timedelta(days=1)).replace(hour=1, minute=30, tzinfo=SAMARA)
    else:
        raise ValueError(f"профиль мира: {profile!r}, ожидался один из {PROFILES}")
    return local.astimezone(_dt.timezone.utc)


def frozen_clock(now_utc: _dt.datetime) -> tuple[type, type]:
    """Классы `datetime` и `date` с остановленным «сейчас». Часы машины — UTC, как в контейнере."""

    naive_utc = now_utc.astimezone(_dt.timezone.utc).replace(tzinfo=None)

    class FrozenDateTime(_dt.datetime):
        @classmethod
        def now(cls, tz=None):  # noqa: D102 — тот же контракт, что у datetime.now
            return naive_utc if tz is None else now_utc.astimezone(tz)

        @classmethod
        def utcnow(cls):  # noqa: D102
            return naive_utc

        @classmethod
        def today(cls):  # noqa: D102
            return naive_utc

    class FrozenDate(_dt.date):
        @classmethod
        def today(cls):  # noqa: D102
            return naive_utc.date()

    return FrozenDateTime, FrozenDate


# --------------------------------------------------------------------------- сеть


class NetworkBlockedError(ConnectionRefusedError):
    """Стенд попытался выйти в сеть. Для кода бота это обычный отказ соединения."""


@dataclass
class NetGuard:
    """Закрытая сеть: всё, кроме localhost и разрешённых хостов, падает сразу и записывается."""

    allow_hosts: frozenset[str] = frozenset()
    attempts: list[str] = field(default_factory=list)
    _saved: dict[str, Any] = field(default_factory=dict)

    def _allowed(self, host: Any) -> bool:
        if isinstance(host, bytes):
            host = host.decode(errors="replace")
        if host in (None, "", "localhost") or host in self.allow_hosts:
            return True
        try:
            return ipaddress.ip_address(host).is_loopback
        except ValueError:
            return False

    @staticmethod
    def _is_ip(host: Any) -> bool:
        try:
            ipaddress.ip_address(host.decode() if isinstance(host, bytes) else host)
        except (ValueError, TypeError, UnicodeDecodeError):
            return False
        return True

    def install(self) -> None:
        real_connect = socket.socket.connect
        real_connect_ex = socket.socket.connect_ex
        real_getaddrinfo = socket.getaddrinfo
        self._saved = {"connect": real_connect, "connect_ex": real_connect_ex, "getaddrinfo": real_getaddrinfo}
        guard = self

        def blocks(address: Any) -> bool:
            return not isinstance(address, (str, bytes)) and not guard._allowed(address[0])

        def connect(sock, address):
            if blocks(address):
                guard.attempts.append(f"{address[0]}:{address[1]}")
                raise NetworkBlockedError(errno.ECONNREFUSED, f"стенду запрещена сеть: {address[0]}")
            return real_connect(sock, address)

        def connect_ex(sock, address):
            if blocks(address):
                guard.attempts.append(f"{address[0]}:{address[1]}")
                return errno.ECONNREFUSED
            return real_connect_ex(sock, address)

        def getaddrinfo(host, *args, **kwargs):
            if not guard._allowed(host) and not guard._is_ip(host):
                guard.attempts.append(f"{host} (DNS)")
                raise socket.gaierror(socket.EAI_NONAME, f"стенду запрещена сеть: {host}")
            return real_getaddrinfo(host, *args, **kwargs)

        socket.socket.connect = connect
        socket.socket.connect_ex = connect_ex
        socket.getaddrinfo = getaddrinfo
        try:
            from urllib3.util.retry import Retry

            self._saved["retry_sleep"] = Retry.sleep
            Retry.sleep = lambda self_, response=None: None  # отказ мгновенный — паузы повторов не нужны
        except ImportError:  # pragma: no cover
            pass

    def uninstall(self) -> None:
        if not self._saved:
            return
        socket.socket.connect = self._saved["connect"]
        socket.socket.connect_ex = self._saved["connect_ex"]
        socket.getaddrinfo = self._saved["getaddrinfo"]
        if "retry_sleep" in self._saved:
            from urllib3.util.retry import Retry

            Retry.sleep = self._saved["retry_sleep"]
        self._saved = {}


def host_of(url: str) -> str:
    """Хост из URL (для разрешения адреса LLM в режиме real)."""

    return urlparse(url if "//" in url else f"http://{url}").hostname or ""


# --------------------------------------------------------------------------- активация


class _Unavailable(ConnectionError):
    """Источник мира не записан: ведёт себя как сбой сети."""


@dataclass
class WorldFacts:
    """Что увидел бот: попадает в шапку прогона (только числа и ключи)."""

    date: str
    profile: str
    now_utc: str
    snapshots: dict[str, Any]
    captured: list[str]
    unavailable: list[str]


# Сколько стенд ждёт тень ответа на висящий вопрос после хода: потолок тени (`pending_reply._SHADOW_CAP_S`) с запасом.
SHADOW_SETTLE_S = 20.0


async def settle_shadows(timeout_s: float = SHADOW_SETTLE_S) -> None:
    """После замера хода, до чтения его вызовов LLM и телеметрии: дождаться теней, которые ход не ждёт (15.10,
    `pending_reply.settle`), — их решение и вызов LLM попадают в этот ход, а время хода — честное."""

    try:
        from messengers_router import pending_reply
    except ImportError:  # версия бота до клиента `pending_reply`
        return
    settle = getattr(pending_reply, "settle", None)
    if settle is not None:
        await settle(timeout_s)


class FrozenWorld:
    """Контекст-менеджер: на время прогона бот живёт в мире `root` с часами `profile`.

    Входить после того, как путь к дереву кода бота выставлен (`run`), — модули бота
    импортируются здесь.
    """

    def __init__(self, root: Path, profile: str = "day", *, allow_hosts: tuple[str, ...] = ()) -> None:
        self.root = Path(root)
        self.manifest = json.loads((self.root / "world.json").read_text(encoding="utf-8"))
        self.date = self.manifest["date"]
        self.profile = profile
        self.now_utc = frozen_now_utc(self.date, profile)
        self.net = NetGuard(allow_hosts=frozenset(h for h in allow_hosts if h))
        self.cleanup_calls: list[str] = []
        self.side_writes: list[str] = []
        self.telemetry: list[dict[str, Any]] = []
        self.unavailable_calls: list[str] = []
        self._undo: list[tuple[Any, str, Any]] = []
        self._tmp: tempfile.TemporaryDirectory | None = None
        self.captured = self._load_captured()

    # -- служебное

    def _load_captured(self) -> dict[str, Any]:
        out = {}
        folder = self.root / "captured"
        for name in ("regions", "promotions", "kb_main_index"):
            path = folder / f"{name}.json"
            if path.is_file():
                out[name] = json.loads(path.read_text(encoding="utf-8"))
        return out

    def _set(self, obj: Any, attr: str, value: Any) -> None:
        self._undo.append((obj, attr, getattr(obj, attr)))
        setattr(obj, attr, value)

    def _counting_noop(self, name: str) -> Callable[..., None]:
        def noop(*_a, **_kw):
            self.cleanup_calls.append(name)

        return noop

    # -- вход и выход

    def __enter__(self) -> "FrozenWorld":
        try:
            self._activate()
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *exc: Any) -> None:
        while self._undo:
            obj, attr, value = self._undo.pop()
            setattr(obj, attr, value)
        self.net.uninstall()
        if self._tmp is not None:
            self._tmp.cleanup()
            self._tmp = None

    def _activate(self) -> None:
        from agent_logic_1 import meilisearch_client
        from agent_logic_2.nayka_api import api_nayka, api_price, api_service_info, cache_paths, snapshots
        from messengers_router import endpoint, llm_runtime, policies
        from messengers_router.services import Services, _price_select_llm, _samara_branches, news

        apidata = self.root / "apidata"
        self._tmp = tempfile.TemporaryDirectory(prefix="evalbench-")
        tmp = Path(self._tmp.name)

        # Срезы: все загрузчики смотрят в копию мира.
        self._set(api_price, "DATA_DIR", apidata)
        self._set(api_price, "DOCTORS_DIR", apidata)
        self._set(api_price, "PRICES_DIR", apidata / "doctor_prices")
        self._set(api_price, "PRICEALL_DIR", apidata / "prices")
        self._set(api_price, "PRICE_BY_REGION_DIR", apidata / "price_by_region")
        self._set(api_price, "PRICE_UNITS_DIR", apidata / "price_units")
        self._set(api_service_info, "DATA_DIR", apidata)
        self._set(api_service_info, "SERVICE_INFO_DIR", apidata / "service_info")
        self._set(api_nayka, "DATA_DIR", apidata)
        self._set(cache_paths, "resolve_cache_data_dir", lambda: apidata)
        self._set(_samara_branches, "_snapshot_path", lambda: apidata / "samara_branches_snapshot.json")
        snapshots._failed_at.clear()

        # Ничего не удаляем и не пишем в мир: очистка, сохранение врачей и филиалов — пустышки.
        self._set(api_price, "cleanup_old", self._counting_noop("api_price.cleanup_old"))
        self._set(api_service_info, "_cleanup_old", self._counting_noop("api_service_info._cleanup_old"))
        self._set(api_nayka, "cleanup_old_doctors_files", self._counting_noop("api_nayka.cleanup_old_doctors_files"))
        self._set(_samara_branches, "persist_snapshot", lambda *_a, **_kw: self.side_writes.append("samara.persist_snapshot"))

        def no_download(*_a, **_kw):
            raise _Unavailable("стенд: скачивание врачей выключено")

        self._set(api_nayka, "get_all_doctors", no_download)
        pin_today_snapshots(self._set)

        # Живые методы МИС (расписание и его «шапка» /doctors) в мире недоступны: отказ как при
        # сбое сети, но без похода в сеть — счётчик отдельный, прогон он не портит.
        import requests

        def mis_unavailable(url, *_a, **_kw):
            self.unavailable_calls.append(str(url).rsplit("/api/", 1)[-1].split("?")[0])
            raise requests.ConnectionError("стенд: живой метод МИС недоступен")

        self._set(api_nayka, "_session_get", mis_unavailable)
        # Скачивание срезов (прайс, priceUnits, цены врачей, service_info) идёт своими сессиями
        # requests мимо _session_get: в мире его нет — отказ сразу, без сети и без пауз повторов.
        for module in (api_price, api_service_info):
            session = getattr(module, "SESSION", None)
            if session is not None:
                self._set(session, "get", mis_unavailable)

        # Живые источники: запись мира или «недоступно».
        regions = self.captured.get("regions")
        self._set(api_nayka, "site_regions", (lambda *_a, **_kw: regions) if regions else (lambda *_a, **_kw: []))
        promotions = self.captured.get("promotions")

        def promos(*_a, **_kw):
            if promotions is None:
                raise _Unavailable("стенд: акции не записаны")
            return promotions

        self._set(api_nayka, "site_promotions", promos)
        if hasattr(news, "site_promotions"):
            self._set(news, "site_promotions", promos)
        self._install_kb(meilisearch_client)

        # Часы.
        frozen_dt, frozen_date = frozen_clock(self.now_utc)
        for module in (api_price, api_service_info, api_nayka, policies, news):
            if hasattr(module, "datetime"):
                self._set(module, "datetime", frozen_dt)
        self._set(api_nayka, "date", frozen_date)

        # Фон, прогрев, побочные записи.
        self._set(Services, "ensure_background_refresh_started", lambda self_: None)
        self._set(endpoint, "schedule_warm_up", lambda: None)
        journal = tmp / "price_synonym_suggestions.jsonl"
        self._set(_price_select_llm, "_journal_path", lambda: journal)
        self._set(llm_runtime, "_prompt_log_enabled", lambda: False)
        self._journal = journal
        self._capture_turn_telemetry()

        self.net.install()

    def _capture_turn_telemetry(self) -> None:
        """Строки телеметрии хода (если она есть в этой версии бота) — в память прогона, не в журнал."""

        try:
            from messengers_router import turn_telemetry
        except ImportError:  # версия до телеметрии (бейслайн 2b9278d)
            return
        self._set(turn_telemetry, "_emit", self.telemetry.append)

    def take_telemetry(self) -> dict[str, Any] | None:
        """Строка телеметрии последнего хода (или None, если её нет в этой версии) — с решениями, дописанными после
        ответа (`pending_shadow`: `turn_telemetry.merge_side`; звать после `settle_shadows`)."""

        rows, self.telemetry[:] = list(self.telemetry), []
        try:
            from messengers_router.turn_telemetry import merge_side
        except ImportError:  # версия до строк после ответа: строки — только ходы
            merge_side = None
        if merge_side is not None:
            rows = merge_side(rows)
        return rows[-1] if rows else None

    def _install_kb(self, meili: Any) -> None:
        docs = self.captured.get("kb_main_index")

        class _Docs:
            def __init__(self, results: list[dict[str, Any]]) -> None:
                self.results = results

        class _Index:
            def get_documents(self, *_a, **_kw):
                return _Docs(list(docs or []))

        class _Client:
            def index(self, _name):
                return _Index()

        def client():
            if docs is None:
                raise _Unavailable("стенд: база знаний не записана")
            return _Client()

        def search(*_a, **_kw):
            # Поиск Meilisearch офлайн не воспроизводим: путь «справка» идёт как при сбое.
            raise _Unavailable("стенд: поиск базы знаний недоступен")

        self._set(meili, "get_meilisearch_client", client)
        self._set(meili, "ensure_default_indexes_once", lambda: None)
        self._set(meili, "search_meili", search)

    # -- отчёт

    def facts(self) -> WorldFacts:
        snaps = {rel: {"rows": meta.get("rows"), "sha8": meta["sha256"][:8]} for rel, meta in self.manifest["files"].items()}
        unavailable = ["schedule", "kb_search"] + [n for n in ("regions", "promotions", "kb_main_index") if n not in self.captured]
        return WorldFacts(
            date=self.date,
            profile=self.profile,
            now_utc=self.now_utc.isoformat(),
            snapshots=snaps,
            captured=sorted(self.captured),
            unavailable=unavailable,
        )

    def journal_lines(self) -> int:
        """Строк журнала синонимов, которые прогон записал бы в настоящий журнал."""

        path = getattr(self, "_journal", None)
        if path is None or not path.exists():
            return 0
        return sum(1 for _ in path.open(encoding="utf-8"))

    def journal_tail(self, start: int) -> list[str]:
        """Строки журнала синонимов с номера `start` — что дописал один ход диалога (`evalbench.dialogs`)."""

        path = getattr(self, "_journal", None)
        if path is None or not path.exists():
            return []
        return path.read_text(encoding="utf-8").splitlines()[start:]

    def save_journal(self, dest: Path) -> int:
        """Копия журнала синонимов прогона в `dest` (пустой файл, если строк нет); число строк.

        Временный каталог мира удаляется на выходе, а журнал — побочный выход, который смотрят
        глазами: 13.10 «Цена анализа» в журнале не видела ни одна метрика стенда («слепое пятно
        плана», мастер-хэндофф §4 (б)). Вызывать до выхода из мира.
        """

        path = getattr(self, "_journal", None)
        data = path.read_bytes() if path is not None and path.exists() else b""
        dest.write_bytes(data)
        return self.journal_lines()
