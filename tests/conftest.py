"""Project-wide pytest fixtures.

ГЛАВНОЕ — изоляция last-good снапшота самарских филиалов (BUG-2026-06-11-01).

S1 (`bd4c5af`) добавил `persist_snapshot()` в `core._ensure_regions_loaded`: на каждой
ЗДОРОВОЙ загрузке `/regions` самарский срез пишется на диск в РЕАЛЬНЫЙ кэш-каталог
(`agent_logic_2/nayka_api/apidata/samara_branches_snapshot.json`). В тестах любой кейс,
мокающий `api_nayka.site_regions` здоровым срезом (BUG-A/BUG-E derive-city и т.п.),
запускает настоящий `_ensure_regions_loaded` → пишет реальный снапшот. Этот файл затем
читается через `fallback_branches()` в ДРУГОМ тесте (`test_address_walkin_degraded_guard`)
→ fallback отдаёт 2-3 чужих филиала вместо 31-филиального seed → ложное падение,
зависящее от порядка тестов и состояния диска (несамодостаточный гейт).

Фикс класса: уводим `_snapshot_path()` в per-test tmp. Реальный кэш doctors/prices
(тот же каталог, но ДРУГИЕ файлы) не трогаем — изолируем ТОЛЬКО снапшот.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_samara_snapshot(tmp_path, monkeypatch):
    """Снапшот самарских филиалов пишется/читается в per-test tmp, не в реальный apidata.

    Делает тест-гейт самодостаточным: persist одного теста не может заразить fallback
    другого. Покрывает ВЕСЬ класс (любой писатель через `_ensure_regions_loaded` и любой
    читатель через `fallback_branches`/`resolve_samara_branches`), а не один кейс.
    """
    from messengers_router.services import _samara_branches as _sb

    snap = tmp_path / "samara_branches_snapshot.json"
    monkeypatch.setattr(_sb, "_snapshot_path", lambda: snap)
    yield


@pytest.fixture(autouse=True)
def _isolate_schedule_refs_cache():
    """TTL-кэш «шапки» find_doctor_schedule (справочники CRM) — чистим между тестами.

    Кэш module-level и живёт процесс: без очистки справочник, закэшированный
    фейком одного теста, дожил бы до чужого теста (тот мокает _session_get и
    ждёт СВОЙ /doctors) → порядкозависимые ложные падения. Тот же класс проблемы,
    что _isolate_samara_snapshot выше.
    """
    from agent_logic_2.nayka_api import api_nayka

    api_nayka._SCHEDULE_REFS_CACHE.clear()
    yield
    api_nayka._SCHEDULE_REFS_CACHE.clear()


@pytest.fixture(autouse=True)
def _hermetic_patient_name_validator(monkeypatch):
    """LLM-валидатор ФИО (#3) герметичен по умолчанию: fail-open = прежнее
    поведение записи (принять по форме). Иначе полный router-флоу записи в
    116 appointment-тестах дёргал бы реальный generate_text (флакость/задержка).
    Тесты самого валидатора импортируют модуль напрямую; интеграционные —
    переопределяют этот мок явно (monkeypatch после autouse побеждает).
    """
    import messengers_router.router as _router

    async def _accept(_text):
        return True

    monkeypatch.setattr(_router, "is_patient_name_reply", _accept)
    yield


@pytest.fixture(autouse=True)
def _hermetic_service_slot_validator(monkeypatch):
    """LLM-валидатор слота услуги (#4) герметичен по умолчанию: fail-open = прежнее
    поведение записи (принять по форме). Иначе appointment-тесты дёргали бы реальный
    generate_text. Тесты самого валидатора импортируют модуль напрямую;
    интеграционные — переопределяют мок явно (monkeypatch после autouse побеждает).
    """
    import messengers_router.router as _router

    async def _accept(_text):
        return True

    monkeypatch.setattr(_router, "is_service_name_reply", _accept)
    yield


@pytest.fixture(autouse=True)
def _hermetic_result_timing_validator(monkeypatch):
    """LLM-различитель сроков результата (#2) герметичен: fail-safe = lookup
    (прежнее поведение TEST_RESULT). Тесты валидатора импортируют модуль
    напрямую; интеграционные переопределяют мок явно.
    """
    from messengers_router.services import lab_tests as _lab

    async def _lookup(_text):
        return False

    monkeypatch.setattr(_lab, "is_result_timing_question", _lookup)
    yield


@pytest.fixture(autouse=True)
def _hermetic_mis_synonyms(monkeypatch):
    """Герметичны ТОЛЬКО синонимы МИС; биоматериалы остаются живыми.

    Синонимы заполняет КЛИНИКА, и дневной срез меняется без единой правки кода.
    15.09.2026 гейт покраснел именно так: администратор заполнила 295 услуг, и
    три теста, читавшие живой срез, поменяли поведение при неизменном коде.
    Гейт обязан быть воспроизводимым, поэтому синонимы по умолчанию пусты, а
    тесты про них подставляют свой словарь явно (monkeypatch после autouse
    побеждает) либо передают `vocab=` параметром.

    `biomatNames` не трогаем: поле заполнено у 98% услуг давно и стабильно, на
    нём построен слой снятия биоматериала, и подмена сломала бы его тесты.
    """
    from messengers_router.services import _biomaterial as _bio

    real = _bio._vocabularies

    def _without_synonyms() -> "_bio._MisVocabularies":
        vocab = real()
        vocab.synonym_to_service = {}
        vocab.synonym_candidates = {}
        return vocab

    monkeypatch.setattr(_bio, "_vocabularies", _without_synonyms)
    yield


# --- Изоляция сети (29.09) ----------------------------------------------------
#
# Гейт обязан быть герметичным. Через локальный `config.ini` тесты ходили в
# прод-LLM (ollama на хосте прода) и в МИС клиники: при плохой сети прогон шёл
# больше часа и всё это время грузил LLM, которая отвечает пациентам.
#
# Теперь любое соединение не на localhost падает СРАЗУ, как при реальном отказе
# сети (исключение — наследник ConnectionError, поэтому код идёт по своему
# штатному пути деградации). Каждая попытка записывается и видна в сводке прогона.
#
# Тест, которому живые системы нужны по смыслу, помечается @pytest.mark.live.
# В гейт такие тесты не входят (addopts в pyproject.toml); запуск вручную:
# `pytest -m live`.
#
# Защита ставится на весь прогон в pytest_configure, а не на каждый тест: иначе
# фоновые потоки бота выходили бы в сеть в промежутках между тестами.

import errno
import ipaddress
import socket

_REAL_CONNECT = socket.socket.connect
_REAL_CONNECT_EX = socket.socket.connect_ex
_REAL_GETADDRINFO = socket.getaddrinfo

try:
    from urllib3.util.retry import Retry as _Urllib3Retry

    _REAL_RETRY_SLEEP = _Urllib3Retry.sleep
except ImportError:  # pragma: no cover — urllib3 приходит с requests
    _Urllib3Retry = None

_NETWORK = {"blocked": True, "test": "<сбор тестов>"}
_NETWORK_ATTEMPTS: list[tuple[str, str]] = []


class NetworkBlockedError(ConnectionRefusedError):
    """Тест попытался выйти в сеть. Для кода это обычный отказ соединения."""


def _is_local_host(host) -> bool:
    if isinstance(host, bytes):
        host = host.decode(errors="replace")
    if host in (None, "", "localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _is_ip_literal(host) -> bool:
    if isinstance(host, bytes):
        host = host.decode(errors="replace")
    try:
        ipaddress.ip_address(host)
    except (ValueError, TypeError):
        return False
    return True


def _blocks(address) -> bool:
    if not _NETWORK["blocked"] or isinstance(address, (str, bytes)):  # AF_UNIX — локально
        return False
    return not _is_local_host(address[0])


def _record(target: str) -> None:
    _NETWORK_ATTEMPTS.append((_NETWORK["test"], target))


def _guarded_connect(sock, address):
    if _blocks(address):
        target = f"{address[0]}:{address[1]}"
        _record(target)
        raise NetworkBlockedError(errno.ECONNREFUSED, f"тестам запрещена сеть: {target}")
    return _REAL_CONNECT(sock, address)


def _guarded_connect_ex(sock, address):
    if _blocks(address):
        _record(f"{address[0]}:{address[1]}")
        return errno.ECONNREFUSED
    return _REAL_CONNECT_EX(sock, address)


def _guarded_getaddrinfo(host, *args, **kwargs):
    # DNS — тоже сеть. IP-литералы разрешаются без неё и пропускаются: их
    # остановит connect.
    if _NETWORK["blocked"] and not _is_local_host(host) and not _is_ip_literal(host):
        name = host.decode(errors="replace") if isinstance(host, bytes) else str(host)
        _record(f"{name} (DNS)")
        raise socket.gaierror(socket.EAI_NONAME, f"тестам запрещена сеть: {name}")
    return _REAL_GETADDRINFO(host, *args, **kwargs)


def _guarded_retry_sleep(self, response=None):
    # Пока сеть закрыта, повторы urllib3 не ждут: попытка всё равно упадёт сразу,
    # а пауза-backoff, умноженная на число запросов, превращала прогон в часы —
    # обновление цен шло отдельным запросом на каждого врача (29.09).
    if _NETWORK["blocked"]:
        return None
    return _REAL_RETRY_SLEEP(self, response)


_PINNED_SNAPSHOT_LOADERS: list[tuple[object, str, object]] = []


def _pin_latest_mis_snapshots() -> None:
    """Гейт не скачивает справочники МИС: берёт последний существующий срез.

    Срез датирован по Самаре, и утром файла «на сегодня» ещё нет — загрузчик идёт
    его качать. С закрытой сетью это роняло СБОР тестов (30.09:
    test_biomaterial_and_synonyms читает service_info при импорте). CLAUDE.md
    велит для любого офлайн-замера прикреплять ОБА среза — служебный справочник и
    прайс. Явно переданная дата уважается: подмена только для «сегодня».
    """
    from agent_logic_2.nayka_api import api_price, api_service_info

    def latest_if_missing(real_path_fn, pattern_of):
        def path(*args, **kwargs):
            target = real_path_fn(*args, **kwargs)
            # Подменяем только файл НА СЕГОДНЯ: явная дата (позиционная или
            # именованная) даёт другое имя файла и уважается как есть.
            if target.exists() or api_service_info.today_str() not in target.name:
                return target
            snapshots = sorted(target.parent.glob(pattern_of(*args)))
            return snapshots[-1] if snapshots else target

        return path

    for module, attr, pattern_of in (
        (api_service_info, "service_info_path", lambda *a: "service_info_*.jsonl"),
        (api_price, "price_by_region_path", lambda region_id, *a: f"price_region_{str(region_id).strip()}_*.jsonl"),
    ):
        real = getattr(module, attr)
        _PINNED_SNAPSHOT_LOADERS.append((module, attr, real))
        setattr(module, attr, latest_if_missing(real, pattern_of))


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "live: тест ходит в живые системы (прод-LLM, МИС) — в гейт не входит; "
        "запуск вручную: pytest -m live",
    )
    socket.socket.connect = _guarded_connect
    socket.socket.connect_ex = _guarded_connect_ex
    socket.getaddrinfo = _guarded_getaddrinfo
    if _Urllib3Retry is not None:
        _Urllib3Retry.sleep = _guarded_retry_sleep
    _pin_latest_mis_snapshots()


def pytest_unconfigure(config):
    socket.socket.connect = _REAL_CONNECT
    socket.socket.connect_ex = _REAL_CONNECT_EX
    socket.getaddrinfo = _REAL_GETADDRINFO
    if _Urllib3Retry is not None:
        _Urllib3Retry.sleep = _REAL_RETRY_SLEEP
    for module, attr, real in _PINNED_SNAPSHOT_LOADERS:
        setattr(module, attr, real)
    _PINNED_SNAPSHOT_LOADERS.clear()


@pytest.fixture(autouse=True)
def _network_per_test(request):
    """Помечает, какой тест идёт, и снимает запрет только для @pytest.mark.live."""
    _NETWORK["test"] = request.node.nodeid
    _NETWORK["blocked"] = request.node.get_closest_marker("live") is None
    yield
    _NETWORK["blocked"] = True
    _NETWORK["test"] = "<между тестами>"


def pytest_terminal_summary(terminalreporter):
    if not _NETWORK_ATTEMPTS:
        return
    by_target: dict[str, set[str]] = {}
    for test, target in _NETWORK_ATTEMPTS:
        by_target.setdefault(target, set()).add(test)
    terminalreporter.section("попытки выйти в сеть (заблокированы)")
    for target, tests in sorted(by_target.items(), key=lambda kv: -len(kv[1])):
        terminalreporter.write_line(f"{target}: {len(tests)} тест(ов)")
        for test in sorted(tests)[:5]:
            terminalreporter.write_line(f"    {test}")
