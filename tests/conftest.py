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

from pathlib import Path

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
def _hermetic_appointment_change_validator(monkeypatch):
    """LLM-различитель «отменить / перенести» без слова «запись» герметичен: None =
    LLM не ответила, бот ведёт себя как до 04.10 (просьба о записи). Тесты, которым
    важен ответ LLM, переопределяют мок явно.
    """
    import messengers_router.router as _router

    async def _unavailable(_text):
        return None

    monkeypatch.setattr(_router, "is_own_appointment_change", _unavailable)
    yield


@pytest.fixture(autouse=True)
def _hermetic_existing_appointment_validator(monkeypatch):
    """LLM-различитель «своя оформленная запись или новая» герметичен: None = LLM не
    ответила, решение не меняется (как до 08.10). Тесты, которым важен ответ LLM,
    переопределяют мок явно.
    """
    import messengers_router.router as _router

    async def _unavailable(_text):
        return None

    monkeypatch.setattr(_router, "is_own_existing_appointment", _unavailable)
    yield


@pytest.fixture(autouse=True)
def _hermetic_empty_evidence_reply(monkeypatch):
    """Различитель вида реплики на ходу без данных (L-05) герметичен: None = LLM не ответила,
    бот отвечает нейтральной фразой. Тесты, которым важен вид, переопределяют мок явно.
    """
    from messengers_router.services import _empty_evidence_reply as _empty

    async def _unavailable(_text):
        return None

    monkeypatch.setattr(_empty, "classify_empty_evidence_turn", _unavailable)
    yield


@pytest.fixture(autouse=True)
def _hermetic_symptom_specialists(monkeypatch):
    """Совет «к каким врачам с такой жалобой» (LLM) герметичен: None = прежний шаблон
    медвопроса. Тесты совета вызывают сохранённый настоящий `advise` или
    `pick_specialists` напрямую.
    """
    from messengers_router.services import _symptom_specialists as _sym

    async def _no_advice(_text, _services):
        return None

    monkeypatch.setattr(_sym, "advise", _no_advice)
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


@pytest.fixture(autouse=True)
def _fresh_llm_decision_state(monkeypatch):
    """Предохранитель LLM-решений (L-11) и кэш выбора — состояние процесса: тест не наследует их.

    Мир гейта по умолчанию — «LLM недоступна» (`_LLMDown`): три теста подряд, дошедшие до выбора,
    разомкнули бы предохранитель, и следующий тест со своей подделкой LLM получил бы «недоступна»
    вместо её ответа — падение, зависящее от порядка тестов.
    """
    from collections import OrderedDict

    from messengers_router import choice, llm_runtime

    monkeypatch.setattr(llm_runtime, "_BREAKER", llm_runtime._Breaker())
    monkeypatch.setattr(choice, "_CACHE", OrderedDict())
    yield


@pytest.fixture(autouse=True)
def _fresh_mis_failure_state(monkeypatch):
    """«Последний прочитанный срез» (`snapshots._last`) и отметка сбоя `/regions` (`api_nayka._REGIONS_FAILED`) —
    состояние процесса: тест не наследует их.

    Под `mis_honest_degradation = on` (прод с 18.10) оба видны пациенту. Давность прайса в ответе (`services/_freshness`)
    читает последний срез, о котором сообщил загрузчик: тест с подделкой загрузчика (о срезе не сообщает) получал давность
    среза соседа по воркеру — локальный срез прайса 08.10 «старше недели», цены врача уходили в перевод. Отметку сбоя
    филиалов оставлял тест, у которого запрос `/regions` упал в закрытой сети, — адресные ответы соседей уходили в перевод.
    27.10, переворот умолчаний: 5 тестов `test_messenger_services` и 22 адресных хода — красные только под `-n`.
    """
    from agent_logic_2.nayka_api import api_nayka, snapshots

    monkeypatch.setattr(snapshots, "_last", {})
    monkeypatch.setitem(api_nayka._REGIONS_FAILED, "at", 0.0)
    yield


@pytest.fixture(autouse=True)
def _fresh_shown_context():
    """«Показанное прошлым ответом» (`shown_context._PREVIOUS` / `_BEFORE`) — переменные хода: тест, который зовёт
    `begin_turn` напрямую, а не через эндпоинт, не оставляет их следующему (17.10: `test_sync_address_boundaries`
    оставлял «Ферритин», и прямой вызов `test_prepare` другого теста спрашивал «к «Ферритин» или к другому анализу?»).
    """
    from messengers_router import shown_context

    previous, before = shown_context._PREVIOUS.set(()), shown_context._BEFORE.set("")
    yield
    shown_context._PREVIOUS.reset(previous)
    shown_context._BEFORE.reset(before)


@pytest.fixture(autouse=True)
def _no_background_jobs(monkeypatch):
    """Фоновые обновления справочников и прогрев в тестах не запускаются.

    С 03.10 старт сервиса (`TestClient(app)` → `_on_startup`) запускает настоящие
    планировщики обновлений и прогрев. Скачать они ничего не могут — сеть закрыта, —
    но обновление врачей после неудачи чистило «старые» срезы и 04.10 стёрло настоящий
    кэш врачей разработчика: 15 тестов упали на пустом каталоге врачей. Тесты самих
    планировщиков вызывают их напрямую (`api_price.ensure_…`, `warm_up.warm_up`);
    тест старта сервиса возвращает настоящий `schedule_warm_up` явно.
    """
    from messengers_router import endpoint
    from messengers_router.services import Services

    monkeypatch.setattr(Services, "ensure_background_refresh_started", lambda self: None)
    monkeypatch.setattr(endpoint, "schedule_warm_up", lambda: None)
    yield


@pytest.fixture(autouse=True)
def _turn_telemetry_to_tmp(tmp_path, monkeypatch):
    """Строки телеметрии хода тесты пишут во временный каталог, не в `app_data/journals`.

    Иначе каждый прогон гейта дописывал бы тысячи строк в журнал разработчика, а журнал
    прода читается при разборе трафика (ход (а), 09.10).
    """
    from messengers_router import turn_telemetry

    monkeypatch.setattr(turn_telemetry, "_journal_path", lambda: tmp_path / "turns.jsonl")
    yield


@pytest.fixture(autouse=True)
def _one_hanging_question(request, monkeypatch):
    """И-1 на каждом ответе гейта (Д2 звено 2, вариант (А); STEP-2026-10-30-EXPECTS-BY-UNDERSTANDING): под
    `expects_by_understanding = on` в сообщении бота не больше одного висящего вопроса, и `_expects` — ровно он
    (`expects.one_question_violation`). Проверяется конец каждого хода (`expects.finish_turn`) любого теста, который довёл ход
    до ответа: нарушение — ошибка теста. Под off — как было (оффер поверх вопроса бота — известный дефект, его не проверяем).

    Замер на всём гейте: `MR_I1_AUDIT=<файл.jsonl>` пишет строку на каждый ход (тест, режим, вид нарушения), `MR_I1_FORCE_ON=1` —
    гейт целиком под `on` (тесты старого текста при этом падают — смотреть только журнал)."""

    import json
    import os

    from messengers_router import expects, pending_answer
    from messengers_router.runtime_config import config as rcfg

    if os.environ.get("MR_I1_FORCE_ON") == "1":
        monkeypatch.setattr(rcfg, "MR_EXPECTS_BY_UNDERSTANDING", "on", raising=False)
    audit = os.environ.get("MR_I1_AUDIT", "")
    real = expects.finish_turn
    seen: list[str] = []

    def finish_turn(state, response):
        written = real(state, response)
        violation = expects.one_question_violation(response.text, written, response.handoff)
        on = pending_answer.by_understanding_on()
        if violation and on:
            seen.append(f"{violation}: {response.text!r}")
        if audit:
            row = {"test": request.node.nodeid, "on": on, "violation": violation, "offer": expects.OPERATOR_OFFER_QUESTION in (response.text or "")}
            with open(audit, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return written

    monkeypatch.setattr(expects, "finish_turn", finish_turn)
    yield
    if seen and not audit:
        pytest.fail("И-1: в сообщении бота больше одного висящего вопроса — " + "; ".join(seen[:3]))


@pytest.fixture(autouse=True)
def _synonym_journal_to_tmp(tmp_path, monkeypatch):
    """Журнал синонимов цены (`price_synonym_suggestions.jsonl`) тесты пишут во временный каталог.

    До 20.10 тест, дошедший до выбора строки прайса LLM без своей подмены журнала, дописывал строку в
    `<cwd>/app_data/journals` — в журнал разработчика, который уходит клинике как список синонимов. Под
    `pytest -n` туда же писали бы параллельно несколько процессов. Тесты журнала подменяют путь сами
    (monkeypatch после autouse побеждает), мир стенда — тоже.
    """
    from messengers_router.services import _price_select_llm

    monkeypatch.setattr(_price_select_llm, "_journal_path", lambda: tmp_path / "price_synonym_suggestions.jsonl")
    yield


@pytest.fixture(autouse=True)
def _operator_hours_day(monkeypatch):
    """Мир гейта по умолчанию — рабочие часы операторов.

    Эндпоинт дописывает к переводу на оператора часы их работы по настоящим часам машины
    (`endpoint.operator_after_hours_note()`). 10.10 гейт покраснел при неизменном коде после
    19:00 по Самаре в субботу: 10 тестов `test_render_llm_error.py` сравнивают текст перевода.
    Мир стенда (`evalbench.world`) часы замораживает сам; приписку как таковую держит
    `tests/test_operator_hours.py` (функция с явным `now`).
    """

    from messengers_router import endpoint

    monkeypatch.setattr(endpoint, "operator_after_hours_note", lambda *_a, **_kw: "")
    yield


class _LLMDown:
    """Клиент LLM гейта по умолчанию: «LLM недоступна» явно, без похода в сеть.

    До 10.10 тест без своей подделки LLM шёл в настоящий клиент и падал на закрытой сети —
    тот же мир, но неявный и с записью в «попытки выйти в сеть». Свой клиент теста
    (monkeypatch после autouse) по-прежнему побеждает.
    """

    async def generate(self, *_args, **_kwargs):
        raise ConnectionError("гейт: LLM недоступна (мир тестов по умолчанию)")


@pytest.fixture(autouse=True)
def _offline_world(request, monkeypatch):
    """Мир гейта по умолчанию (ход (ж), решение 08.10 «мир — параметр»): МИС офлайн, LLM недоступна.

    - `/regions` — пустое дерево (так отвечает `site_regions` при сбое сети): филиалы — из среза/seed;
    - срез врачей не качается (нет файла «на сегодня» — последний существующий);
    - живые методы МИС (`api_nayka._session_get`: расписание, «шапка» /doctors) — отказ соединения;
    - LLM — `_LLMDown`.
    Тест, которому нужен настоящий загрузчик МИС, помечается `@pytest.mark.real_mis_loader`.
    """
    from agent_logic_2.nayka_api import api_nayka
    from messengers_router import llm_runtime

    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", _LLMDown())
    if request.node.get_closest_marker("real_mis_loader") is None:
        import requests

        def _mis_unavailable(*_a, **_kw):
            raise requests.ConnectionError("гейт: живой метод МИС недоступен (мир тестов по умолчанию)")

        monkeypatch.setattr(api_nayka, "site_regions", lambda *_a, **_kw: [])
        monkeypatch.setattr(api_nayka, "get_all_doctors", lambda: [])
        # Живые методы (расписание и его «шапка» /doctors) — отказ без сети, как в мире стенда.
        monkeypatch.setattr(api_nayka, "_session_get", _mis_unavailable)
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
    прайс; с 10.10 — ещё priceUnits и цены врачей. Правило одно со стендом
    (`evalbench.world.pin_today_snapshots`, решение 08.10 «один модуль мира»).
    Явно переданная дата уважается.
    """
    from evalbench.world import pin_today_snapshots

    def set_attr(module, attr, value):
        _PINNED_SNAPSHOT_LOADERS.append((module, attr, getattr(module, attr)))
        setattr(module, attr, value)

    pin_today_snapshots(set_attr)


_DOCTORS_EXCERPT = Path(__file__).resolve().parent / "fixtures" / "doctors_excerpt" / "doctors_20261008.jsonl"


def doctors_fallback(real_find, gate_dir: Path, excerpt: Path):
    """`api_nayka.find_existing_doctors_file`, который при пустой папке срезов гейта отдаёт выдержку из репо.

    Срез врачей (`doctors_*.jsonl`) не в git: без него ~68 тестов записи и врачей красные (ловушка CLAUDE.md, замер
    18.10), и в CI они не попадают. Выдержка (`tests/fixtures/doctors_excerpt`, решение владельца 18.10: врачи,
    названные в тестах, + до 4 на специальность, описание — первая строка) — только запасной путь: есть локальный срез —
    берётся он; тест подменил папку (мир стенда, временная папка) — выдержка не подставляется.
    """
    from agent_logic_2.nayka_api import api_nayka

    def find():
        found = real_find()
        if found is None and Path(api_nayka.DATA_DIR) == gate_dir:
            return excerpt
        return found

    return find


def _pin_doctors_excerpt_if_no_local_snapshot() -> None:
    from agent_logic_2.nayka_api import api_nayka

    gate_dir = Path(api_nayka.DATA_DIR)
    if any(api_nayka._file_has_doctors(p) for p in gate_dir.glob("doctors_*.jsonl")):
        return
    real = api_nayka.find_existing_doctors_file
    _PINNED_SNAPSHOT_LOADERS.append((api_nayka, "find_existing_doctors_file", real))
    api_nayka.find_existing_doctors_file = doctors_fallback(real, gate_dir, _DOCTORS_EXCERPT)


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "live: тест ходит в живые системы (прод-LLM, МИС) — в гейт не входит; "
        "запуск вручную: pytest -m live",
    )
    config.addinivalue_line(
        "markers",
        "real_mis_loader: тест проверяет настоящий загрузчик МИС (site_regions, get_all_doctors) — "
        "офлайн-подмены мира гейта к нему не применяются; сеть всё равно закрыта",
    )
    socket.socket.connect = _guarded_connect
    socket.socket.connect_ex = _guarded_connect_ex
    socket.getaddrinfo = _guarded_getaddrinfo
    if _Urllib3Retry is not None:
        _Urllib3Retry.sleep = _guarded_retry_sleep
    _pin_latest_mis_snapshots()
    _pin_doctors_excerpt_if_no_local_snapshot()


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


def pytest_sessionfinish(session):
    # Под `pytest -n` тесты идут в процессах-воркерах, а сводку печатает главный процесс: воркер отдаёт
    # свои попытки через `workeroutput` (xdist передаёт его главному при остановке воркера).
    workeroutput = getattr(session.config, "workeroutput", None)
    if workeroutput is not None:
        workeroutput["network_attempts"] = list(_NETWORK_ATTEMPTS)


@pytest.hookimpl(optionalhook=True)
def pytest_testnodedown(node, error):
    """Хук xdist в главном процессе: попытки воркера — в общую сводку прогона."""
    for test, target in getattr(node, "workeroutput", {}).get("network_attempts", ()):
        _NETWORK_ATTEMPTS.append((test, target))


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
