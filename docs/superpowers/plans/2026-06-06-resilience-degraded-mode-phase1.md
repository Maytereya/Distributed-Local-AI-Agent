# Resilience degraded-mode (Фаза 1) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Честно различать тех-сбой апстрима и бизнес-пусто; при тех-сбое отвечать «по техническим причинам…» (best-effort кэш иначе), логировать degraded-события структурно, и fail-fast на realtime-вызовах.

**Architecture:** Подход C из спека — новый изолированный модуль `messengers_router/resilience.py` (классификатор исхода + degraded-логгер + константы), доменные сервисы вызывают его и решают cache-fallback vs честный tech-ответ; сообщения через существующий `HANDOFF_REASON_MATRIX`; realtime-профиль ретраев в `api_nayka`.

**Tech Stack:** Python 3.13, asyncio, requests (+urllib3 Retry), pytest, ruff. Запуск тестов: `./venv/bin/python -m pytest`. Линт: `ruff check`.

**Спек:** `docs/superpowers/specs/2026-06-06-resilience-degraded-mode-design.md`

**Конвенции проекта (ОБЯЗАТЕЛЬНО):** атомарные коммиты; класс-инвариант-тесты (параметризованные, не один инстанс); ruff+pytest гейт после каждой задачи (полный suite перед коммитом fix-задачи); прод NLU-движок = `legacy_v2`. Co-Authored-By строка в коммитах: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

**✅ OWNER-CHECKPOINTS — ВСЕ ПОДТВЕРЖДЕНЫ владельцем (2026-06-06):**
- **OC-1 (Task 4): ДА.** Для РЕЗУЛЬТАТОВ при 5xx/таймаут — честный `tech_unavailable` БЕЗ авто-эскалации на оператора (пользователь сам пишет «оператор»). Прежний `service_error_results`→авто-оператор убираем для этого случая.
- **OC-2 (Task 8): ДА.** realtime-профиль: `Retry(total=0)` + read-timeout **8с**; background (прогрев кэша) — без изменений (`total=3`).
- **OC-3: ДА.** Формулировка `tech_unavailable_text`: «По техническим причинам сейчас не удаётся загрузить {что}. Пожалуйста, попробуйте позже или напишите «оператор».» (как в плане).

---

## PROGRESS (обновлять при исполнении — durable resume-точка)

- [x] **Task 1** resilience.py классификатор — `885883b`
- [x] **Task 2** resilience.py log_degraded/mark_degraded/tech_unavailable_text — `74da638`
- [x] **Task 3** HANDOFF_REASON_MATRIX `tech_unavailable` — `e8295f1`
- [x] **Task 4** результаты 5xx/таймаут/exc → tech_unavailable (OC-1) — `d58cb5f` (full suite 954 passed)
- [x] **Task 5** — ПОКРЫТ Task 4: `build_test_result_response` отдаёт `result_preview` как текст при `ready=False` без `missing_fields` (response_builder.py:304-306). Отдельной работы для результатов не нужно.
- [x] **Task 6** addresses: `_ensure_regions_loaded` сигнал тех-сбоя + best-effort/degraded — `2070a38` (full suite 957 passed, 1 xfailed)
- [x] **Task 7** schedule tech-failure: структурный degraded-лог + `mark_degraded` (РЕШЕНИЕ ВЛАДЕЛЬЦА Option 1 — handoff сохранён, response_builder НЕ тронут) — `38292bf` (full suite 959 passed, 1 xfailed)
- [x] **Task 8** api_nayka realtime fail-fast профиль (OC-2: realtime Retry total=0 + read 8с) — `2a34452` (core + 4 теста) + review-polish (2 комментария + schedule/cells realtime-тест) попал в `8a14124` (full suite 964 passed, 1 xfailed). **ФАЗА 1 ЗАВЕРШЕНА.**
  - _Примечание (git): `8a14124` под message `style(nayka): import lint` фактически содержит И lint-чистку владельца, И мой Task-8 review-polish — мой `--amend` совпал с параллельным style-коммитом владельца (между ними легли `2bf0a92` docs + lint-чип). Решение владельца: оставить как есть (ветка не запушена, сквош при мерже)._
- [x] **Task 8.1** (из финального ревью фазы, Important-находка): полный fail-fast расписания — 4 ведущих вызова в `find_doctor_schedule` (`site_regions`/`/doctors`/`/doctorCompanyUnits`/`/doctorRegions`) помечены `realtime=True` → закрыт пробел Goal 4 на хотспоте «doctorSchedule тормозит» (cold/degraded окно). `2479aa2` (full suite 964 passed, 1 xfailed).

**Resume:** **ФАЗА 1 ЗАВЕРШЕНА (Tasks 1-8 + 8.1).** Resilience-коммиты: `885883b` `74da638` `e8295f1` `d58cb5f` (1-4), `2070a38` (6), `38292bf` (7), `2a34452`+`8a14124` (8), `2479aa2` (8.1). resilience.py API: `R.OK/NOT_FOUND/TECH_UNAVAILABLE`, `classify_api_response`, `failure_mode_from_response`, `log_degraded`, `mark_degraded`, `tech_unavailable_text(what)`. В сервисах импорт: `from .. import resilience as _R`.

---

## File Structure

- **NEW** `messengers_router/resilience.py` — изолированный: outcome-константы, `classify_api_response`, `failure_mode_from_response`, `log_degraded`, `mark_degraded`, `tech_unavailable_text`. Без сети, без доменных зависимостей.
- `messengers_router/policies.py` — `HANDOFF_REASON_MATRIX` (+`tech_unavailable`).
- `messengers_router/services/lab_tests.py` — `test_result_status`: 5xx/таймаут/исключение → `tech_unavailable`.
- `messengers_router/services/core.py` — `_ensure_regions_loaded`: прокинуть флаг тех-сбоя (сейчас теряется).
- `messengers_router/services/addresses.py` — `address_info`: TECH `/regions` → best-effort doctors-cache + degraded; иначе NOT_FOUND.
- `messengers_router/services/doctors.py` — `doctors_schedule_week`: schedule-API тех-сбой → `tech_unavailable` vs not-found-doctor.
- `messengers_router/response_builder.py` — payload с `tech_unavailable` → честное сообщение.
- `agent_logic_2/nayka_api/api_nayka.py` — realtime fail-fast session/профиль (OC-2).
- Тесты: `tests/test_resilience.py` (NEW), дополнения в `tests/test_messenger_services.py`.

Порядок задач = порядок исполнения. Каждая задача — самодостаточный атомарный коммит с зелёным гейтом.

---

## Task 1: `resilience.py` — классификатор исхода (изолированный)

**Files:**
- Create: `messengers_router/resilience.py`
- Test: `tests/test_resilience.py`

- [ ] **Step 1: Написать падающий тест**

```python
# tests/test_resilience.py
import pytest
from messengers_router import resilience as R


@pytest.mark.parametrize("resp,exc,expected", [
    ({"ok": True, "data": "%PDF..."}, None, R.OK),
    ({"ok": True, "data": [1, 2]}, None, R.OK),
    ({"ok": True, "data": None}, None, R.NOT_FOUND),
    ({"ok": True, "data": []}, None, R.NOT_FOUND),
    ({"ok": False, "status_code": 404}, None, R.NOT_FOUND),
    ({"ok": False, "status_code": 500}, None, R.TECH_UNAVAILABLE),
    ({"ok": False, "status_code": 503}, None, R.TECH_UNAVAILABLE),
    ({"ok": False, "status_code": None}, None, R.TECH_UNAVAILABLE),   # таймаут/conn
    ({"ok": False, "status_code": 400}, None, R.TECH_UNAVAILABLE),    # прочие 4xx → tech (безопасно)
    (None, None, R.TECH_UNAVAILABLE),
    ({"ok": True, "data": "x"}, ValueError("boom"), R.TECH_UNAVAILABLE),  # exc приоритетнее
])
def test_classify_api_response_class_invariant(resp, exc, expected):
    assert R.classify_api_response(resp, exc=exc) == expected
```

- [ ] **Step 2: Запустить — убедиться, что падает**

Run: `./venv/bin/python -m pytest tests/test_resilience.py -q -p no:cacheprovider`
Expected: FAIL (`ModuleNotFoundError: messengers_router.resilience`).

- [ ] **Step 3: Создать модуль с минимальной реализацией**

```python
# messengers_router/resilience.py
"""Слой устойчивости: классификация исхода апстрим-вызова + degraded-логи.

Изолированный модуль: без сетевых вызовов и доменных зависимостей, чтобы
тестироваться независимо. См. docs/superpowers/specs/2026-06-06-resilience-degraded-mode-design.md
"""
from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

# --- Классы исхода апстрим-вызова ---
OK = "ok"
NOT_FOUND = "not_found"          # дозвонились, но по данным пусто (404 / ok+empty)
TECH_UNAVAILABLE = "tech_unavailable"  # не дозвонились/не распарсили (timeout/5xx/conn/exc)

# --- failure_mode для логов ---
FM_TIMEOUT = "timeout"
FM_CONN_ERROR = "conn_error"
FM_HTTP_5XX = "http_5xx"
FM_HTTP_404 = "http_404"
FM_EMPTY_DATA = "empty_data"
FM_EXCEPTION = "exception"


def classify_api_response(resp: Any, *, exc: Exception | None = None) -> str:
    """Классифицирует исход вызова api_nayka.

    :param resp: dict от api_nayka ({ok,status_code,error,data}) или None
    :param exc: пойманное исключение при вызове (если было)
    :return: OK | NOT_FOUND | TECH_UNAVAILABLE
    """
    if exc is not None:
        return TECH_UNAVAILABLE
    if not isinstance(resp, dict):
        return TECH_UNAVAILABLE
    if resp.get("ok"):
        return OK if resp.get("data") else NOT_FOUND
    sc = resp.get("status_code")
    if sc == 404:
        return NOT_FOUND
    if sc is None:
        return TECH_UNAVAILABLE
    if isinstance(sc, int) and 500 <= sc <= 599:
        return TECH_UNAVAILABLE
    return TECH_UNAVAILABLE
```

- [ ] **Step 4: Запустить — убедиться, что проходит**

Run: `./venv/bin/python -m pytest tests/test_resilience.py -q -p no:cacheprovider`
Expected: PASS (11 параметров).

- [ ] **Step 5: Коммит**

```bash
git add messengers_router/resilience.py tests/test_resilience.py
git commit -m "feat(resilience): upstream outcome classifier (OK/NOT_FOUND/TECH_UNAVAILABLE)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: `resilience.py` — failure_mode, log_degraded, mark_degraded, tech_unavailable_text

**Files:**
- Modify: `messengers_router/resilience.py`
- Test: `tests/test_resilience.py`

- [ ] **Step 1: Добавить падающие тесты**

```python
# tests/test_resilience.py (добавить)
import logging


@pytest.mark.parametrize("resp,exc,expected", [
    ({"ok": False, "status_code": 404}, None, R.FM_HTTP_404),
    ({"ok": False, "status_code": 500}, None, R.FM_HTTP_5XX),
    ({"ok": False, "status_code": None}, None, R.FM_TIMEOUT),
    ({"ok": False, "status_code": 400}, None, R.FM_CONN_ERROR),
    (None, ValueError("x"), R.FM_EXCEPTION),
    ({"ok": True, "data": []}, None, R.FM_EMPTY_DATA),
])
def test_failure_mode_from_response(resp, exc, expected):
    assert R.failure_mode_from_response(resp, exc=exc) == expected


def test_log_degraded_emits_structured_line(caplog):
    with caplog.at_level(logging.WARNING):
        R.log_degraded(upstream="result_for_patient", failure_mode=R.FM_TIMEOUT,
                       latency_ms=1234, session_id="s1", fallback_used=False)
    msg = caplog.text
    assert "degraded_upstream" in msg
    assert "upstream=result_for_patient" in msg
    assert "failure_mode=timeout" in msg
    assert "fallback_used=False" in msg


def test_mark_degraded_sets_payload_fields():
    p = {"data": "x"}
    R.mark_degraded(p, upstream="regions", failure_mode=R.FM_TIMEOUT, fallback_used=True)
    assert p["degraded"] is True
    assert p["degraded_upstream"] == "regions"
    assert p["degraded_mode"] == "timeout"
    assert p["degraded_fallback_used"] is True


def test_tech_unavailable_text_substitutes_domain():
    t = R.tech_unavailable_text("результаты анализов")
    assert "результаты анализов" in t
    assert "техническ" in t.lower()
    # дефолт
    assert "информацию" in R.tech_unavailable_text()
```

- [ ] **Step 2: Запустить — убедиться, что падает**

Run: `./venv/bin/python -m pytest tests/test_resilience.py -q -p no:cacheprovider`
Expected: FAIL (нет `failure_mode_from_response`/`log_degraded`/...).

- [ ] **Step 3: Дописать модуль**

```python
# messengers_router/resilience.py (добавить в конец)

def failure_mode_from_response(resp: Any, *, exc: Exception | None = None) -> str:
    """Детализирует режим сбоя для логов/метрик."""
    if exc is not None:
        return FM_EXCEPTION
    if not isinstance(resp, dict):
        return FM_EXCEPTION
    if resp.get("ok"):
        return FM_EMPTY_DATA
    sc = resp.get("status_code")
    if sc == 404:
        return FM_HTTP_404
    if sc is None:
        return FM_TIMEOUT
    if isinstance(sc, int) and 500 <= sc <= 599:
        return FM_HTTP_5XX
    return FM_CONN_ERROR


def log_degraded(
    *,
    upstream: str,
    failure_mode: str,
    latency_ms: int | None = None,
    session_id: str | None = None,
    fallback_used: bool = False,
) -> None:
    """Одна структурная строка на degraded-событие (Datadog-ready схема)."""
    log.warning(
        "degraded_upstream upstream=%s failure_mode=%s latency_ms=%s session_id=%s fallback_used=%s",
        upstream, failure_mode, latency_ms, session_id, fallback_used,
    )


def mark_degraded(payload: dict, *, upstream: str, failure_mode: str, fallback_used: bool) -> dict:
    """Помечает payload как degraded (для логов и Фаза-3 сигнала оператору)."""
    payload["degraded"] = True
    payload["degraded_upstream"] = upstream
    payload["degraded_mode"] = failure_mode
    payload["degraded_fallback_used"] = fallback_used
    return payload


def tech_unavailable_text(what: str = "информацию") -> str:
    """Честное сообщение при тех-сбое апстрима (OC-3: финал утверждает владелец)."""
    return (
        f"По техническим причинам сейчас не удаётся загрузить {what}. "
        "Пожалуйста, попробуйте позже или напишите «оператор»."
    )
```

- [ ] **Step 4: Запустить — PASS**

Run: `./venv/bin/python -m pytest tests/test_resilience.py -q -p no:cacheprovider`
Expected: PASS (все).

- [ ] **Step 5: ruff + коммит**

```bash
ruff check messengers_router/resilience.py tests/test_resilience.py
git add messengers_router/resilience.py tests/test_resilience.py
git commit -m "feat(resilience): failure_mode, log_degraded, mark_degraded, tech_unavailable_text

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: `HANDOFF_REASON_MATRIX` — причина `tech_unavailable`

**Files:**
- Modify: `messengers_router/policies.py` (`HANDOFF_REASON_MATRIX`, ~строка 539)
- Test: `tests/test_messenger_services.py` (тест матрицы причин, ~строка 357)

- [ ] **Step 1: Прочитать текущую матрицу и её тест**

Run: `grep -n "HANDOFF_REASON_MATRIX\|tech_unavailable\|handoff_message" messengers_router/policies.py | head`
Run: `grep -n "ambiguous_price_service\|city_not_supported\|HANDOFF_REASON" tests/test_messenger_services.py | head`
Зафиксировать формат, чтобы Edit совпал дословно.

- [ ] **Step 2: Добавить падающий тест**

```python
# tests/test_messenger_services.py (рядом с существующим matrix-тестом)
def test_handoff_message_tech_unavailable():
    from messengers_router.policies import handoff_message
    msg = handoff_message("tech_unavailable")
    assert "техническ" in msg.lower()
    # override-подстановка домена работает
    from messengers_router.resilience import tech_unavailable_text
    over = handoff_message("tech_unavailable", override=tech_unavailable_text("результаты анализов"))
    assert "результаты анализов" in over
```

- [ ] **Step 3: Запустить — FAIL**

Run: `./venv/bin/python -m pytest tests/test_messenger_services.py::test_handoff_message_tech_unavailable -q -p no:cacheprovider`
Expected: FAIL (reason `tech_unavailable` нет → вернётся generic/`Передаю диалог оператору.`).

- [ ] **Step 4: Добавить причину в матрицу**

В `messengers_router/policies.py`, в словарь `HANDOFF_REASON_MATRIX`, добавить запись (после `"generic": ...` или рядом):

```python
    "tech_unavailable": (
        "По техническим причинам сейчас не удаётся загрузить информацию. "
        "Пожалуйста, попробуйте позже или напишите «оператор»."
    ),
```

- [ ] **Step 5: Запустить — PASS**

Run: `./venv/bin/python -m pytest tests/test_messenger_services.py::test_handoff_message_tech_unavailable -q -p no:cacheprovider`
Expected: PASS.

- [ ] **Step 6: ruff + коммит**

```bash
ruff check messengers_router/policies.py
git add messengers_router/policies.py tests/test_messenger_services.py
git commit -m "feat(handoff): add tech_unavailable reason to HANDOFF_REASON_MATRIX

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: `test_result_status` — 5xx/таймаут/исключение → tech_unavailable ⚠️ OC-1

**Files:**
- Modify: `messengers_router/services/lab_tests.py` (`test_result_status`, ~строки 259-294)
- Test: `tests/test_messenger_services.py` (`test_result_404_is_not_ready_not_operator`, ~строка 4397)

**OC-1: перед коммитом подтвердить у владельца** — для результатов 5xx/таймаут заменить авто-оператор на честный `tech_unavailable` без авто-handoff.

- [ ] **Step 1: Прочитать текущий `test_result_status` (259-294)** и существующий тест 404 (4397+), чтобы Edit совпал дословно.

Run: `sed -n '259,295p' messengers_router/services/lab_tests.py`
Run: `sed -n '4397,4435p' tests/test_messenger_services.py`

- [ ] **Step 2: Обновить тест (класс-инвариант)** — 5xx/None/исключение теперь → tech_unavailable (НЕ auto-handoff); 404/empty → честный not-found (как сейчас).

```python
# заменить тело параметризации/ассертов test_result_404_is_not_ready_not_operator на:
@pytest.mark.parametrize("status_code", [404, 500, 503, None])
def test_result_failure_taxonomy(monkeypatch, status_code):
    # BUG/спек resilience: 404 → честное «не нашёл» (NOT_FOUND, без оператора);
    # 5xx/None → честный tech_unavailable (TECH, без авто-оператора, текст про техпричины).
    def fake_site_result(**kwargs):
        return {"ok": False, "status_code": status_code, "error": f"{status_code} err", "params": kwargs}
    monkeypatch.setattr(lab_tests_mod.api_nayka, "site_result_for_patient", fake_site_result)
    entities = {"surname": "Тестов", "year": "1990", "filial": "Бг", "number": "1"}
    res = asyncio.run(lab_tests_mod.test_result_status(None, "Тестов, 1990, Бг, 1", entities))
    assert res.get("ready") is False, status_code
    assert not res.get("handoff_required"), status_code  # авто-эскалации НЕТ ни при 404, ни при tech
    preview = str(res.get("result_preview") or "").lower()
    if status_code == 404:
        assert "не нашёл" in preview, status_code
        assert "техническ" not in preview, status_code
    else:
        assert "техническ" in preview, status_code  # tech_unavailable
        assert "результат пока не готов" not in preview, status_code


def test_result_exception_is_tech_unavailable(monkeypatch):
    def boom(**kwargs):
        raise ConnectionError("down")
    monkeypatch.setattr(lab_tests_mod.api_nayka, "site_result_for_patient", boom)
    res = asyncio.run(lab_tests_mod.test_result_status(None, "Тестов, 1990, Бг, 1",
                      {"surname": "Тестов", "year": "1990", "filial": "Бг", "number": "1"}))
    assert res.get("ready") is False
    assert "техническ" in str(res.get("result_preview") or "").lower()
```

- [ ] **Step 3: Запустить — FAIL**

Run: `./venv/bin/python -m pytest tests/test_messenger_services.py::test_result_failure_taxonomy tests/test_messenger_services.py::test_result_exception_is_tech_unavailable -q -p no:cacheprovider`
Expected: FAIL (5xx сейчас → оператор; исключение не ловится как tech).

- [ ] **Step 4: Реализация в `test_result_status`**

Импорт вверху файла: `from messengers_router import resilience as R`.
Обернуть вызов `api_nayka.site_result_for_patient` в try/except и классифицировать:

```python
    try:
        api_resp = await asyncio.to_thread(
            api_nayka.site_result_for_patient,
            surname=fields["surname"], year=int(fields["year"]),
            filial=fields["filial"], number=int(fields["number"]),
            lang=fields["lang"], with_time=None,
        )
        outcome = R.classify_api_response(api_resp)
        fmode = R.failure_mode_from_response(api_resp)
    except Exception as exc:
        api_resp, outcome, fmode = None, R.TECH_UNAVAILABLE, R.FM_EXCEPTION

    if outcome == R.TECH_UNAVAILABLE:
        R.log_degraded(upstream="result_for_patient", failure_mode=fmode,
                       session_id=str((entities or {}).get("session_id") or "") or None,
                       fallback_used=False)
        return {
            "ready": False,
            "note": "result_tech_unavailable",
            "result_preview": R.tech_unavailable_text("результаты анализов"),
            "entities_used": entities,
        }

    if outcome == R.NOT_FOUND:
        # существующий честный «не нашёл» (реворд BUG-2026-06-04-05)
        return {
            "ready": False,
            "note": "result_not_ready",
            "result_preview": _RESULT_NOT_READY_TEXT,
            "entities_used": entities,
        }
    # OK → существующий путь построения ссылки (payload = api_resp["data"] ...) без изменений.
```
Удалить прежние ветки `status_code==404`/`not payload`→`_RESULT_NOT_READY_TEXT` и `_result_fallback(service_error_results)` для not-ok (их заменяет классификатор). Сохранить построение ссылки для OK.

- [ ] **Step 5: Запустить целевые + соседние result-тесты**

Run: `./venv/bin/python -m pytest tests/test_messenger_services.py -q -p no:cacheprovider -k "result"`
Expected: PASS.

- [ ] **Step 6: Полный suite + ruff (гейт fix-задачи)**

Run: `ruff check messengers_router/services/lab_tests.py tests/test_messenger_services.py`
Run: `./venv/bin/python -m pytest -q -p no:cacheprovider`
Expected: PASS (полностью зелёный).

- [ ] **Step 7: Коммит (после OC-1)**

```bash
git add messengers_router/services/lab_tests.py tests/test_messenger_services.py
git commit -m "fix(results): tech failure (5xx/timeout/exc) -> honest tech_unavailable, not operator (resilience Фаза1)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: `response_builder` — payload `tech_unavailable`/`result_tech_unavailable` → честное сообщение

**Files:**
- Modify: `messengers_router/response_builder.py` (TEST_RESULT/результат-рендер; найти, где `result_preview` превращается в ответ)
- Test: `tests/test_messenger_services.py`

- [ ] **Step 1: Найти, как `result_preview`/TEST_RESULT payload рендерится в ответ**

Run: `grep -n "result_preview\|result_tech_unavailable\|TEST_RESULT\|result_links\|build_test_result" messengers_router/response_builder.py`
Зафиксировать функцию-строителя ответа результатов.

- [ ] **Step 2: Тест — tech payload даёт честный текст, не «не готов»/не оператор-эскалацию**

```python
def test_response_result_tech_unavailable_renders_honest_text():
    # синтетический evidence результата с tech_unavailable → ответ про техпричины,
    # без «результат пока не готов» и без auto-handoff.
    from messengers_router import resilience as R
    preview = R.tech_unavailable_text("результаты анализов")
    # минимальная проверка строителя: см. Step 1 (подставить реальную функцию/контракт)
    assert "техническ" in preview.lower() and "результат пока не готов" not in preview.lower()
```
(Если строитель результата просто отдаёт `result_preview` как текст — проверка покрыта Task 4; тогда здесь добавить e2e через `run_pipeline`-мок результата, либо отметить, что отдельный рендер не требуется.)

- [ ] **Step 3: Реализация** — если строитель уже передаёт `result_preview` в текст ответа без потери, изменений не нужно (Task 4 достаточно). Если есть отдельная ветка, добавить распознавание `note in {"result_tech_unavailable"}` → отдать `result_preview`, `handoff=False`. Показать точный diff после Step 1.

- [ ] **Step 4: Тест PASS** — `./venv/bin/python -m pytest tests/test_messenger_services.py -q -p no:cacheprovider -k "tech_unavailable or result"`

- [ ] **Step 5: ruff + коммит**

```bash
git add messengers_router/response_builder.py tests/test_messenger_services.py
git commit -m "fix(results): render tech_unavailable honest message in response builder

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: `_ensure_regions_loaded` сигналит тех-сбой; `address_info` → degraded best-effort

**Files:**
- Modify: `messengers_router/services/core.py` (`_ensure_regions_loaded`, 306-329)
- Modify: `messengers_router/services/addresses.py` (`address_info` — fallback-ветка doctors-cache + финальная пустая)
- Test: `tests/test_messenger_services.py`

**Проблема:** `_ensure_regions_loaded` при исключении возвращает `[]` И кэширует пусто → сигнал тех-сбоя теряется, не отличить от «нет регионов».

- [ ] **Step 1: Прочитать `_ensure_regions_loaded` (306-329)** дословно (для Edit).

- [ ] **Step 2: Тест — addresses помечает degraded при сбое /regions, отдаёт best-effort кэш**

```python
def test_address_info_regions_tech_failure_marks_degraded_uses_cache(monkeypatch):
    # /regions падает (исключение) → address_info использует doctors-cache (best-effort),
    # payload помечен degraded; адреса непустые (если кэш есть).
    import asyncio
    from messengers_router.services import Services
    s = Services()
    # форсим сбой regions-загрузки
    async def boom():
        raise ConnectionError("regions down")
    monkeypatch.setattr(s, "_ensure_regions_loaded", boom)
    ent = {"city": "Самара", "specialty": "гинеколог", "__appointment_mode": True}
    r = asyncio.run(s.address_info("Запись к врачу-гинекологу", ent))
    assert r.get("degraded") is True
    assert r.get("degraded_upstream") == "regions"
    # best-effort: если doctors-cache доступен — адреса есть; если нет — tech payload
    # (см. политику best-effort→tech). Достаточно проверить degraded-флаг.
```

- [ ] **Step 3: FAIL** — `./venv/bin/python -m pytest tests/test_messenger_services.py::test_address_info_regions_tech_failure_marks_degraded_uses_cache -q -p no:cacheprovider`

- [ ] **Step 4: Реализация — прокинуть флаг сбоя**

В `core.py::_ensure_regions_loaded`: при `except Exception` НЕ кэшировать пусто навсегда и выставить флаг на self:
```python
            try:
                regions = await asyncio.to_thread(api_nayka.site_regions)
                if not isinstance(regions, list):
                    regions = []
                self._regions_last_failed = False
            except Exception:
                regions = []
                self._regions_last_failed = True
```
(инициализировать `self._regions_last_failed = False` в `__init__`/где инициализируются regions-поля.)

В `addresses.py::address_info`, в doctors-cache fallback-ветке (после Гагарина-фикса): если `getattr(self, "_regions_last_failed", False)` → `R.log_degraded(upstream="regions", failure_mode=R.FM_EXCEPTION, fallback_used=bool(fallback))` и `R.mark_degraded(payload, upstream="regions", failure_mode=R.FM_EXCEPTION, fallback_used=bool(fallback))`. Если fallback пуст И regions упал → вернуть tech payload (`note="address_tech_unavailable"`, `result_preview`/`handoff_message` через `tech_unavailable_text("список филиалов")`).

- [ ] **Step 5: PASS + полный suite + ruff**

Run: `./venv/bin/python -m pytest -q -p no:cacheprovider` · `ruff check messengers_router/services/core.py messengers_router/services/addresses.py`

- [ ] **Step 6: Коммит**

```bash
git add messengers_router/services/core.py messengers_router/services/addresses.py tests/test_messenger_services.py
git commit -m "fix(address): signal /regions tech-failure, mark degraded + best-effort cache (resilience Фаза1)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: `doctors_schedule_week` — schedule-API тех-сбой → tech_unavailable vs not-found

**Files:**
- Modify: `messengers_router/services/doctors.py` (`doctors_schedule_week`, 685+; exception-ветка ~836)
- Test: `tests/test_messenger_services.py`

> **РЕШЕНИЕ ВЛАДЕЛЬЦА (2026-06-07) — Option 1, scope сужен (реализовано в `38292bf`):**
> Домен расписания УЖЕ честно обрабатывает тех-сбой: operator-handoff `service_error_schedule`
> + stale-serve + фильтр по запрошенному врачу (зафиксировано тестами 476/515/540/576/653/738).
> Буквальный Step 4 ниже (заменить handoff на `tech_unavailable` без авто-оператора + рендер в
> response_builder) **НЕ выполнялся** — он сломал бы 3 теста, убрал бизнес-корректную эскалацию
> при записи и противоречит собственному гейту плана «полный suite зелёный» + спеку §4.4.
> **РЕАЛИЗОВАНО:** ТОЛЬКО добавлены `R.log_degraded(upstream="doctor_schedule", failure_mode=FM_EXCEPTION,
> fallback_used=False)` + `R.mark_degraded(payload, …)` в exception-ветке (chokepoint hard-fail).
> Поведение/handoff/`response_builder.py` НЕ изменены. stale-serve уже наблюдаем через собственный
> лог `schedule_ttl_cache`. **Step 4 и response_builder-правка ниже — исторический текст, НЕ применять.**

- [ ] **Step 1: Прочитать `doctors_schedule_week` (685-840)** — понять, где schedule-fetch падает (try/except ~836) и как сейчас формируется payload (`schedule_unavailable_reason`, `doctor_lookup`).

- [ ] **Step 2: Тест — schedule-fetch исключение → degraded + tech payload (не «врач не найден», не дамп чужих)**

```python
def test_doctors_schedule_week_tech_failure_is_degraded(monkeypatch):
    import asyncio
    from messengers_router.services import Services
    s = Services()
    # форсим сбой источника расписания (подменить внутренний fetch на исключение —
    # точное имя метода зафиксировать в Step 1, напр. _schedule_for_doctor)
    ...
    # ожидание: payload.get("degraded") is True; schedule_unavailable_reason == "tech_unavailable"
```
(точный мок-таргет — из Step 1.)

- [ ] **Step 3: FAIL** — запустить целевой тест.

- [ ] **Step 4: Реализация** — в exception-ветке schedule-fetch: `R.log_degraded(upstream="doctor_schedule", failure_mode=R.FM_EXCEPTION, fallback_used=False)`, `schedule_unavailable_reason="tech_unavailable"`, `R.mark_degraded(payload, ...)`. `response_builder` (build_doctor_schedule / appointment preview) при `schedule_unavailable_reason == "tech_unavailable"` → честное `tech_unavailable_text("расписание")` вместо «расписание не найдено»/дампа. Показать точный diff после Step 1.

- [ ] **Step 5: PASS + полный suite + ruff.**

- [ ] **Step 6: Коммит**

```bash
git add messengers_router/services/doctors.py messengers_router/response_builder.py tests/test_messenger_services.py
git commit -m "fix(schedule): doctor-schedule tech-failure -> honest tech_unavailable, not wrong fallback (resilience Фаза1)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: `api_nayka` — realtime fail-fast профиль ретраев ⚠️ OC-2

**Files:**
- Modify: `agent_logic_2/nayka_api/api_nayka.py` (49-73: SESSION/_retry/_session_get)
- Test: `tests/test_resilience.py` или `tests/test_api_doctors_cache.py`

**OC-2: подтвердить числа у владельца** (total ретраев realtime, read-timeout). По умолчанию предлагаем: realtime `Retry(total=0, ...)`, read-timeout 8с; background — без изменений (`total=3`).

- [ ] **Step 1: Прочитать 45-73** дословно.

- [ ] **Step 2: Тест — realtime-сессия имеет fail-fast профиль (total<=1), background сохраняет total=3**

```python
def test_realtime_session_is_fail_fast():
    from agent_logic_2.nayka_api import api_nayka as A
    # realtime adapter total <= 1; background (SESSION) total == 3
    rt = A.SESSION_REALTIME.get_adapter("http://x")  # тип/имя — из Step 3
    bg = A.SESSION.get_adapter("http://x")
    assert rt.max_retries.total <= 1
    assert bg.max_retries.total == 3
```

- [ ] **Step 3: FAIL** — `./venv/bin/python -m pytest -q -p no:cacheprovider -k realtime_session`

- [ ] **Step 4: Реализация — добавить вторую сессию + realtime-флаг в `_session_get`**

```python
_retry_realtime = Retry(
    total=0, connect=0, read=0, backoff_factor=0.0,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=frozenset(["GET"]),
)
SESSION_REALTIME = requests.Session()
_adapter_rt = HTTPAdapter(max_retries=_retry_realtime)
SESSION_REALTIME.mount("https://", _adapter_rt)
SESSION_REALTIME.mount("http://", _adapter_rt)

REALTIME_READ_TIMEOUT = float(os.getenv("NAUKA_TIMEOUT_READ_REALTIME", "8"))
REALTIME_TIMEOUT = (REQ_CONNECT_TIMEOUT, REALTIME_READ_TIMEOUT)

def _session_get(url: str, *, realtime: bool = False, **kwargs):
    kwargs.setdefault("auth", auth)
    kwargs.setdefault("timeout", REALTIME_TIMEOUT if realtime else DEFAULT_TIMEOUT)
    kwargs.setdefault("verify", VERIFY_ARG)
    sess = SESSION_REALTIME if realtime else SESSION
    return sess.get(url, **kwargs)
```
Затем пометить realtime-вызовы (`resultForPatient`, `doctorSchedule`, `regions`-в-запросе) флагом `realtime=True` в их `_session_get(...)`. Background cache-loads оставить без флага.

- [ ] **Step 5: PASS + полный suite + ruff.**

- [ ] **Step 6: Коммит (после OC-2)**

```bash
git add agent_logic_2/nayka_api/api_nayka.py tests/test_resilience.py
git commit -m "perf(nayka): realtime fail-fast retry profile (no retry storm on degraded upstream)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review (для исполнителя — пройти перед стартом)

- **Покрытие спека:** §4.1 таксономия→Task1; §4.2 модуль→Task1-2; §4.3 политика→Task4,6,7; §4.4 сообщения→Task3,5; §4.5 логи→Task2(+вызовы в 4,6,7); §4.6 fail-fast→Task8; §4.7 шов оператора→`mark_degraded` поля (Task2) готовы к Фазе3. ✓
- **Открытые вопросы §7:** OC-1 (Task4), OC-2 (Task8), OC-3 (текст) — отмечены, не блокируют код, блокируют коммит затронутой задачи.
- **Типы/имена согласованы:** `R.OK/NOT_FOUND/TECH_UNAVAILABLE`, `classify_api_response`, `failure_mode_from_response`, `log_degraded`, `mark_degraded`, `tech_unavailable_text` — единообразны во всех задачах.
- **Точки, требующие чтения текущего кода перед Edit (exact-match):** Task3 (матрица+тест), Task4 (259-294 + тест 4397), Task5 (рендер результата), Task6 (306-329 + addresses fallback), Task7 (685-840), Task8 (45-73) — у каждой Step 1 = «прочитать дословно».

## Заметки
- Фаза 1 НЕ включает: `asyncio.gather` параллелизацию (Фаза 2), сигнал оператору в агрегаторе (Фаза 3), семантический матчер каталога (трек A).
- Каждая fix-задача (4,6,7,8) перед коммитом гоняет ПОЛНЫЙ `pytest` (анти-регресс по shared-путям). Foundational (1,2,3) — целевые тесты + ruff.

## Phase-2 follow-ups (из финального ревью фазы, SHIP-WITH-FOLLOWUPS)
1. **schedule fail-fast leading-calls** — ✅ ЗАКРЫТО в Task 8.1 (`2479aa2`).
2. **session_id** — пробросить из router-контекста во все `log_degraded` (сейчас всегда `None`; спек §7 open question — service-слой не имеет session в scope).
3. **latency_ms** — тонкая обёртка-таймер вокруг realtime upstream-вызовов (сейчас всегда `None`; спек: optional) для полноты Datadog-схемы.
4. **`_schedule_by_specialty` observability** — в двух `except`-ветках per-doctor fetch (doctors.py ~265-268) добавить `log_degraded` (сейчас specialty-путь молча глотает сбои, в отличие от single-doctor пути).
5. **schedule envelope refactor** — `find_doctor_schedule` возвращает `{ok,status_code}` вместо magic-строк → гранулярный `failure_mode` (timeout vs 5xx) для расписания, убрать `is_api_error_message`-shim.
6. **per-profile realtime connect-timeout** — отдельный `NAUKA_TIMEOUT_CONNECT_REALTIME` (сейчас realtime использует общий `REQ_CONNECT_TIMEOUT`=10с; worst-case connect+read; см. комментарий в api_nayka.py).
