# Дизайн: слой устойчивости (degraded-mode honesty + наблюдаемость + fail-fast) — Фаза 1

- **Дата:** 2026-06-06
- **Статус:** утверждён (brainstorming), готов к writing-plans
- **Область:** `messengers_router` (NLU-движок прод = `legacy_v2`)
- **Автор обсуждения:** Владимир + Claude (сессия 2026-06-04…06)

---

## 1. Проблема (из анализа buglog, 15 багов)

Точечные фиксы багов долговечны (класс-инвариант-тесты, авторитетные источники, приоритет правил). Но три **системных** бреши продолжают плодить баги:

- **(B) Нет честной таксономии degraded-mode.** Бот валит РАЗНЫЕ сбои апстримов в одно (часто неверное) сообщение и **дезинформирует**:
  - 404 от тестового medserver на реального пациента → «Результат пока не готов» (BUG-2026-06-04-05) — ложь.
  - `/regions` недоступен → запись к гинекологу предложила лабораторный «Гагарина 64» (BUG-2026-06-04-04).
  - `doctorSchedule` тормозит → расписание Паничевой подменилось списком чужих врачей (BUG-2026-06-04, отложен).
  - 404-логика уже **флип-флопнула** дважды (BUG-2026-06-02-03 сделал 404→«в работе»; BUG-...-05 нашёл, что это дезинформация) — признак отсутствия системы.
- **(C) Нет наблюдаемости.** Разработчики не знают частоту degraded-событий; операторы не видят, что проблема в сети/инфре, а не в пациенте.
- **Латентность.** `Retry(total=3, backoff_factor=0.5)` в `agent_logic_2/nayka_api/api_nayka.py` усиливает задержку на больном апстриме → eval-таймауты 45-75с (RLY_02 ЭКГ→/regions, RLY_15 «нет»→ollama). Латентность — триггер, заставляющий degraded-mode срабатывать часто.

Прод-апстримы: `ollama` (LLM), medserver `172.16.0.246:8081/medserver-test` (`/regions`, `/resultForPatient`, `/doctorSchedule`).

## 2. Цели и не-цели

**Цели (Фаза 1):**
1. Честно различать **тех-сбой** (не дозвонились/не распарсили) и **бизнес-пусто** (дозвонились, данных нет) → при тех-сбое отвечать «По техническим причинам сейчас не удаётся загрузить …», НЕ дезинформировать и НЕ угадывать.
2. Сохранять полезный ответ, когда есть безопасный кэш (best-effort), помечая событие degraded.
3. Структурно логировать каждое degraded-событие (для разработчиков; схема Datadog-ready).
4. Fail-fast на пациент-фейсинг realtime-вызовах, чтобы тех-ответ приходил за секунды, а не за 45-75с (попутно чинит eval-таймауты).

**Не-цели (отдельные фазы / треки):**
- **Фаза 2:** параллелизация независимых апстрим-вызовов (`asyncio.gather`).
- **Фаза 3:** сигнал оператору в агрегаторе (`support-messenger-aggregator`, код друга) — Фаза 1 лишь ставит флаг в handoff-payload.
- **Брешь A** (семантический матчер каталога — ОАК/Са-125/T-SPOT/сахар/коагулограмма) — отдельный трек.
- Метрики в Datadog (Фаза 2/3) — схема логов делается Datadog-ready, но интеграция позже.

## 3. Подход

**Подход C + новый модуль `messengers_router/resilience.py`.** Переиспользуем существующие швы вместо новой архитектуры:
- `agent_logic_2/nayka_api/api_nayka.py` уже отдаёт `{ok, status_code, error, data}`.
- `messengers_router/policies.py::HANDOFF_REASON_MATRIX` + `evidence_requires_handoff` — готовая точка для сообщений/handoff.
- `evidence`-слой несёт `note`/`handoff_reason`/payload по доменам.

Новый модуль `resilience.py` инкапсулирует ТОЛЬКО общую логику: классификатор исхода + degraded-логгер + константы. Доменные сервисы вызывают его и решают cache-fallback vs tech-ответ.

Отвергнутый подход A (обёртка вокруг каждого вызова) — больше переплетения; C ниже по риску и опирается на проверенные места (как фиксы BUG-03/04/05).

## 4. Дизайн

### 4.1. Таксономия исхода апстрим-вызова

Три класса (enum/строки-константы в `resilience.py`):
- `OK` — данные присутствуют.
- `NOT_FOUND` — апстрим ответил, но по данным пусто: HTTP 404 ИЛИ `ok=True`+пустой `data`. Бизнес-состояние «не нашёл / не готов / не найден».
- `TECH_UNAVAILABLE` — не дозвонились/не распарсили: таймаут, conn-error, HTTP 5xx, ретраи исчерпаны, исключение, `status_code is None`.

Классификатор читает структуру ответа `api_nayka` (`ok`/`status_code`/`error`). Правила:
- `status_code == 404` → `NOT_FOUND`.
- `ok=True` и `data` пуст → `NOT_FOUND`.
- `status_code` 5xx ИЛИ `None` (таймаут/conn) ИЛИ исключение при вызове → `TECH_UNAVAILABLE`.
- `ok=True` и `data` есть → `OK`.

### 4.2. `messengers_router/resilience.py` (новый модуль)

Минимальная поверхность:
- `class UpstreamOutcome` (или строковые константы `OK`/`NOT_FOUND`/`TECH_UNAVAILABLE`).
- `classify_api_response(resp: dict | None, *, exc: Exception | None = None) -> str` — возвращает класс по правилам 4.1.
- `log_degraded(*, upstream: str, failure_mode: str, latency_ms: int | None, session_id: str | None, fallback_used: bool) -> None` — пишет ОДНУ структурную строку лога (стабильное имя события `degraded_upstream`), поля как в 4.5.
- Константа честного сообщения `TECH_UNAVAILABLE_TEXT` (или фабрика по домену, см. 4.4).
- (Опц.) хелпер `mark_degraded(payload: dict, *, upstream, failure_mode, fallback_used)` — проставляет в payload `degraded=True`, `degraded_upstream`, `degraded_mode` для логирования и Фазы-3-сигнала.

Модуль НЕ ходит в сеть, НЕ зависит от доменных сервисов (только stdlib + logging) — тестируется изолированно.

### 4.3. Политика (best-effort кэш → честный тех-ответ)

Каждый доменный сервис при `TECH_UNAVAILABLE`:
1. Если есть **безопасный кэш** (например, `doctors-cache` для филиалов/расписания) → использовать его, вернуть данные, но `mark_degraded(payload, fallback_used=True)` + `log_degraded(...)`. Ответ остаётся полезным.
2. Если кэша нет/пуст (например, `resultForPatient`) → вернуть payload с `tech_unavailable` (без данных, **без угадывания**) + `log_degraded(fallback_used=False)`.

`NOT_FOUND` — прежние доменные честные тексты (результаты: реворд BUG-05; адреса: «не нашёл по этим данным»). `OK` — без изменений.

### 4.4. Честные сообщения (через `HANDOFF_REASON_MATRIX`)

- Новая причина `tech_unavailable` в `HANDOFF_REASON_MATRIX` (`policies.py`):
  > «По техническим причинам сейчас не удаётся загрузить {что}. Пожалуйста, попробуйте позже или напишите «оператор».»
  где `{что}` — доменная подстановка («результаты анализов», «список филиалов», «расписание»), дефолт «информацию».
- `response_builder`: payload с `tech_unavailable` → это сообщение; `handoff_required` НЕ ставим автоматически (как и при 404 — без спама оператора), но текст предлагает написать «оператор» (user-initiated).
- ВАЖНО: `tech_unavailable` ≠ текущий `service_error_*` (тот эскалирует на оператора). Для realtime-доменов без кэша заменяем «service_error → оператор» на честный «тех-причины» (мягче, без авто-эскалации). Сохранять авто-эскалацию там, где она бизнес-правильна (решается по домену в плане).

### 4.5. Наблюдаемость (структурные логи)

Один вызов `log_degraded(...)` на degraded-событие → строка с **стабильным именем** `degraded_upstream` и полями:
- `upstream`: `regions` | `result_for_patient` | `doctor_schedule` | `price` | `ollama` | …
- `failure_mode`: `timeout` | `conn_error` | `http_5xx` | `http_404` | `empty_data` | `exception`
- `latency_ms`: длительность вызова (если измерима), иначе `None`.
- `session_id`: из контекста запроса (если доступен).
- `fallback_used`: bool (использовали ли кэш).

Формат лога согласован с текущим стилем (`log.warning("event_name", ...)`); поля — в структурированном виде (kwargs/extra), чтобы парсер/Datadog позже агрегировал по `upstream`×`failure_mode`. БЕЗ новых зависимостей.

### 4.6. Fail-fast (перф-вход Фазы 1)

Проблема: глобальный `Retry(total=3, connect=3, read=3, backoff_factor=0.5)` на больном апстриме даёт 45-75с.

Решение — **раздельные профили**, БЕЗ изменения глобального фонового профиля:
- **realtime-профиль** (пациент ждёт ответа: `resultForPatient`, `doctorSchedule`, `regions` в запросе) — `total=1` (или `0`), ужатый read-timeout (ориентир: connect ~2s, read ~5-8s) → больной вызов возвращается за секунды → `TECH_UNAVAILABLE` быстро.
- **background-профиль** (прогрев кэша вне запроса) — текущий `Retry(total=3)` сохраняется.

Реализация: либо отдельная requests-session с другим `Retry`, либо параметр `timeout`/`max_retries` на конкретных realtime-вызовах. Точные числа — **открытый вопрос для владельца** (см. §7), осторожная валидация (это общий HTTP-слой).

### 4.7. Шов для оператора (Фаза 3, не сейчас)

degraded-payload несёт `degraded`/`degraded_reason=tech_unavailable`. В Фазе 3 при передаче оператору этот маркер кладётся в handoff-payload → агрегатор друга показывает операторам «проблема с сетью». Фаза 1: только ставим флаг + логируем.

## 5. Точки интеграции (file:function)

- **NEW** `messengers_router/resilience.py` — классификатор, `log_degraded`, константы.
- `messengers_router/policies.py` — `HANDOFF_REASON_MATRIX` (+`tech_unavailable`).
- `messengers_router/services/lab_tests.py::test_result_status` — 5xx/таймаут → `tech_unavailable` (сейчас → `service_error_results`→оператор); 404/empty → `NOT_FOUND` (уже реворд).
- `messengers_router/services/addresses.py::address_info` — отличать «/regions недоступен» (TECH, best-effort doctors-cache + degraded) от «дозвонились, пусто» (NOT_FOUND).
- `messengers_router/services/doctors.py` / `core.py` — `doctors_schedule_week`: schedule-API сбой → TECH (best-effort кэш) vs «врач не найден» (NOT_FOUND).
- `agent_logic_2/nayka_api/api_nayka.py` — realtime/background профили `Retry`/timeout (§4.6).
- `messengers_router/response_builder.py` — payload `tech_unavailable` → честное сообщение (§4.4).

## 6. Тестирование (класс-инвариант, не инстанс)

- `resilience.classify_api_response`: параметризовано — timeout/5xx/conn/None/exception → `TECH_UNAVAILABLE`; 404/empty-data → `NOT_FOUND`; data → `OK`.
- Политика по доменам: TECH + есть кэш → данные + `degraded=True`; TECH без кэша → `tech_unavailable` (НЕ дезинформация, НЕ угадывание).
- `response_builder`: `tech_unavailable` → честный тех-текст (НЕ «результат не готов», НЕ «Гагарина 64»).
- `log_degraded`: проверка полей (capture log / monkeypatch).
- Регресс: `OK`-путь не меняется; существующие `NOT_FOUND`-тексты (реворд результатов) не меняются; авто-эскалация на оператора сохраняется там, где задумана.
- Fail-fast: синтетический медленный/падающий апстрим → вызов возвращается в пределах realtime-бюджета (без 45с-зависания).
- Гейт: `ruff` + полный `pytest` (зелёный) после каждого атомарного шага.

## 7. Риски и открытые вопросы

- **Общий HTTP-слой (`api_nayka`):** изменение `Retry`/timeout затрагивает ВСЕ вызовы — раздельные профили обязательны; осторожная валидация. **Открытый вопрос (владелец):** конкретные числа realtime-профиля (`total`, connect/read timeout).
- **`service_error`→`tech_unavailable` миграция:** не везде заменять авто-эскалацию; по домену решить в плане, где «тех-причины без оператора» уместно, где сохранить эскалацию.
- **`session_id` в логе:** доступен ли в точках сервиса (может потребоваться прокинуть из контекста) — уточнить при планировании.
- **Сообщение `tech_unavailable`:** финальная формулировка — как и реворд BUG-05, утверждается владельцем.

## 8. Фазировка

- **Фаза 1 (этот спек):** таксономия + `resilience.py` + best-effort/tech-политика + честные сообщения + degraded-логи + fail-fast realtime-профиль.
- **Фаза 2:** `asyncio.gather` параллелизация независимых апстрим-вызовов (perf).
- **Фаза 3:** сигнал оператору в агрегаторе + (опц.) метрики Datadog.
- **Отдельный трек:** брешь A — семантический матчер каталога.
