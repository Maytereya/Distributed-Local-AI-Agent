# Messengers Router: Current Architecture Status (v2)

Этот файл — единая точка входа для разработчиков и их Codex при работе с `messengers_router`.
Цель: быстро понять текущую архитектуру, отличие от старой версии, болевые точки и безопасные направления рефакторинга.

## 1) Что это за контур

`messengers_router` — самостоятельный диалоговый контур пациентского бота (Telegram/WhatsApp), который:
- принимает пользовательский текст,
- определяет интент и сущности,
- управляет flow-состоянием сессии,
- ходит в сервисы/CRM API,
- формирует пациенту итоговый ответ или handoff оператору.

Важно: это не операторский бот. Здесь выше требования к предсказуемости и безопасности ответа.

## 2) Отличие v2 от старого подхода

### Было (legacy-style)
- основной вес логики в роутере + точечные rule-патчи;
- слабая явная модель состояний диалога;
- часть конфликтов интентов решалась ad-hoc.

### Стало (v2)
- оркестрация по модульному pipeline;
- rollout между `legacy_v2` и `llm_primary` через feature flag;
- `llm_primary` path: `guardrail_pre -> primary LLM JSON -> deterministic postprocess`;
- явный FSM/graph слой;
- structured recovery/clarify-политика;
- entity grounding перед записью в state;
- bounded context summary для длинных сессий.

## 3) Пайплайн запроса (как идет одно сообщение)

1. `endpoint.py`
   - читает request (`/api/messenger-generate`, `/api/messenger-generate-once`);
   - достает session state из `memory.py`.

2. `router.py` (оркестратор)
   - вызывает NLU (`nlu_pipeline.py`, shadow/fallback на legacy classifier);
   - применяет flow-хелперы (`flow_policy.py`);
   - применяет `entity_grounder.py`;
   - собирает план инструментов (`build_plan`);
   - выполняет план через `services.py`;
   - применяет graph transition (`dialog_graph.py`);
   - отдает финальный response через `renderer.py`.

3. `memory.py`
   - хранит историю, pending-слоты, last_entities, summary.

4. `renderer.py`
   - форматирует человекочитаемый ответ пациенту.

## 4) Модули и ответственность

- `endpoint.py`: только HTTP-контракт + стрим/once режим.
- `router.py`: orchestration pipeline, без “heavy text parsing”.
- `nlu_pipeline.py`: выбор engine (`legacy_v2` vs `llm_primary`) и debug-trace.
- `classifier.py`: guardrails, primary LLM JSON classification, deterministic postprocess.
- `entity_grounder.py`: валидация/нормализация сущностей перед merge в state.
- `flow_policy.py`: stateful-хелперы APPOINTMENT/pending/quick-fill.
- `policies.py`: детекторы интентов, clarify-тексты, slot-политики, quick-fill.
- `dialog_graph.py`: FSM переходы (`IDLE/APPOINTMENT_FLOW/...`).
- `recovery_policy.py`: low-confidence clarify/escalation logic.
- `context_summary.py`: bounded summary контекста для NLU.
- `services.py`: интеграции и нормализация данных из внешних источников.
- `prompt_registry.py`: версионирование prompt-шаблонов.
- `prompt_contracts.py`: проверка/санитизация LLM JSON-контракта.
- `city.py`: распознавание города, fuzzy-матч.
- `mess_types.py`: доменные dataclass-типы.

## 4.1) Структура папок (сжатая карта)

```text
messengers_router/
  endpoint.py
  router.py
  nlu_pipeline.py
  classifier.py
  entity_grounder.py
  flow_policy.py
  policies.py
  dialog_graph.py
  recovery_policy.py
  context_summary.py
  memory.py
  services.py
  renderer.py
  city.py
  mess_types.py
  prompt_registry.py
  prompt_contracts.py
  prompts/
    versions/v2/*.txt   # активные шаблоны
    *.txt               # legacy fallback (неосновной путь)
  data/
    cities.txt
    nonbookable_points.json
  scripts/
    audit_nayka_site_api.py
    eval_stage1_cases.py
    eval_stage3_appointment_flow.py
    eval_stage4_reliability.py
    eval_stage5_corpus.py
    compare_analysis_addresses.py
    build_stage5_golden_cases.py
  messengers_mds_to_collect_thoughts/analysis/
    stage5_golden_cases.jsonl
    golden_versions/*.jsonl
```

## 5) Источники адресов (критично для support)

### Для адресных и nonbookable сценариев
Основной источник: live `api_nayka.site_regions()` (`/regions`) с фильтрацией по флагам филиала:
- `analysis`, `ecg`, `usi`, `doctorService`.

Текущий факт:
- адреса/контакты и фильтрация филиалов идут из live `/regions`;
- fallback — только адреса из doctor cache, если live API недоступен.

## 5.1) Источники цен (актуально)

- Розничные цены: `api_price.load_price_by_region(3)` (`/priceByRegion/3`, Самара).
- Врачебные цены: `api_price.load_doctor_prices()` (кэш `doctorServicePricesByRegion`).
- Для запроса вида "цена у врача X" выполняется резолв `doctor_name -> doctor_id` в `services.py`.

## 6) Feature flags / runtime toggles

- `MR_ROUTER_V2_ENABLE` (default: on): включение v2 NLU pipeline.
- `MR_ROUTER_V2_SHADOW` (default: off): сравнение v2 с legacy classifier.
- `MR_NLU_ENGINE=legacy_v2|llm_primary` (default: `legacy_v2`): выбор primary-NLU engine.
- `MR_NLU_SHADOW=0|1` (default: `0`): shadow compare для нового NLU path.
- `MR_PROMPT_VERSION`: оставлен для совместимости, но фактически поддерживается только `v2`.
- Prompt policy: загрузка идет только по versioned-файлам (`prompts/versions/v2/*`).

### `llm_mode` semantics

- `strict`: не использует free-form LLM-primary NLU; остается на legacy/fallback path.
- `hybrid`: основной production-target для `llm_primary`.
- `rich`: тот же `llm_primary` NLU + richer renderer/self-check + optional rare refine.

## 7) Известные слабые места (актуально)

1. `APPOINTMENT` логика все еще распределена между `router.py`, `flow_policy.py`, `policies.py`.
2. Есть overlap проверок `doctor_name` (`router._verify_doctor_entity` и `entity_grounder`).
3. `llm_primary` улучшает free-form recall, но качество все еще ограничено grounding и качеством live data.
4. При недоступности live `/regions` адресная выдача деградирует до doctor-cache fallback.
5. Latency в сценариях расписания чаще упирается во внешние API, а не в локальную логику.
6. Текущий Stage 5 golden недостаточен как единственный gate: 49 кейсов и перекос в `PRICE/APPOINTMENT/TEST_ASSIST`.

## 8) Что рефакторить дальше (рекомендуемый порядок)

1. Убрать дубли doctor-validation в один слой (`entity_grounder` как единственный source of truth).
2. Перенести APPOINTMENT step-machine целиком в отдельный модуль (`appointment_flow.py`) и держать `router.py` только как coordinator.
3. Унифицировать quick-fill правила через “ожидаемый слот” (pending-driven extraction only).
4. Расширить golden/eval на `DOCTOR_INFO`, `DOCTOR_SCHEDULE`, `PREPARE`, `OTHER`, non-Samara и follow-up turns.
5. После стабилизации удалить legacy-shadow ветки и лишний rule-duplication.

## 9) Аудит перед пушем: что лишнее/шумное

### Шум, который не должен попадать в git
- `.DS_Store` (в разных каталогах),
- `__pycache__/` и `*.pyc`.

### Не runtime-артефакты (держать осознанно)
- `messengers_router/messengers_mds_to_collect_thoughts/analysis/*.jsonl` — golden-корпус и версии для eval.
- `messengers_router/чаты из ватсап для ии/*.txt` — сырой корпус чатов (очень большой объем, не участвует в runtime).

### Потенциальный долг по коду
- legacy/shadow path в `router.py` (`analyze` + `_nlu_shadow`) полезен для сравнения, но со временем может быть удален после стабилизации v2.

## 10) Минимальный onboarding для нового разработчика/Codex

1. Прочитать этот файл.
2. Прочитать:
   - `messengers_router/router.py`
   - `messengers_router/nlu_pipeline.py`
   - `messengers_router/flow_policy.py`
   - `messengers_router/services.py`
3. Запустить локальный smoke:
   - `python messenger_simulator.py "http://localhost:8000/api/messenger-generate" <session_id>`
4. Для дебага одного хода использовать:
   - `POST /api/messenger-generate-once` с `"debug": true`.
   - Смотреть `state_update.debug.nlu_trace` для `guardrail_pre/llm_primary_raw/guardrail_post/final_decision`.

## 11) Критерий “не ломаем” при следующих изменениях

- не ломать API-контракт `text/attachments/handoff/state_update`;
- не увеличивать false handoff на простых пациентских сценариях;
- не допускать запись неподтвержденных сущностей в state;
- проверять базовые happy-path кейсы:
  - greeting → intent,
  - doctor schedule by surname,
  - appointment city→doctor→time flow,
  - test-result guidance + link generation,
  - nonbookable address flow по городу.

## 12) Прогресс по ТЗ (оперативный статус)

Актуально после цикла доработок по endpoint-аудиту и мессенджерному роутеру.

### 12.1 Что уже сделано технически

- Проведен live-аудит `api/v1/site/*` с примерами ответов и usage-map по коду:
  - `docs/nayka_site_api_live_audit_latest.md`
  - `messengers_router/scripts/audit_nayka_site_api.py`
- Цена в мессенджере:
  - retail: `priceByRegion/3` (Самара),
  - doctor-specific: `doctorServicePricesByRegion` через daily cache.
- Возвращен стабильный PRICE intent (без автоматического handoff к оператору).
- Для сценария "цена у врача":
  - добавлен path `doctor_name -> doctor_id -> doctor prices`.
- Усилена Samara-only фильтрация по врачам/расписанию/адресам (исключение иногородних данных).
- Prompt-слой приведен к `v2-only` (legacy prompt-файлы удалены).

### 12.2 Статус 8 пунктов ТЗ

1. **Результаты анализов (ссылка)** — `READY`
   - Работает в `TEST_RESULT`.
2. **Стоимость анализов** — `PARTIAL`
   - Работает через `priceByRegion/3`, но нужно довести релевантность/синонимы.
3. **Услуга + розничный прайс + top-4 врачей по ord + доступность + подготовка** — `PARTIAL`
   - Части готовы по отдельности, нет единого сквозного сценария.
4. **Конкретный врач по ФИО + прайс его услуг** — `PARTIAL`
   - Связка `ФИО -> doctor_id` добавлена; нужен финальный UX-контракт и edge-cases.
5. **Врачи выбранной специальности с учетом ord** — `PARTIAL`
   - Сортировка по `ord` есть; нужно закрепить правило top-4 + availability.
6. **Расписание врача по ФИО** — `READY`
7. **Запись к врачу по ФИО** — `PARTIAL`
   - Flow записи есть, финал сейчас через handoff оператору (не прямой CRM commit).
8. **Справка в налоговую** — `NOT_IN_SCOPE`
   - Делегировано коллеге.

## 13) Что брать коллеге в работу (приоритет)

### P1 (сразу)

1. Закрыть пункт 3 ТЗ как единый use-case:
   - услуга -> retail price -> top-4 врачей (`ord asc`) -> проверка доступности расписания -> подготовка.
2. Закрыть пункт 5 ТЗ:
   - выдача врачей по специальности строго top-4 с понятным deterministic сортом.

### P2 (следом)

3. Дошлифовать пункт 2 ТЗ:
   - улучшить матчинг цен анализов (синонимы/морфология, меньше шумных совпадений).
4. Дошлифовать пункт 4 ТЗ:
   - стабилизировать сценарий "цена услуги у конкретного врача" (падежи ФИО, редкие формулировки).

### P3 (архитектурно)

5. Решить policy по пункту 7:
   - остается handoff или делаем прямой commit записи в CRM.
6. Уточнить security policy по пункту 1:
   - обязательна ли строгая авторизация перед выдачей ссылки на результат.
