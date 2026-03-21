# Messengers Router: Current Architecture Status (v2)

Этот файл — единая точка входа для разработчиков и их Codex при работе с `messengers_router`.
Цель: быстро понять текущую архитектуру, отличие от старой версии, болевые точки и безопасные направления рефакторинга.

> Актуализация: **2026-03-21** (повторная сверка после Wave1 + server eval).
> Текущим источником правды для ближайших работ считается раздел **0** ниже.

## 0) Актуализация архитектуры (2026-03-21)

### 0.1 Критерии архитектурной состоятельности (зафиксировано)

Архитектуру считаем состоятельной, если одновременно выполняется:

1. Четкие границы слоев:
   - transport/API слой не содержит бизнес-решений;
   - оркестрация не тянет инфраструктуру напрямую.
2. Отсутствуют циклические зависимости между core-модулями пакета.
3. Интеграции изолированы в сервисном/адаптерном слое.
4. Расширяемость:
   - добавление канала/провайдера не требует правок ядра роутинга.
5. Наблюдаемость:
   - debug-trace и причины fallback/handoff доступны без чтения внутренностей кода.

### 0.2 Фактическая карта модулей (по коду)

Инвентаризация выполнена по всем `messengers_router/*.py` (**28** core-модулей).

- Точки входа: `endpoint.py` (`/api/messenger-generate`, `/api/messenger-generate-once`).
- Оркестрация: `router.py`, `nlu_pipeline.py`, `planner.py`, `executor.py`, `response_builder.py`.
- Flow-guard: `appointment_flow_guard.py`, `flow_policy.py`, `dialog_graph.py`, `memory.py`.
- NLU/Recovery/Render: `classifier.py`, `entity_grounder.py`, `recovery_policy.py`, `renderer.py`, `self_check.py`.
- Интеграции: `services.py`, `llm_runtime.py`, `prompt_registry.py`.
- Базовые политики и типы: `policies.py`, `mess_types.py`, `city.py`, `topic_registry.py`, `text_templates.py`.

### 0.3 Проверка зависимостей и нарушений границ

#### Что в порядке

- Циклы импортов между core-модулями: **не обнаружены** (`0` SCC > 1 узла).
- Архитектурный guardrail локально проходит: `ARCHITECTURE CHECK PASSED` (`modules checked: 28`, `internal edges: 89`).
- Debug/eval контуры стабильны: последний remote eval (`run_id=1774094099`) прошел `100%` по всем стадиям, включая `critical` и `coverage_ext`.

#### Что требует исправления (актуальные нарушения)

1. **God-модули / концентрация ответственности**:
   - `services.py` (~2456 LOC),
   - `policies.py` (~1847 LOC),
   - `classifier.py` (~1155 LOC),
   - `router.py` (~1034 LOC).

2. **Сильная централизация оркестратора**:
   - `router.py` имеет fan-out `21` модуль (наибольшее в пакете).

3. **Неявные контрактные связи**:
   - тесты активно используют приватные функции `router.py` (`_...`), что фиксирует внутреннюю реализацию вместо публичного контракта.

4. **Расширяемость каналов была ограничена**:
   - проблема закрыта в Wave1/Step3: `endpoint.py` переведен на dependency-factory (`Depends` + провайдеры `get_memory_store/get_services`) вместо module-level singleton.

5. **Infra-границы в core закрыты (контроль включен)**:
   - прямые `agent_logic_2.*` импорты убраны из orchestration/policy;
   - оставлены только в адаптерах/портах (`runtime_config`, `doctor_name_port`, `services`, `llm_runtime`, `prompt_registry`);
   - правило закреплено архитектурным guardrail-скриптом и CI gate.

### 0.4 Реестр проблем (приоритизация)

- `P1`: чрезмерная связанность `router.py`.
  Риск: высокий; Цена: средняя; Эффект: ускорение безопасных изменений.
- `P2`: god-модули `services.py` и `policies.py`.
  Риск: высокий; Цена: высокая; Эффект: управляемость и тестопригодность.
- `P2`: приватные контракты в тестах.
  Риск: средний; Цена: низкая/средняя; Эффект: устойчивость рефакторинга.
- `[closed]`: ограниченная DI-расширяемость endpoint слоя (закрыто Wave1/Step3, 2026-03-21).
  Эффект: endpoint готов к dependency overrides и альтернативным провайдерам без правок core-router.
- `[closed]`: infra-утечки (`agent_logic_2.*`) в orchestration/policy слоях (закрыто Wave1/Step2, закреплено guardrail в CI).
  Эффект: границы слоев формализованы и автоматически контролируются.

### 0.5 План рефакторинга (2 волны, текущий)

#### Волна 1 — быстрые исправления без смены поведения

1. [done] Консолидировать doctor-validation в одном месте (`entity_grounder.py`), убрать дубли из `router.py`.
2. [done] Ввести тонкий слой runtime-settings/ports для `agent_logic_2.config`, убрать прямые импорты из orchestration/policy модулей.
3. [done] Ослабить связанность endpoint: `memory/services` вынесены в dependency-factory (DI-ready).
4. [done] Добавить архитектурную автопроверку зависимостей (скрипт + CI gate):
   - запрет циклов,
   - whitelist межслойных импортов;
   - запрет прямых `agent_logic_2.*` импортов вне адаптеров/портов.

#### Волна 2 — структурные изменения

1. Выделить полноценный `appointment_flow` engine из `router.py` (router = coordinator only).
2. Декомпозировать `services.py` на адаптеры (schedule/pricing/knowledge/documents) с явными интерфейсами.
3. Декомпозировать `policies.py` на подмодули (`intent_detectors`, `slot_policy`, `appointment_texts`).
4. Перевести тесты на контрактный уровень:
   - меньше прямых проверок `_private` функций,
   - больше сценарных/контрактных тестов.

### 0.6 Обязательные quality-gates после каждого шага

1. `pytest` по модульным тестам.
2. `bash messengers_router/eval_suite/run_remote_eval.sh --url <server>/api/messenger-generate-once` — целевой результат `100%`.
3. Проверка архитектурного gate:
   - локально: `python3 messengers_router/scripts/check_architecture_imports.py`;
   - в CI: `.github/workflows/messengers_router_arch_guardrails.yml`.

### 0.7 Прогресс выполнения (оперативно)

- 2026-03-21: Wave1/Step1 завершен (doctor-validation перенесен в `entity_grounder`, дубли в `router` убраны).
- 2026-03-21: Wave1/Step2 завершен:
  - добавлен `runtime_config` порт;
  - добавлен `doctor_name_port`;
  - orchestration/policy модули переведены на внутренние порты;
  - quality-gate пройден: `run_remote_eval` `run_id=1774086292`, `All stages passed`.
- 2026-03-21: Wave1/Step3 завершен:
  - `endpoint.py` переведен с module-level singleton на dependency factories (`get_memory_store`, `get_services`);
  - локальный `pytest`: `140 passed`;
  - quality-gate пройден: `run_remote_eval` `run_id=1774088264`, `All stages passed`.
- 2026-03-21: Wave1/Step4 завершен:
  - добавлен guardrail-скрипт `messengers_router/scripts/check_architecture_imports.py`;
  - добавлен обязательный CI job `.github/workflows/messengers_router_arch_guardrails.yml`;
  - локальная проверка: `ARCHITECTURE CHECK PASSED`;
  - quality-gate пройден: `run_remote_eval` `run_id=1774089602`, `All stages passed`.
- 2026-03-21: закрыт регресс nonbookable follow-up (`TEST_ASSIST -> ADDRESS`):
  - усилены intent-детекторы (`сдать/здать`, profile-follow-up с контекстом `test_goal`, prepare-вопросы по времени/натощак);
  - добавлен soft-yes для secondary-offer (`спасибо`, `хорошо`, `ладно`, `окей`);
  - добавлены регрессионные тесты на формулировки и typo-cases;
  - локальный quality-gate: `pytest` `150 passed`;
  - server quality-gate: `run_remote_eval` `run_id=1774094099`, `All stages passed`.
- 2026-03-21: P3 policy-решения обновлены:
  - пункт 7 ТЗ зафиксирован как `handoff-only` (без прямого CRM commit);
  - пункт 8 ТЗ остается `pending` до юридического решения заказчика.
- 2026-03-21: подготовлен расширенный eval coverage-kit (P2, без включения в default gate):
  - stage5 extension golden: `analysis/golden_versions/stage5_golden_extension_v3.jsonl`;
  - critical extension cases: `eval_suite/critical_cases_extended.jsonl`;
  - DoD + coverage-check: `eval_suite/EVAL_EXPANSION_DOD.md`, `scripts/check_eval_coverage.py`.

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
- `appointment_flow_guard.py`: guard/state-политики для активного APPOINTMENT flow.
- `nlu_pipeline.py`: выбор engine (`legacy_v2` vs `llm_primary`) и debug-trace.
- `llm_mode_policy.py`: нормализация runtime-опций (`strict/hybrid/rich`, self-check, queue timeout).
- `classifier.py`: guardrails, primary LLM JSON classification, deterministic postprocess.
- `llm_runtime.py`: единый runtime-слой вызовов LLM (очередь, таймауты, stream/text генерация).
- `entity_grounder.py`: валидация/нормализация сущностей перед merge в state.
- `flow_policy.py`: stateful-хелперы APPOINTMENT/pending/quick-fill.
- `planner.py`: сборка tool/service плана из `RouteDecision`.
- `executor.py`: исполнение plan steps и сбор evidence.
- `response_builder.py`: детерминированная сборка структурированных ответов по лейблам.
- `policies.py`: детекторы интентов, clarify-тексты, slot-политики, quick-fill.
- `dialog_graph.py`: FSM переходы (`IDLE/APPOINTMENT_FLOW/...`).
- `recovery_policy.py`: low-confidence clarify/escalation logic.
- `self_check.py`: critic/self-check разбор для `rich`-рендера и decision о регенерации ответа.
- `context_summary.py`: bounded summary контекста для NLU.
- `services.py`: интеграции и нормализация данных из внешних источников.
- `text_templates.py`: централизованные фиксированные пользовательские шаблоны текста.
- `prompt_registry.py`: runtime-загрузка prompt-шаблонов из `app_data/prompts` с fallback на bundle-файлы в `messengers_router/prompts`.
- `prompt_contracts.py`: проверка/санитизация LLM JSON-контракта.
- `city.py`: распознавание города, fuzzy-матч.
- `topic_registry.py`: rule-based topic fallback/override (`topic_id -> label`).
- `mess_types.py`: доменные dataclass-типы.

## 4.1) Структура папок (сжатая карта)

```text
messengers_router/
  endpoint.py
  router.py
  appointment_flow_guard.py
  nlu_pipeline.py
  llm_mode_policy.py
  classifier.py
  llm_runtime.py
  entity_grounder.py
  flow_policy.py
  planner.py
  executor.py
  response_builder.py
  policies.py
  dialog_graph.py
  recovery_policy.py
  self_check.py
  context_summary.py
  memory.py
  services.py
  renderer.py
  text_templates.py
  city.py
  topic_registry.py
  mess_types.py
  prompt_registry.py
  prompt_contracts.py
  prompts/
    classifier_patient.txt
    classifier_refine_patient.txt
    recovery_patient.txt
    renderer_critic_patient_alignment.txt
    renderer_patient.txt
    renderer_patient_rich.txt
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
- Prompt policy:
  - сначала читается host override `app_data/prompts/mr_<key>.txt`;
  - если override нет, используется bundle prompt `messengers_router/prompts/<key>.txt`;
  - legacy-имя host override `mr_<key>_v2.txt` мягко мигрируется в `mr_<key>.txt`.

### `llm_mode` semantics

- `strict`: не использует free-form LLM-primary NLU; остается на legacy/fallback path.
- `hybrid`: основной production-target для `llm_primary`.
- `rich`: тот же `llm_primary` NLU + richer renderer/self-check + optional rare refine.

## 6.1) Какие prompt-файлы реально используются

Ключи, которые грузятся runtime-кодом через `load_prompt_text(...)`:

- `classifier_patient` -> `classifier.py`
- `classifier_refine_patient` -> `classifier.py`
- `renderer_patient` -> `renderer.py`
- `renderer_patient_rich` -> `renderer.py`
- `recovery_patient` -> `recovery_policy.py`
- `renderer_critic_patient_alignment` -> `self_check.py`

Практический вывод:
- базовые bundle prompt-файлы в `messengers_router/prompts/*.txt` должны существовать всегда;
- на реальном сервере при наличии `app_data/prompts/mr_<key>.txt` используются именно host override-файлы.

### 6.2) Текущая таблица prompt-источников

Ниже зафиксировано текущее состояние prompt-слоя в репозитории и на локальном runtime.

| Prompt key | Где используется | Bundle default | Host override | Что реально используется сейчас | Состояние |
|---|---|---|---|---|---|
| `classifier_patient` | `classifier.py` | `messengers_router/prompts/classifier_patient.txt` | `app_data/prompts/mr_classifier_patient.txt` | host override | bundle и host различаются |
| `classifier_refine_patient` | `classifier.py` | `messengers_router/prompts/classifier_refine_patient.txt` | `app_data/prompts/mr_classifier_refine_patient.txt` | host override | bundle и host различаются |
| `renderer_patient` | `renderer.py` | `messengers_router/prompts/renderer_patient.txt` | `app_data/prompts/mr_renderer_patient.txt` | host override | bundle и host различаются |
| `renderer_patient_rich` | `renderer.py` | `messengers_router/prompts/renderer_patient_rich.txt` | `app_data/prompts/mr_renderer_patient_rich.txt` | host override | bundle и host различаются |
| `recovery_patient` | `recovery_policy.py` | `messengers_router/prompts/recovery_patient.txt` | `app_data/prompts/mr_recovery_patient.txt` | host override | bundle и host различаются |
| `renderer_critic_patient_alignment` | `self_check.py` | `messengers_router/prompts/renderer_critic_patient_alignment.txt` | `app_data/prompts/mr_renderer_critic_patient_alignment.txt` | host override | bundle и host совпадают |

Ключевой риск:
- правка только `messengers_router/prompts/*.txt` не меняет поведение runtime, если соответствующий `app_data/prompts/mr_<key>.txt` уже существует;
- из-за этого bundle prompt-ы и host override prompt-ы могут разъехаться по смыслу.

Что это значит для следующего этапа:
- нужно принять единое решение, какие тексты считаются каноническими;
- после этого синхронизировать bundle и host prompt-ы по каждому ключу;
- до синхронизации любые изменения prompt-ов надо проверять сразу в обоих местах.

### 6.3) Рекомендуемая prompt-policy для команды

Рекомендуемое решение для текущего этапа проекта:

- канонический источник prompt-ов для командной разработки — `messengers_router/prompts/*.txt`;
- `app_data/prompts/mr_*.txt` — только host/runtime override:
  - временный hotfix,
  - правка через будущий интерфейс,
  - точечный серверный эксперимент без redeploy.

Почему это рекомендуется:
- bundle prompt-ы живут в git, проходят review и видны в diff;
- они воспроизводимы между разработчиками и стендами;
- `app_data` удобно для runtime-override, но неудобно как командный source of truth, потому что легко разъезжается с репозиторием.

Практический вывод:
- если prompt признан удачным на сервере через `app_data`, его нужно переносить обратно в `messengers_router/prompts/*.txt`;
- eval и локальные тесты нужно прогонять осознанно: либо на чистом bundle, либо понимая, что поведение задает именно host override.

### 6.4) Практический план синхронизации prompt-ов

Следующий рабочий шаг для команды:

1. Для каждого ключа сравнить `app_data/prompts/mr_<key>.txt` и `messengers_router/prompts/<key>.txt`.
2. По каждому ключу выбрать каноническую версию:
   - `classifier_patient`
   - `classifier_refine_patient`
   - `renderer_patient`
   - `renderer_patient_rich`
   - `recovery_patient`
   - `renderer_critic_patient_alignment`
3. Победившую версию сохранить в bundle (`messengers_router/prompts/*.txt`).
4. После этого:
   - либо удалить соответствующий `app_data/prompts/mr_<key>.txt`,
   - либо пересоздать его из bundle, если на сервере нужен осознанный override.
5. После синхронизации прогнать:
   - локальный smoke в `messenger_simulator.py`,
   - `eval_stage5_corpus.py`,
   - несколько ручных happy-path кейсов по `PRICE`, `APPOINTMENT`, `TEST_RESULT`, `ADDRESS`.

Рекомендуемый порядок принятия решений:

1. `classifier_patient`
2. `classifier_refine_patient`
3. `recovery_patient`
4. `renderer_patient`
5. `renderer_patient_rich`
6. `renderer_critic_patient_alignment`

Причина такого порядка:
- сначала стабилизируется NLU/clarify behavior;
- потом формат финального ответа;
- critic-пrompt трогается последним, потому что он влияет только на `rich`/self-check path.

## 7) Известные слабые места (актуально)

1. `APPOINTMENT` логика все еще распределена между `router.py`, `flow_policy.py`, `policies.py`.
2. `router.py` все еще большой coordinator (feature-rich), что увеличивает цену точечных изменений.
3. `llm_primary` улучшает free-form recall, но качество все еще ограничено grounding и качеством live data.
4. При недоступности live `/regions` адресная выдача деградирует до doctor-cache fallback.
5. Latency в сценариях расписания чаще упирается во внешние API, а не в локальную логику.
6. Текущий Stage 5 golden недостаточен как единственный gate: 49 кейсов и перекос в `PRICE/APPOINTMENT/TEST_ASSIST`.

## 8) Что рефакторить дальше (рекомендуемый порядок)

1. Перенести APPOINTMENT step-machine целиком в отдельный модуль (`appointment_flow.py`) и держать `router.py` только как coordinator.
2. Унифицировать quick-fill правила через “ожидаемый слот” (pending-driven extraction only).
3. Расширить golden/eval на `DOCTOR_INFO`, `DOCTOR_SCHEDULE`, `PREPARE`, `OTHER`, non-Samara и follow-up turns.
4. После стабилизации удалить legacy-shadow ветки и лишний rule-duplication.

## 9) Аудит перед пушем: что лишнее/шумное

### Шум, который не должен попадать в git
- `.DS_Store` (в разных каталогах),
- `__pycache__/` и `*.pyc`.

### Не runtime-артефакты (держать осознанно)
- `messengers_router/messengers_mds_to_collect_thoughts/analysis/*.jsonl` — golden-корпус и версии для eval.

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
- Prompt-слой переведен на схему `host override -> bundle default` без runtime-переключения версий.
- Для справок/документов:
  - `DOC_REQUEST -> main_index_info` (`main_index`).
  - При `no matches` включается handoff оператору с явным сообщением.
- Для подготовки:
  - добавлен rule-роутинг `PREPARE` (вопросы "как подготовиться..." и близкие формулировки).
  - текущий runtime путь: `test_prepare -> Meili main_index`.
  - в коде есть явная заглушка под будущий `API-first` для подготовки к анализам (`serviceInfoAll/preparation`).

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
7. **Запись к врачу по ФИО** — `READY (handoff-policy)`
   - Flow записи и подтверждения работает.
   - Политика зафиксирована: финал только через handoff оператору; прямой CRM commit намеренно не используется как anti-scam/anti-spam защита.
8. **Справка в налоговую / документальные запросы** — `PARTIAL`
   - `DOC_REQUEST` работает через `main_index`.
   - При отсутствии совпадений (`no matches`) — автоматический перевод на оператора.
   - Дальнейшая доработка: расширение покрытия knowledge-контента и качество матчинга формулировок.

## 13) Что брать коллеге в работу (приоритет, актуально на 2026-03-21)

### Уже закрыто в текущем цикле

1. [done] Wave1 / Step1: консолидирована doctor-validation логика в `entity_grounder`, дубли в `router` убраны.
2. [done] Wave1 / Step2: добавлены внутренние порты `runtime_config` и `doctor_name_port`, core-модули переведены на них.
3. [done] Wave1 / Step3: `endpoint.py` переведен на DI-фабрики зависимостей (`get_memory_store`, `get_services`) вместо module-level singleton.
4. [done] Wave1 / Step4: добавлен архитектурный guardrail (проверка циклов, whitelist межслойных импортов, контроль direct host-imports) + CI gate в GitHub Actions.
5. [done] Quality-gate после текущего цикла: `run_remote_eval` прошел на `100%` (run_id `1774094099`).

### P1 (сразу, следующий спринт)

Временный статус: отложено до готовности API серверов заказчика.

1. Довести пункт 3 ТЗ как единый сценарий:
   - услуга -> retail price -> top-4 врачей (`ord asc`) -> availability -> подготовка.
2. Довести пункт 5 ТЗ:
   - выдача врачей по специальности строго top-4 с deterministic сортировкой.

### P2 (следом)

Временный статус: отложено до готовности API серверов заказчика.

3. Перевести блок подготовки к анализам в `API-first`:
   - основной источник: `serviceInfoAll/preparation`;
   - fallback: `Meili main_index`.
4. Реализовать загрузку/обновление `serviceInfoAll` в файловый кэш.
5. Подключить API-кэш подготовки в runtime и покрыть тестами + eval-кейсами.
6. [in progress] Расширить eval/golden на `DOCTOR_INFO`, `DOCTOR_SCHEDULE`, `PREPARE`, `OTHER`, non-Samara, follow-up turns.
   - подготовлены extension-наборы и DoD;
   - подключение в default gate отложено до решения команды.

### P3 (стратегические решения)

7. [done] Policy по пункту 7 ТЗ зафиксирована:
   - финал записи только через handoff оператору;
   - прямой CRM commit не реализуется (anti-scam/anti-spam guardrail).
8. [pending] Security policy по выдаче результатов анализов:
   - решение отложено до юридической проработки у заказчика;
   - текущий режим оставляем без изменений до финального решения.
