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
- dual-pass NLU (`rule pass` + `LLM pass`) и merge;
- явный FSM/graph слой;
- recovery-политика (clarify/escalation);
- entity grounding перед записью в state;
- bounded context summary для длинных сессий.

## 3) Пайплайн запроса (как идет одно сообщение)

1. `endpoint.py`
   - читает request (`/api/messenger-generate`, `/api/messenger-generate-once`);
   - достает session state из `memory.py`.

2. `router.py` (оркестратор)
   - вызывает NLU (`nlu_pipeline.py`, fallback/shadow на `classifier.py`);
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
- `nlu_pipeline.py`: rule+LLM merge в одно `RouteDecision`.
- `classifier.py`: hard-rules + LLM-классификация, JSON-схема вывода.
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
    classifier_patient.txt
    classifier_refine_patient.txt
    renderer_patient.txt
    versions/v2/*.txt
  data/
    cities.txt
    nonbookable_points.json
  scripts/
    eval_stage1_cases.py
    eval_stage3_appointment_flow.py
    eval_stage5_corpus.py
    compare_analysis_addresses.py
  messengers_mds_to_collect_thoughts/
    *.md, analysis/*.json*
```

## 5) Источники адресов (критично для support)

### Для nonbookable сценариев (анализы/ЭКГ)
Приоритет источника:
1. `messengers_router/data/nonbookable_points.json` (статический справочник; сейчас заполнен для Самары).
2. fallback на live API:
   - `api_nayka.site_regions()` (`/regions`)
   - фильтрация по `api_price.load_price_all()` (`/priceAll`, `regionId`).

Текущий факт:
- для Самары nonbookable-ветка берется из статического каталога;
- для городов, отсутствующих в `nonbookable_points.json`, результат зависит от live API и фильтров.

## 6) Feature flags / runtime toggles

- `MR_ROUTER_V2_ENABLE` (default: on): включение v2 NLU pipeline.
- `MR_ROUTER_V2_SHADOW` (default: off): сравнение v2 с legacy classifier.
- `MR_PROMPT_VERSION` (`v1`/`v2`, default: `v2`): выбор prompt-версии.

## 7) Известные слабые места (актуально)

1. `APPOINTMENT` логика все еще распределена между `router.py`, `flow_policy.py`, `policies.py`.
2. Есть overlap проверок `doctor_name` (`router._verify_doctor_entity` и `entity_grounder`).
3. Качество по rare/свободным формулировкам зависит от сочетания quick-fill + grounding + pending-контекста.
4. Для городов вне статического nonbookable-каталога поведение зависит от доступности/полноты live API.
5. Latency в сценариях расписания чаще упирается во внешние API, а не в локальную логику.

## 8) Что рефакторить дальше (рекомендуемый порядок)

1. Убрать дубли doctor-validation в один слой (`entity_grounder` как единственный source of truth).
2. Перенести APPOINTMENT step-machine целиком в отдельный модуль (`appointment_flow.py`) и держать `router.py` только как coordinator.
3. Унифицировать quick-fill правила через “ожидаемый слот” (pending-driven extraction only).
4. Ввести единый tracing-объект pipeline для debug (вместо разрозненных флагов).
5. Добавить contract tests на response-shape + state transitions по ключевым сценариям.

## 9) Аудит перед пушем: что лишнее/шумное

### Шум, который не должен попадать в git
- `.DS_Store` (в разных каталогах),
- `__pycache__/` и `*.pyc`.

### Не runtime-артефакты (держать осознанно)
- `messengers_router/messengers_mds_to_collect_thoughts/*` — аналитические md/json для разработки.
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
