# Messengers Router: Architecture Status (v3)

Этот файл — актуальная техническая сводка по `messengers_router` после последнего цикла рефакторинга.
Фокус: архитектура, границы слоёв, фактические метрики, риски и безопасные направления следующих изменений.

> Обновлено: **2026-03-31**
> Область проверки: `messengers_router/*`, `tests/*`, архитектурные и eval-скрипты.

## 0) Короткий срез состояния (на сегодня)

### 0.1 Подтверждено локальными проверками

1. Архитектурный guardrail:
   - `python3 messengers_router/scripts/check_architecture_imports.py`
   - результат: `ARCHITECTURE CHECK PASSED`
   - текущее состояние: **29 модулей**, **92 внутренних ребра импортов**, циклы не обнаружены.
2. Тесты:
   - `.agent_venv/bin/python -m pytest -q`
   - результат: **167 passed**.
3. Расширенное eval-покрытие:
   - `python3 messengers_router/scripts/check_eval_coverage.py`
   - результат: `COVERAGE CHECK PASSED` (stage5 extension/critical extension/follow-up/non-Samara/DOCTOR_INFO/DOCTOR_SCHEDULE/PREPARE/OTHER покрыты).

### 0.2 Что изменилось относительно ревизии 2026-03-21

1. Карта слоёв и метрики изменились:
   - было: 28 модулей / 89 ребер;
   - стало: 29 модулей / 92 ребра.
2. Введён отдельный domain-модуль `service_phrase.py` (извлечение процедурных фраз).
3. Актуализирован `topic_registry` как YAML-first реестр с безопасным CRUD API.
4. Объём интеграционного слоя вырос:
   - `services.py` теперь ~3409 LOC (главный источник технического долга).
5. Документ приведён к текущему формату без устаревших допущений по структуре пакета.

## 1) Карта архитектуры по слоям

Текущий `LAYER_BY_MODULE` (источник: `scripts/check_architecture_imports.py`):

| Слой | Модулей | Модули |
|---|---:|---|
| `interface` | 1 | `endpoint` |
| `orchestration` | 7 | `router`, `nlu_pipeline`, `classifier`, `planner`, `executor`, `response_builder`, `renderer` |
| `policy` | 9 | `appointment_flow_guard`, `entity_grounder`, `flow_policy`, `dialog_graph`, `recovery_policy`, `policies`, `context_summary`, `self_check`, `topic_registry` |
| `domain` | 6 | `mess_types`, `city`, `text_templates`, `llm_mode_policy`, `prompt_contracts`, `service_phrase` |
| `infrastructure` | 4 | `services`, `memory`, `llm_runtime`, `prompt_registry` |
| `port` | 2 | `runtime_config`, `doctor_name_port` |

Правило импорта слоёв контролируется guardrail-скриптом и CI (`.github/workflows/messengers_router_arch_guardrails.yml`).

## 2) Пайплайн одного пользовательского сообщения

1. `endpoint.py`
   - принимает HTTP (`/api/messenger-generate`, `/api/messenger-generate-once`);
   - нормализует runtime-options (`llm_mode/self_check/timeout`);
   - получает `SessionState` через `MemoryStore` (DI через `Depends` + `lru_cache`-фабрики).
2. `router.py`
   - orchestrator без прямого доступа к host-config/API мимо портов/сервисов;
   - шаги: NLU -> policy/flow guards -> planner -> executor -> response_builder/renderer;
   - формирует `ResponseEnvelope` и debug payload (в `debug=true` режиме once-endpoint).
3. `memory.py`
   - хранение истории/slot-state/pending + TTL + per-session async-lock.
4. `services.py`
   - интеграции с Nayka API / price / Meili;
   - graceful fallback-контракты с `handoff`-ориентированным поведением.
5. `renderer.py` + `self_check.py`
   - итоговый текст, rich-рендер и дополнительная критика/перегенерация (в `rich`-режиме).

## 3) Контракты и runtime-политика

### 3.1 Публичный API-контракт

- Streaming endpoint: `POST /api/messenger-generate` (`application/x-ndjson`).
- Debug-once endpoint: `POST /api/messenger-generate-once` (обычный JSON).
- Основной envelope-контракт: `text`, `attachments`, `handoff`, `state_update`.

### 3.2 Runtime-параметры LLM

- `strict`: без `llm_primary` NLU.
- `hybrid`: `llm_primary` NLU + детерминированные guardrails.
- `rich`: `hybrid` + self-check/refine path.

Источник нормализации: `llm_mode_policy.py`.

### 3.3 Prompt-policy

Загрузка prompt-ов: `prompt_registry.py`.

Порядок:
1. `app_data/prompts/mr_<key>.txt` (host override),
2. `messengers_router/prompts/<key>.txt` (bundle default).

Поддерживается мягкая миграция legacy-файлов `mr_<key>_v2.txt`.

### 3.4 Topic Registry

- Реестр: `messengers_router/data/topic_registry.yaml`.
- Модуль: `topic_registry.py` (loader/matcher + CRUD для UI).
- Используется в `router.py` как controlled override/hint layer.

## 4) Источники данных (операционные)

1. Адреса/филиалы: Nayka site API (`/regions` и производные данные).
2. Цены:
   - retail: `priceByRegion/3` (Самара),
   - врачебные цены: `doctorServicePricesByRegion`.
3. Knowledge fallback:
   - Meili (`main_index`) для информационных/подготовительных сценариев.
4. PREPARE:
   - `API-first` ветка пока заглушена в `services.py` (`_prepare_from_analysis_api_cache_stub`),
   - рабочий путь сейчас — Meili + relevance/fallback policy.

## 5) Фактические узкие места

### 5.1 Крупные модули (LOC)

- `services.py`: ~3409
- `policies.py`: ~1726
- `classifier.py`: ~1274
- `router.py`: ~1034

### 5.2 Связанность

- `router.py` остаётся центральным координатором с fan-out ≈ 21 internal module.
- Это даёт удобство orchestration, но повышает стоимость локальных правок.

### 5.3 Тестовый контракт

- Часть тестов всё ещё импортирует private API роутера:
  - `tests/test_specialty_nlu.py` (`_is_samara_city`)
  - `tests/test_llm_primary_nlu.py` (`_debug_meta`)
- Риск: фиксация внутренней реализации вместо публичного контракта.

## 6) Приоритеты следующего рефакторинга

### P1

1. Декомпозиция `services.py` на под-адаптеры:
   - `schedule_adapter`,
   - `pricing_adapter`,
   - `knowledge_adapter`,
   - `address_adapter`,
   - `prepare_adapter`.
2. Уменьшение orchestration-нагрузки `router.py`:
   - перенос части APPOINTMENT-веток в более изолированный flow-engine.

### P2

3. Перевод PREPARE на реальный `API-first` (вместо заглушки) с явным cache-contract.
4. Замена private-test импортов на публичные сценарные контракты.

### P3

5. Дальнейшее расширение default-gate eval-пакета (при командном решении):
   - follow-up,
   - non-Samara,
   - редкие label-комбинации.

## 7) Обязательные quality-gates после изменений

1. Юнит/интеграционные тесты:
   - `.agent_venv/bin/python -m pytest -q`
2. Архитектурный guardrail:
   - `python3 messengers_router/scripts/check_architecture_imports.py`
3. Eval coverage sanity:
   - `python3 messengers_router/scripts/check_eval_coverage.py`
4. Remote eval по серверу (если доступен стенд):
   - `bash messengers_router/eval_suite/run_remote_eval.sh --url <server>/api/messenger-generate-once`

## 8) Сжатая карта каталогов (актуально)

```text
messengers_router/
  endpoint.py
  router.py
  nlu_pipeline.py
  classifier.py
  planner.py
  executor.py
  response_builder.py
  renderer.py
  appointment_flow_guard.py
  entity_grounder.py
  flow_policy.py
  dialog_graph.py
  recovery_policy.py
  policies.py
  context_summary.py
  self_check.py
  topic_registry.py
  mess_types.py
  city.py
  text_templates.py
  service_phrase.py
  llm_mode_policy.py
  prompt_contracts.py
  services.py
  memory.py
  llm_runtime.py
  prompt_registry.py
  runtime_config.py
  doctor_name_port.py
  prompts/*.txt
  data/
    cities.txt
    nonbookable_points.json
    topic_registry.yaml
    TOPIC_REGISTRY_GUIDE.md
  eval_suite/
  scripts/
```

## 9) Быстрый onboarding для разработчика/Codex

1. Прочитать этот файл.
2. Прочитать в порядке:
   - `endpoint.py`
   - `router.py`
   - `nlu_pipeline.py`
   - `services.py`
3. Запустить проверки из раздела 7.
4. Для отладки одного хода использовать:
   - `POST /api/messenger-generate-once` с `"debug": true`,
   - смотреть `state_update.debug` (`decision/plan/evidence/nlu_trace`).

## 10) Критерии “не ломаем” при дальнейших изменениях

1. Не ломать публичный контракт `text/attachments/handoff/state_update`.
2. Не допускать роста ложного handoff на простых пользовательских сценариях.
3. Не записывать неподтверждённые сущности в `state.last_entities`.
4. Не обходить слой портов/адаптеров прямыми host-import в core-слоях.

