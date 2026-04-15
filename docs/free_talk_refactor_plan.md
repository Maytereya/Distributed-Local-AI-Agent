# План рефакторинга Free Talk

Статус: v0.2 (синхронизировано с кодом)  
Дата: 2026-04-15

## 1. Назначение

Этот документ фиксирует целевой план рефакторинга FT и его фактический статус после основной волны изменений.

Главная идея:

- не встраивать новый adapter в уже перегруженный `agent.py`;
- сначала сделать структуру FT более прямолинейной;
- затем переносить доменную логику в новые модули поэтапно;
- legacy оставлять только на boundary к backend.

Связанные документы:

- [free_talk_runtime_architecture.md](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/docs/free_talk_runtime_architecture.md)
- [free_talk_data_registry.md](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/docs/free_talk_data_registry.md)
- [free_talk_translation_rules.md](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/docs/free_talk_translation_rules.md)
- [free_talk_processing_pipeline.md](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/docs/free_talk_processing_pipeline.md)
- [free_talk_feature_integration_playbook.md](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/docs/free_talk_feature_integration_playbook.md)

## 2. Цели рефакторинга

Нужно добиться четырёх результатов:

1. убрать критическую перегрузку `agent.py`;
2. ввести явный adapter layer между FT и legacy backend;
3. отделить deterministic state/mapping logic от LLM-driven wording/routing;
4. сделать дальнейшие багфиксы и регрессионные тесты локальными, а не сквозными по всему runtime.

## 3. Что не является целью первого этапа

На первом этапе не планируется:

1. переписывать `messengers_router/services.py`;
2. переписывать Redis engine;
3. менять внешний FT API / UI contract;
4. делать большой rename всего пакета за один коммит;
5. ломать существующий `LegacyServicesPort`.

## 4. Целевые архитектурные принципы

### 4.1 Code owns contracts

Следующие вещи должны быть детерминированы кодом:

- names of `missing_slots`;
- normalized entities;
- topic shift reset;
- state merge rules;
- adapter translation;
- unsupported city policy;
- memory reuse rules.

### 4.2 LLM owns language, not contracts

LLM должна отвечать за:

- route wording при неоднозначности;
- natural clarification wording;
- synthesis/recommendation после получения данных.

LLM не должна:

- придумывать slot names;
- вводить legacy aliases;
- сама решать backend mapping.

### 4.3 Boundary compatibility only

На рефакторинге сохраняются только необходимые внешние границы:

- FT API / runner / Gradio integration;
- backend contract на стороне `messengers_router/services.py`;
- regression coverage.

Внутренние FT legacy names и переходные мосты не считаются обязательными к сохранению.

## 5. Целевая структура модулей

Ниже приведена целевая структура. Она не обязана появиться за один коммит, но должна быть ориентиром.

```text
src/localragagent/freetalk/
  runner.py
  orchestrator.py
  dialog_state.py
  routing_contract.py
  routing_prompting.py
  tool_planning.py
  followup_policy.py
  memory_policy.py
  adapter.py
  adapter_contracts.py
  rendering.py
  contracts.py
  memory_redis.py
  memory_persist.py
  config.py
  observability.py
```

### 5.1 Роли целевых модулей

| Целевой модуль | Назначение |
|---|---|
| `orchestrator.py` | Главный chat-loop и orchestration стадий |
| `dialog_state.py` | Canonical dialog state, merge/reset/versioning |
| `routing_contract.py` | Канон router output и normalization |
| `routing_prompting.py` | Prompt builders для router/verifier |
| `tool_planning.py` | Heuristic signals и fallback tool planning |
| `followup_policy.py` | Анафора, contextual follow-up, topic shift |
| `memory_policy.py` | Session entity memory и правила reuse/reset |
| `adapter.py` | FT entities -> backend payload и обратно |
| `adapter_contracts.py` | Adapter request/response dataclasses |
| `rendering.py` | Fallback render, source fragments, final shaping |

## 6. Реальная карта модулей после миграции

| Область | Текущий source-of-truth | Состояние |
|---|---|---|
| Public FT API facade | `agent.py` | Активен |
| Top-level orchestration | `orchestrator.py` | Активен |
| Dialog state lifecycle | `dialog_state.py` | Активен |
| Follow-up / topic shift | `followup_policy.py` | Активен |
| Session memory reuse | `memory_policy.py` | Активен |
| Routing schema | `routing_contract.py` | Активен |
| Router / verifier prompting | `routing_prompting.py` | Активен |
| Heuristic tool planning | `tool_planning.py` | Активен |
| FT -> backend translation | `adapter.py` | Активен |
| Fallback render / source shaping | `rendering.py` | Активен |

Важно:

- целевые модули уже стали реальным source-of-truth;
- transitional file names должны отсутствовать в runtime-импортах и документации.

## 7. Порядок реализации

### Этап 0. Документирование [выполнено]

Артефакты:

- data registry;
- capability matrix;
- translation rules;
- processing pipeline;
- runtime architecture;
- refactor plan.

Цель:

- зафиксировать термины и границы до кода.

### Этап 1. Разукрупнение без изменения поведения [выполнено]

Что сделано:

1. создаём новые модули-заготовки;
2. переносим pure/helper logic из `agent.py` без смены поведения;
3. оставляем `FreeTalkAgent` как фасад над новым orchestrator.

Приоритетные выносы:

1. `dialog_state.py`
2. `rendering.py`
3. `followup_policy.py`
4. `memory_policy.py`

Критерий:

- tests green;
- поведение FT не меняется функционально.

### Этап 2. Введение adapter contracts [выполнено]

Что сделано:

1. создаём `adapter_contracts.py`;
2. вводим FT-facing request/response структуры;
3. отделяем user-facing entities от backend aliases.

Приоритетный домен:

- `test_result_status`

Причина:

- там mapping наиболее явный:
  - `result_surname -> surname`
  - `result_year_of_birth -> year`
  - `result_analysis_code -> filial`
  - `result_analysis_number -> number`

### Этап 3. Реализация adapter layer [выполнено для clinic domains]

Что сделано:

1. создаём `adapter.py`;
2. переносим domain-specific translation из `agent.py`;
3. нормализуем backend payload обратно в FT-friendly shape.

Порядок доменов:

1. `test_result`
2. `doctor_info`
3. `doctor_schedule`
4. `service / price / prepare`
5. `address`
6. `local indexes` как отдельный source-mode сценарий без clinic-adapter translation

### Этап 4. Чистка router contract и prompts [выполнено]

Что сделано:

1. обновляем `missing_slots` под новый канон;
2. убираем устаревшие slot names из prompt schema;
3. синхронизируем router/verifier с adapter contracts.

Результат:

- router больше не генерирует старые поля FT-слоя вроде `doctor_name_or_specialty`, `result_filial`, `result_number`.

### Этап 5. Упрощение orchestration [выполнено]

Что сделано:

1. orchestration использует только новые policy/adapters;
2. удаляем ad hoc-translation из `agent.py`;
3. делаем flow более straight-line:
   - route
   - normalize
   - ground
   - clarify or call
   - normalize response
   - render

### Этап 6. Legacy cleanup и rename [выполнено]

Что сделано:

1. `agent.py` сокращён до публичного фасада над `orchestrator.py` и policy-модулями;
2. source-of-truth перенесён в `routing_contract.py`, `routing_prompting.py`, `tool_planning.py`;
3. transitional helpers и legacy bridges в FT-слое удалены или сведены к backend boundary.

## 8. Совместимость на переходный период

### 8.1 API compatibility

Нужно сохранить:

- FT runner contract;
- FT API endpoint contract;
- Gradio integration contract.

### 8.2 Redis compatibility

Нужно сохранить:

- текущий storage engine;
- канонический `clinical_dialog_state`;
- корректную работу `session_entity_memory`.

### 8.3 Backend compatibility

Нужно сохранить:

- вызов `LegacyServicesPort`;
- legacy payload names внутри backend;
- отсутствие обязательных правок в `messengers_router/services.py` на первом этапе.

## 9. Тестовая стратегия

Рефакторинг считается безопасным только если проверяются:

1. unit tests по routing/normalization;
2. integration tests по multi-turn clarification;
3. remote eval suite против FT endpoint;
4. regression-cases из реальных экспортов диалогов.

Минимальные обязательные сценарии:

1. `doctor -> schedule -> specialty topic shift`
2. `fuzzy service -> confirm -> tool`
3. `test_result multi-step collection`
4. `self-error correction after wrong doctor`
5. `unsupported city`

## 10. Критерии готовности

Рефакторинг можно считать успешным, когда:

1. `agent.py` перестаёт быть единственным местом критической логики;
2. adapter translation живёт в отдельном модуле;
3. prompts используют новый канон слотов и сущностей;
4. Redis-слой хранит новый state без хаоса legacy names;
5. remote eval ловит содержательные ошибки, а не архитектурные сбои state/merge.

По состоянию на 2026-04-15 критерии 1-4 закрыты. Основной оставшийся акцент — качество реальных multi-turn диалогов и regression/eval coverage.

## 11. Практическое правило для реализации

Любой новый кусок FT-логики должен отвечать на вопрос:

`Это orchestration, policy, adapter, routing contract или rendering?`

Если ответ неясен, код не должен попадать обратно в `agent.py`.
