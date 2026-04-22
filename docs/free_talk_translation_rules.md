# Правила трансляции данных Free Talk по доменам

Статус: v0.2 (синхронизировано с кодом)  
Дата: 2026-04-15

## 1. Назначение

Этот документ фиксирует domain-specific translation rules:

- из FT `missing_slots`;
- в FT normalized entities;
- затем в adapter-facing payload;
- затем в legacy backend aliases/arguments.

Документ фиксирует правила трансляции, которые уже должны соблюдаться в `adapter.py` и связанных policy-модулях.

Связанные документы:

- [free_talk_data_registry.md](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/docs/free_talk_data_registry.md)
- [free_talk_capability_matrix.md](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/docs/free_talk_capability_matrix.md)
- [free_talk_processing_pipeline.md](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/docs/free_talk_processing_pipeline.md)
- [free_talk_runtime_architecture.md](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/docs/free_talk_runtime_architecture.md)
- [free_talk_refactor_plan.md](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/docs/free_talk_refactor_plan.md)

## 2. Где это будет жить в коде

Текущая реализация использует явный translation layer:

- [adapter.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/src/localragagent/freetalk/adapter.py)
- [adapter_contracts.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/src/localragagent/freetalk/adapter_contracts.py)
- [freetalk_services_port.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/src/localragagent/ports/freetalk_services_port.py)

Роль слоёв:

- FT runtime оперирует каноническими FT entities;
- `adapter.py` переводит их в backend-facing payload и нормализует ответ обратно;
- `freetalk_services_port.py` остаётся gateway к `messengers_router.Services`.

## 3. Общие правила трансляции

### 3.1 Что не должно уходить в adapter как есть

Следующие имена не должны быть backend-facing каноном FT:

- `branch_or_city`
- `result_analysis_code`
- `result_analysis_number`
- `service_variant`

Это user-facing или state-facing имена.  
Перед adapter они должны быть разложены в нормализованные поля.

Примечание:

- legacy composite-slot `doctor_name_or_specialty` удалён из FT-канона и заменён на отдельные `doctor_name` и `specialty`.

### 3.2 Common normalization before adapter

Перед вызовом backend adapter должен:

1. Удалить незакрытые `missing_slots` из payload.
2. Не передавать служебные FT-поля:
   - `_ft_*`
   - `candidate_entities`
   - `confirmation_target`
   - `clarify_type`
3. Для doctor-related tools:
   - не тащить stale doctor, если текущая реплика явно про `specialty`.
4. Для service-related tools:
   - если есть `service_name` и `service_variant`, сначала собрать эффективное название услуги.
5. Для doctor schedule/info:
   - если город не указан, считать default `Самара` как policy, но не обязательно сохранять это как user entity.

### 3.3 Internal derived flags

Следующие флаги не являются ни user-facing слотами, ни LLM-entities.  
Они должны вычисляться алгоритмически в коде adapter/orchestration слоя.

#### `availability_scope`

Возможные значения:

1. `doctor_visit`
2. `service_delivery`
3. `lab_collection`

Смысл:

- `doctor_visit` — где принимает конкретный врач или врачи по специальности;
- `service_delivery` — где реально выполняется процедура/услуга с учетом оборудования и условий;
- `lab_collection` — где можно сдать анализ/биоматериал.

Правило:

- `availability_scope` выводится из domain/tool plan и не должен генерироваться LLM как свободная сущность.

## 4. Таблица по доменам

## 4.1 Doctor Info Domain

Используется для:

- карточки врача;
- списка врачей по специальности;
- списка врачей, которые делают услугу.

Основной backend tool:

- `doctors_info`

### Translation table

| FT layer | Канон FT | Adapter payload | Legacy backend aliases в `Services.doctors_info` | Комментарий |
|---|---|---|---|---|
| missing slot | `doctor_name` | не передаётся | не передаётся | Нужно дособрать конкретного врача, если фамилия неоднозначна |
| missing slot | `specialty` | не передаётся | не передаётся | Нужно дособрать специальность |
| normalized entity | `doctor_name` | `doctor_name` | `doctor_name`, `doctor`, `fio`, `last_name`, `doctor_last_name` | Для конкретного врача |
| normalized entity | `doctor_surname` | `doctor_last_name` | `last_name`, `doctor_last_name` | Использовать только если полного `doctor_name` ещё нет |
| normalized entity | `specialty` | `specialty` | `specialty`, `specialization`, `spec` | Основной путь для `все урологи` |
| normalized entity | `service_name` | `service_name` | `service_name`, `test_name` | В FT doctor-related queries сюда должен попадать только medical service/procedure, а не лабораторный анализ |
| normalized entity | `branch_name` | `branch` | `region`, `branch`, `company_unit` | В `doctors_info` лучше маппить branch name в `branch` |
| normalized entity | `city` | не передавать по умолчанию | не используется прямо | Для `doctors_info` city-фильтр отдельно не нужен в FT baseline |

### Правила

1. Если есть `specialty` и нет явного person-like doctor signal, stale `doctor_name` должен быть отброшен.
2. Если есть `doctor_name`, `doctor_surname` не обязателен.
3. Если есть только `doctor_surname`, adapter может передавать его как `doctor_last_name`.
4. `test_name` не должен использоваться как канонический вход для doctor availability/info сценариев, даже если backend legacy-слой терпит alias `test_name`.

## 4.2 Doctor Schedule Domain

Используется для:

- расписания конкретного врача;
- расписания по специальности.

Основной backend tool:

- `doctors_schedule_week`

### Translation table

| FT layer | Канон FT | Adapter payload | Legacy backend aliases в `Services.doctors_schedule_week` | Комментарий |
|---|---|---|---|---|
| missing slot | `doctor_name` | не передаётся | не передаётся | Если нужен конкретный врач |
| missing slot | `specialty` | не передаётся | не передаётся | Если пользователь хочет расписание по специальности |
| missing slot | `branch_or_city` | не передаётся | не передаётся | Только если пользователь реально сужает расписание по филиалу/городу |
| missing slot | `date` | не передаётся | не передаётся | FT-layer фильтр, backend сам живёт недельным окном |
| missing slot | `time` | не передаётся | не передаётся | FT-layer фильтр после ответа backend |
| normalized entity | `doctor_name` | `doctor_name` | `last_name`, `doctor_last_name`, `doctor`, `doctor_name`, `fio` | Backend сам вынимает фамилию для расписания |
| normalized entity | `doctor_surname` | `doctor_last_name` | `last_name`, `doctor_last_name` | Безопасный fallback |
| normalized entity | `specialty` | `specialty` | `specialty`, `specialization`, `spec` | Для `расписание уролога` |
| normalized entity | `branch_name` | `branch_name` | `region`, `branch`, `company_unit`, `unit`, `branch_name` | Можно маппить напрямую |
| normalized entity | `city` | `city` | `city` | Для `Самара` / unsupported city behavior |

### Правила

1. Если текущая реплика явно про специальность, stale doctor не должен подмешиваться.
2. Если `city != Самара`, adapter может передавать `city` как есть, а backend уже вернёт unsupported fallback.
3. `date/time` сейчас в backend contract явно не участвуют как API аргументы; это фильтры FT-layer/post-processing.

## 4.3 Service / Price / Prepare Domain

Используется для:

- `service_bundle_info`
- `price_info`
- `test_prepare`
- `test_assist`

### Translation table

| FT layer | Канон FT | Adapter payload | Legacy backend aliases | Комментарий |
|---|---|---|---|---|
| missing slot | `service_or_analysis_name` | не передаётся | не передаётся | Базовое уточнение услуги/анализа |
| normalized entity | `service_name` | `service_name` | `service_name`, `test_name` | Основной service-facing ключ |
| normalized entity | `test_name` | `test_name` | `test_name`, `service_name` | Для чисто лабораторных сценариев |
| normalized entity | `service_variant` | merge into `service_name_effective` | backend не знает `service_variant` как отдельное поле | Модификатор услуги должен быть склеен до adapter call |
| normalized entity | `doctor_name` | `doctor_name` | `doctor_name`, `doctor`, `fio`, `last_name`, `doctor_last_name` | Для doctor-specific price |
| normalized entity | `doctor_surname` | `doctor_last_name` | `last_name`, `doctor_last_name` | Fallback для doctor-specific price |
| derived flag | `availability_scope=service_delivery` | internal | не передаётся как entity | Для процедур/услуг, привязанных к врачам и оборудованию |
| derived flag | `availability_scope=lab_collection` | internal | не передаётся как entity | Для анализов, которые можно сдать как лабораторный сбор |

### Правила

1. Если есть `service_variant`, adapter должен сначала собрать эффективный service query:
   - `service_name_effective = compose(service_name, service_variant)`
2. В adapter payload лучше передавать уже итоговое `service_name`, а не две независимые сущности.
3. Для doctor-specific price adapter может передавать и врача, и услугу одновременно.
4. `service_name` и `test_name` — не синонимы FT-слоя:
   - `service_name` = медицинская услуга/процедура;
   - `test_name` = лабораторный анализ.
5. Legacy backend aliases могут принимать оба поля, но adapter не должен смешивать их без причины.

## 4.4 Address Domain

Используется для:

- `address_info`

### Translation table

| FT layer | Канон FT | Adapter payload | Legacy backend aliases в `Services.address_info` | Комментарий |
|---|---|---|---|---|
| missing slot | `branch_or_city` | не передаётся | не передаётся | Уточнение локации |
| normalized entity | `branch_name` | `branch` | `region`, `branch`, `company_unit`, `unit`, `city` | Для address tool branch name лучше маппить в `branch` |
| normalized entity | `city` | `city` | `city` | Backend умеет unsupported city fallback |
| normalized entity | `service_name` | `service_name` | `service_name`, `test_name` | Где выполняется медицинская услуга/процедура |
| normalized entity | `test_name` | `test_name` | `test_name`, `service_name` | Где можно сдать лабораторный анализ |
| derived flag | `availability_scope=service_delivery` | internal | не передаётся как entity | Филиалы выполнения процедуры не равны автоматически всем офисам |
| derived flag | `availability_scope=lab_collection` | internal | не передаётся как entity | Анализы можно сдавать шире, чем выполняются процедуры |

### Правила

1. Если указан другой город, adapter не должен пытаться переписать его в Самару.
2. Если есть `branch_name`, приоритетно маппить его в `branch`.
3. Если вопрос про адреса услуги, передавать также `service_name`/`test_name`.
4. Для `service_name` и `test_name` adapter должен различать два режима адресной доступности:
   - `service_delivery` для услуг/процедур;
   - `lab_collection` для анализов.

## 4.5 Test Result Domain

Используется для:

- `test_result_status`

### Translation table

| FT layer | Канон FT | Adapter payload | Legacy backend aliases в `Services.test_result_status` | Комментарий |
|---|---|---|---|---|
| missing slot | `result_surname` | не передаётся | не передаётся | Нужно уточнение |
| missing slot | `result_year_of_birth` | не передаётся | не передаётся | Нужно уточнение |
| missing slot | `result_analysis_code` | не передаётся | не передаётся | User-facing название |
| missing slot | `result_analysis_number` | не передаётся | не передаётся | User-facing название |
| normalized entity | `result_surname` | `surname` | `surname`, `result_surname` | Legacy already accepts `result_surname` in extractor |
| normalized entity | `result_year_of_birth` | `year` | `year` | Дальше идёт в int(year) |
| normalized entity | `result_analysis_code` | `filial` | `filial`, `result_filial` | Legacy naming mismatch, но текущее required поле |
| normalized entity | `result_analysis_number` | `number` | `number`, `order_number`, `order_id` | Сейчас backend приводит к int(number) |

### Правила

1. В FT user-facing слое никогда не называть `result_analysis_code` словом `filial`.
2. Adapter обязан делать явный перевод:
   - `result_analysis_code -> filial`
   - `result_analysis_number -> number`
3. После появления нового backend contract этот mapping должен быть первым кандидатом на миграцию.

## 4.6 Local Indexes Domain

Используется для:

- `main_index_info`
- `news_info`

### Translation table

| FT layer | Канон FT | Adapter payload | Backend contract | Комментарий |
|---|---|---|---|---|
| source/task | `source_mode=local_indexes` | query only | `main_index_info(query, entities)` / `news_info(query, entities)` | Основной сигнал здесь — route/tool selection |
| normalized entity | domain entities | pass-through optional | `entities` mostly opaque | Локальные индексы в основном живут от `query`, не от жёсткого entity contract |

### Правила

1. Для local indexes важнее route selection, чем богатый entity mapping.
2. В adapter достаточно передавать `query` и при необходимости минимум сущностей.

## 4.7 Web Search Domain

Используется для:

- `web_search`

### Translation table

| FT layer | Канон FT | Adapter payload | Backend contract | Комментарий |
|---|---|---|---|---|
| source/task | `source_mode=web` | query only | `WebSearchPort.search(query, entities={})` | Здесь главное route selection, а не entity mapping |

### Правила

1. Для web search entities не являются основным контрактом.
2. Решение о web search должно приниматься до adapter translation слоя.

## 4.8 Self Knowledge / Analysis Domain

Используется для:

- ответы без clinic API;
- synthesis поверх clinic data + web + local indexes + пользовательский контекст.

### Translation table

| FT layer | Канон FT | Adapter payload | Backend contract | Комментарий |
|---|---|---|---|---|
| source/task | `source_mode=self_knowledge` | нет | нет | Adapter не участвует |
| source/task | `task_mode=analyze/recommend/compare` | нет | нет | Это orchestration/synthesis layer |

### Правила

1. Эти режимы не должны пытаться притворяться clinic-data entity translation.
2. Они живут выше adapter layer.

## 5. Domain summary table

| Домен | FT canonical entities | Adapter-facing keys | Derived flags | Основной backend tool |
|---|---|---|---|---|
| Doctor info | `doctor_name`, `doctor_surname`, `specialty`, `service_name`, `branch_name` | `doctor_name`, `doctor_last_name`, `specialty`, `service_name`, `branch` | `availability_scope=doctor_visit` | `doctors_info` |
| Doctor schedule | `doctor_name`, `doctor_surname`, `specialty`, `branch_name`, `city` | `doctor_name`, `doctor_last_name`, `specialty`, `branch_name`, `city` | `availability_scope=doctor_visit` | `doctors_schedule_week` |
| Service/price | `service_name`, `test_name`, `service_variant`, `doctor_name` | `service_name`, `test_name`, `doctor_name`, `doctor_last_name` | `service_delivery` or `lab_collection` | `price_info`, `service_bundle_info`, `test_prepare`, `test_assist` |
| Address | `branch_name`, `city`, `service_name`, `test_name` | `branch`, `city`, `service_name`, `test_name` | `service_delivery` or `lab_collection` | `address_info` |
| Test result | `result_surname`, `result_year_of_birth`, `result_analysis_code`, `result_analysis_number` | `surname`, `year`, `filial`, `number` | `none` | `test_result_status` |
| Local indexes | route/query-first | query + optional entities | `none` | `main_index_info`, `news_info` |
| Web | route/query-first | query | `none` | `web_search` |
| Self knowledge / analysis | source/task-first | `none` | `none` | `none` |

## 6. Что надо будет реализовать в adapter после согласования

1. `DoctorTranslationRules`
2. `ScheduleTranslationRules`
3. `ServiceTranslationRules`
4. `AddressTranslationRules`
5. `ResultTranslationRules`
6. Общую функцию:

```text
translate_ft_entities(domain, entities, missing_slots, source_mode, task_mode) -> adapter_payload
```

7. Отдельный слой sanitation:

- убрать служебные FT-поля;
- не передавать незакрытые user-facing slot names;
- фиксировать applied translation rule в debug trace.
8. Отдельный derived-layer:

- вычислять `availability_scope`;
- отделять `service_delivery` от `lab_collection`;
- не отдавать derived flags в LLM как свободные сущности.
