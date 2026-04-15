# Карта возможностей Free Talk

Статус: draft v0.1  
Дата: 2026-04-12

## 1. Назначение

Этот документ фиксирует текущую карту возможностей FT поверх:

- `src/localragagent/freetalk/*`
- `messengers_router/services.py`
- `agent_logic_2/nayka_api/*`

Цель:

- видеть, что FT уже умеет делать над clinic data;
- понимать, где ограничения задаются не FT, а backend/legacy contract;
- отделять реальные пробелы возможностей от багов orchestration/state management.

Связанные документы:

- [free_talk_data_registry.md](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/docs/free_talk_data_registry.md)
- [free_talk_translation_rules.md](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/docs/free_talk_translation_rules.md)
- [free_talk_processing_pipeline.md](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/docs/free_talk_processing_pipeline.md)
- [free_talk_runtime_architecture.md](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/docs/free_talk_runtime_architecture.md)
- [free_talk_refactor_plan.md](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/docs/free_talk_refactor_plan.md)
- [free_talk_feature_integration_playbook.md](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/docs/free_talk_feature_integration_playbook.md)

## 2. Слои

### 2.1 FT orchestration layer

Отвечает за:

- routing;
- clarification;
- dialog state;
- memory reuse;
- source/task mode;
- tool plan selection.

### 2.2 Adapter layer

Отвечает за:

- перевод FT normalized entities в backend-facing payload;
- нормализацию user-facing терминов в legacy-ключи;
- future migration away from legacy names (`filial`, `number`, etc.).

### 2.3 Backend capability layer

Отвечает за:

- реальный доступ к данным;
- ограничения по городам, кэшу, внешним API и fallback-поведению;
- формат payload, который FT потом рендерит.

## 3. Matrix по основным сценариям

| Сценарий | FT tool / source | Backend entrypoint | Статус | Ограничения / замечания |
|---|---|---|---|---|
| Карточка врача | `doctors_info` | `Services.doctors_info(...)` | Есть | Работает по кэшу врачей, не realtime. Явно отсекает не-самарские площадки из doctor cache. |
| Список врачей по специальности | `doctors_info` | `Services.doctors_info(...)` | Есть | Это основной путь для `все урологи`, `все дерматологи`. |
| Расписание врача | `doctors_schedule_week` | `Services.doctors_schedule_week(...)` | Есть | Realtime, Самара only для записи/расписания. Для других городов backend уже умеет вернуть unsupported/fallback. |
| Цена услуги | `price_info` | `Services.price_info(...)` | Есть | Завязано на Samara retail price region для clinic API части. |
| Инфо по услуге | `service_bundle_info` | `Services.service_bundle_info(...)` | Есть | Комбинирует цены, врачей, подготовку, филиалы. |
| Подготовка к анализу/услуге | `test_prepare` | `Services.test_prepare(...)` | Есть | API-first / cached / fallback logic уже в backend. |
| Подбор анализов | `test_assist` | `Services.test_assist(...)` | Есть | Ищет shortlist анализов. |
| Адреса филиалов / где сдать | `address_info` | `Services.address_info(...)` | Есть | Для city outside Samara backend уже возвращает unsupported fallback. |
| Получение результата анализа | `test_result_status` | `Services.test_result_status(...)` | Есть, но legacy-contract | User-facing договоренность: `фамилия, год рождения, код анализа, номер анализа`. Backend пока ждёт `surname/year/filial/number`. |
| Документы / статьи | `main_index_info` | `Services.main_index_info(...)` | Есть, опционально | Через Meili/local indexes. |
| Новости / акции | `news_info` | `Services.news_info(...)` | Есть, опционально | Через Meili/local indexes. |
| Поиск в интернете | `web_search` | `WebSearchPort.search(...)` | Есть, опционально | Через SearXNG, вне clinic API. |
| Анализ / рекомендация на основе нескольких источников | FT synthesis | Нет отдельного backend tool | Частично есть | Сейчас это orchestration concern, не отдельный tool. Требует source/task policy. |

## 4. Подробности по ограничениям

### 4.1 Doctors info

Источник:

- [services.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/messengers_router/services.py#L6039)

Что умеет:

- искать по ФИО;
- искать по специальности;
- фильтровать по региону/филиалу;
- отдавать карточки врачей из кэша.

Ключевое ограничение:

- backend сам отбрасывает явно не-самарские площадки из кэша врачей.

### 4.2 Doctors schedule

Источник:

- [services.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/messengers_router/services.py#L6189)

Что умеет:

- realtime расписание;
- расписание по врачу;
- расписание по специальности.

Ключевое ограничение:

- для не-самарского города backend возвращает unsupported city fallback.

### 4.3 Address info

Источник:

- [services.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/messengers_router/services.py#L7164)

Что умеет:

- филиалы/адреса;
- адреса по услуге;
- частично doctor-capable branch filtering в appointment mode.

Ключевое ограничение:

- сейчас backend адресной части тоже работает только по Самаре для clinic API/regions API сценариев.

### 4.4 Test result

Источник:

- [services.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/messengers_router/services.py#L6926)
- [api_nayka.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/agent_logic_2/nayka_api/api_nayka.py#L852)

User-facing FT contract:

1. фамилия
2. год рождения
3. код анализа
4. номер анализа

Legacy backend contract:

1. `surname`
2. `year`
3. `filial`
4. `number`

Наблюдение:

- это не semantic match one-to-one;
- это legacy naming, которое FT должен скрывать за adapter layer.

## 5. Capability gaps

Это не обязательно баги. Это текущие пробелы/зоны неопределенности.

### 5.1 Не хватает полного реестра adapter translations

Сейчас точечно понятен mapping для:

- doctor lookup;
- service lookup;
- test result lookup;

Но ещё нет единой таблицы translation rules для всех FT domains.

### 5.2 Не хватает явной capability policy по городам

Сейчас ограничения по городам размазаны по backend functions.

Нужно в будущем формализовать:

- `doctor_info_supported_cities`
- `doctor_schedule_supported_cities`
- `address_info_supported_cities`
- `test_result_supported_cities`

### 5.3 Нет отдельного analysis/recommendation tool

Вопросы вроде:

- `к какому врачу лучше обратиться с натоптышем`
- `какие анализы включают холестерин`
- `кто из врачей подходит для X`

сейчас должны решаться orchestration-слоем FT поверх нескольких источников, а не отдельным backend tool.

Это возможно, но требует:

- устойчивого source selection;
- корректного task mode;
- хорошего post-tool synthesis.

## 6. Что уже можно считать baseline для FT

1. `doctor_info`
2. `doctor_schedule`
3. `price`
4. `prepare`
5. `tests`
6. `test_result`
7. `address`
8. `clinic_documents`
9. `clinic_news`
10. `web_search`

То есть FT уже покрывает основное business ядро, но ещё не стабилизировал:

- multi-turn state transitions;
- city support policy;
- adapter naming consistency;
- recommendation/analyze scenarios.

## 7. Следующие шаги по документации

1. Держать в синхроне [routing_contract.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/src/localragagent/freetalk/routing_contract.py), [orchestrator.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/src/localragagent/freetalk/orchestrator.py) и [adapter.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/src/localragagent/freetalk/adapter.py) с этим matrix.
2. Поддерживать translation rules по доменам как отдельный source-of-truth для adapter layer.
3. Отдельно держать актуальной city-support policy.
4. Расширять matrix новыми capability только через playbook внедрения функций.
