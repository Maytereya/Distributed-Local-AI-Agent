# Карта возможностей Free Talk

Статус: draft v0.2  
Дата: 2026-04-17

## 1. Назначение

Этот документ фиксирует текущую карту возможностей FT поверх:

- `src/localragagent/freetalk/*`
- `messengers_router/services.py`
- `agent_logic_2/nayka_api/*`

Цель:

- видеть, что FT уже умеет делать над clinic data;
- отделять backend-ограничения от ограничений самого FT runtime;
- держать в одном месте не только business capability, но и dialog-control capability.

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
- flow descriptor;
- mixed utterance resolve;
- topic switch / interrupt / reset;
- tool plan selection.

### 2.2 Adapter layer

Отвечает за:

- перевод FT normalized entities в backend-facing payload;
- скрытие legacy naming;
- нормализацию user-facing payload обратно в FT-friendly contract.

### 2.3 Backend capability layer

Отвечает за:

- реальный доступ к данным;
- ограничения по городам, кэшу и внешним API;
- формат payload, который FT потом рендерит.

## 3. Matrix по основным сценариям

| Сценарий | FT tool / source | Backend entrypoint | Статус | Ограничения / замечания |
|---|---|---|---|---|
| Карточка врача | `doctors_info` | `Services.doctors_info(...)` | Есть | Работает по кэшу врачей, не realtime. |
| Список врачей по специальности | `doctors_info` | `Services.doctors_info(...)` | Есть | Основной путь для `все урологи`, `все дерматологи`. |
| Расписание врача | `doctors_schedule_week` | `Services.doctors_schedule_week(...)` | Есть | Realtime, Samara-only для doctor-related clinic API. |
| Запись к врачу | `appointment_policy` + `doctors_schedule_week` / `doctors_info` / `address_info` | Нет отдельного booking tool | Есть, через handoff | FT дособирает запись, подтверждает и завершает `handoff`; прямого booking API пока нет. |
| Цена услуги | `price_info` | `Services.price_info(...)` | Есть | Завязано на Samara retail price region для clinic API части. |
| Инфо по услуге | `service_bundle_info` | `Services.service_bundle_info(...)` | Есть | Комбинирует цены, врачей, подготовку, филиалы. |
| Подготовка к анализу/услуге | `test_prepare` | `Services.test_prepare(...)` | Есть | API-first / cached / fallback logic уже в backend. |
| Подбор анализов | `test_assist` | `Services.test_assist(...)` | Есть | Ищет shortlist анализов. |
| Адреса филиалов / где сдать | `address_info` | `Services.address_info(...)` | Есть | Для city outside Samara backend уже возвращает unsupported fallback. |
| Получение результата анализа | `test_result_status` | `Services.test_result_status(...)` | Есть, но legacy-contract | FT скрывает backend `surname/year/filial/number` за adapter layer. |
| Документы / статьи | `main_index_info` | `Services.main_index_info(...)` | Есть, опционально | Через Meili/local indexes. |
| Новости / акции | `news_info` | `Services.news_info(...)` | Есть, опционально | Через Meili/local indexes. |
| Поиск в интернете | `web_search` | `WebSearchPort.search(...)` | Есть, опционально | Через SearXNG, вне clinic API. |
| Анализ / рекомендация на основе нескольких источников | FT synthesis | Нет отдельного backend tool | Частично есть | Это orchestration concern, а не отдельный tool. |

## 4. Диалоговые control capabilities

| Capability | Owner | Статус | Замечания |
|---|---|---|---|
| `interrupt_current_flow` | `interrupt_policy.py` | Есть | Останавливает текущий flow без полного reset сессии. |
| `hard_reset_session` | `interrupt_policy.py` + `orchestrator.py` | Есть | Полный reset FT-session и выдача нового `session_id`. |
| `topic_switch_confirm` | `interrupt_policy.py` + `orchestrator.py` | Есть | Сохраняет `pending_user_message`, поддерживает re-entry. |
| Mixed utterance resolve | `flow_local_policy.py` | Есть | Работает для `appointment`, `result_lookup`, `clarify`, `confirmation`. |
| `no_preference` handling | `flow_local_policy.py` | Частично есть | Автоподбор уже есть для `appointment`; для остальных flow политика консервативнее. |
| Repeated non-answer escalation | `flow_local_policy.py` | Есть | `appointment -> handoff`, `result_lookup -> stop`, `clarify/confirmation -> stop`. |
| Terminal handoff reset | `orchestrator.py` | Есть | Любой handoff проходит через общий session-reset finalizer. |

## 5. Подробности по ограничениям

### 5.1 Doctors info

Источник:

- [services.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/messengers_router/services.py#L6039)

Что умеет:

- искать по ФИО;
- искать по специальности;
- фильтровать по региону/филиалу;
- отдавать карточки врачей из кэша.

### 5.2 Doctors schedule

Источник:

- [services.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/messengers_router/services.py#L6189)

Что умеет:

- realtime расписание;
- расписание по врачу;
- расписание по специальности.

Ключевое ограничение:

- для не-самарского города backend возвращает unsupported city fallback.

### 5.3 Appointment

Источники в FT:

- [appointment_policy.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/src/localragagent/freetalk/appointment_policy.py)
- [flow_local_policy.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/src/localragagent/freetalk/flow_local_policy.py)
- [orchestrator.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/src/localragagent/freetalk/orchestrator.py)

Что умеет:

- стартовать запись из schedule-context;
- дособирать `date/time/patient_name`;
- поддерживать `interrupt`, `topic switch`, `mixed utterance`, repeated non-answer;
- завершать сценарий через `handoff`.

Ключевое ограничение:

- прямого booking API в FT пока нет; terminal completion записи идёт через handoff к оператору.

### 5.4 Address info

Источник:

- [services.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/messengers_router/services.py#L7164)

Что умеет:

- филиалы/адреса;
- адреса по услуге;
- doctor-capable branch filtering в appointment mode.

### 5.5 Test result

Источники:

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

## 6. Capability gaps

Это не обязательно баги. Это текущие зоны, которые ещё можно дорастить.

### 6.1 Нет прямого booking API

Appointment-domain уже есть, но финал сценария пока handoff-based.

### 6.2 `No preference` ещё не одинаково развит для всех flow

Сейчас strongest implementation есть для `appointment` и части `confirmation`.

### 6.3 Recommendation / synthesis остаётся orchestration concern

Вопросы вроде:

- `к какому врачу лучше обратиться с натоптышем`
- `какие анализы включают холестерин`
- `кто из врачей подходит для X`

не являются отдельным backend tool и по-прежнему требуют хорошего post-tool synthesis.

## 7. Что уже можно считать baseline для FT

1. `doctor_info`
2. `doctor_schedule`
3. `appointment`
4. `price`
5. `prepare`
6. `tests`
7. `test_result`
8. `address`
9. `clinic_documents`
10. `clinic_news`
11. `web_search`
12. dialog-control:
   - `interrupt`
   - `hard reset`
   - `topic switch`
   - mixed utterance resolve
   - terminal handoff reset

## 8. Следующие шаги по документации

1. Держать в синхроне [routing_contract.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/src/localragagent/freetalk/routing_contract.py), [orchestrator.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/src/localragagent/freetalk/orchestrator.py) и [adapter.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/src/localragagent/freetalk/adapter.py) с этой matrix.
2. Отдельно держать актуальной city-support policy.
3. Расширять matrix новыми capability только через playbook внедрения функций.
