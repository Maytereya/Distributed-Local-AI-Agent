# FreeTalk: план рефакторинга управления диалогом

Статус: draft для согласования  
Дата: 2026-06-18  
Основание: ручное тестирование FT в Gradio, remote smoke/eval, ревизия кода `src/localragagent/freetalk`.

## 1. Назначение

Этот документ фиксирует пошаговый план рефакторинга FreeTalk на ближайшую итерацию.

Главная цель:

- повысить гибкость и отзывчивость бота;
- убрать залипание в активных сценариях;
- сделать память сущностей управляемой;
- сохранить уже работающие механики FT;
- не превратить систему в набор хардкодов под отдельные скриншоты.

## 2. Что Сейчас Работает И Должно Быть Сохранено

В коде уже есть несколько ценных механик. Их нельзя снести при рефакторинге.

1. `FreeTalkAdapter`
   - FT-facing entities переводятся в legacy backend entities;
   - backend payload нормализуется обратно в FT payload.

2. `ToolDispatcher` outcome contract
   - `ok`;
   - `not_found`;
   - `tech_unavailable`;
   - `error`;
   - `degraded`;
   - `handoff`;
   - `attachments`.

3. Catalog grounding
   - `match_catalog_doctor`;
   - `match_catalog_service`;
   - exact/fuzzy/miss/unavailable statuses.

4. Fuzzy candidate confirmation
   - кандидат врача;
   - кандидат услуги;
   - подтверждение/отклонение кандидата.

5. Deterministic rendering
   - расписание;
   - цены;
   - результаты анализов;
   - адреса;
   - fallback к LLM только если детерминированный renderer не справился.

6. Post-tool verifier
   - `direct`;
   - `clarify`;
   - `not_found`.

7. Active-flow локальные реакции
   - `uncertainty`;
   - `no_preference`;
   - partial slot answers;
   - mixed utterance.

8. Session memory
   - doctor/service/specialty/branch/date/time/result fields;
   - `last_doctor_name`;
   - appointment windows.

9. Context guard / compaction
   - предупреждение о переполнении контекста;
   - summary;
   - persistent snapshots.

10. Observability
   - `log_event`;
   - `answer_source_trace`;
   - debug fields в API.

## 3. Главные Выявленные Проблемы

### 3.1 Неверный Верхний Порядок Управления Ходом

`interrupt_precheck` формально стоит высоко, но покрывает не все глобальные команды.

Проблемы:

- `оператор`, `переведи на оператора`, `дай оператора` не являются полноценным global intent;
- `ты несешь бред` работает только при active flow;
- context guard может перехватить control-команду, если сам ждет `да/нет`;
- active flow слишком легко протаскивает новую реплику в старый сценарий.

### 3.2 Slot Filling Идет Раньше, Чем Проверка Типа Хода

Фразы вроде `Переведи на оператора` при ожидании `service_or_analysis_name` могут стать `service_name`.

Проблема не в одной фразе, а в отсутствии явного этапа:

```text
что это за ход пользователя?
```

До извлечения слота нужно понять, является ли реплика:

- ответом на слот;
- исправлением слота;
- новой темой;
- просьбой об операторе;
- негативной обратной связью;
- stop/reset-командой;
- обычным новым вопросом.

### 3.3 Result Lookup Завершается Неверным Reprocess

После сбора фамилии, года, кода и номера анализа текущая реплика перекидывается обратно в общий router.

Следствие:

- `Иванов, 1990, код 123, номер 45678` может снова трактоваться как service/test query;
- появляется лишний `service_or_analysis_name`;
- бот просит название анализа, хотя сам просил другие поля.

### 3.4 Память Сущностей Накопительная, Но Не Управляемая

Сейчас память хранит важные сущности, но не различает:

- явно выбранную сущность;
- сущность из широкого списка;
- сущность из первого результата;
- сущность из уточнения;
- старую сущность, которая стала неактуальной.

Типовые ошибки:

- широкая выдача по специальности может перезаписать `last_doctor_name`;
- `Хорошо. А теперь расписание его работы` может извлечь ложного врача из обычной речи;
- новая специальность не всегда очищает старого врача;
- новый филиал/дата/время не очищают зависимые appointment windows;
- нет механизма targeted reset одной сущности без полного сброса диалога.

### 3.5 Web / Local Knowledge / Contact Routing Неполный

Web search работает в основном по явным маркерам:

- `найди в интернете`;
- `поищи`;
- `в сети`;
- `последние новости`;
- `latest/today`.

Проблемы:

- клинический fallback часто перетягивает web-запрос в clinic tools;
- `телефоны клиники`, `контакты`, `как связаться` не выделены как contact/address class;
- backend `address_info` уже умеет возвращать `branches[].phone` и `branches[].work_time`, но FT renderer сейчас печатает в основном только адреса;
- `актуальные рекомендации...` не всегда запускают web;
- local Meili/docs/news имеют узкие триггеры и должны оставаться отдельным legacy/local knowledge контуром.

Принятое продуктовое правило:

- контакты/телефоны/адреса клиники берутся из clinic address/contact API;
- web search не используется для контактов клиники, даже если пользователь формулирует `найди в сети телефон клиники`;
- внешний web search нужен только для явно внешней актуальной информации или после подтвержденного предложения поискать альтернативу вне клиники.

Текущий факт по коду:

- `messengers_router.services.addresses.address_info` берет Самару из live `/regions`;
- при деградации есть fallback на last-good snapshot и seed `messengers_router/data/nonbookable_points.json`;
- seed содержит адреса, телефоны, графики и service flags;
- в новых клинических API-ветках Meili не является источником адресов/цен/врачей/расписания;
- исключение: `prepare.py` еще использует Meili `main_index` как fallback-источник текста подготовки, если API `serviceInfoAll` не дал пригодный ответ.

### 3.6 Schedule / Appointment Filters Применяются Непоследовательно

Проблемы:

- `на выходных` может распознаться, но не отфильтроваться в выводе;
- `А на Ленина утром?` может уйти в unsupported city/handoff;
- дедупликация дат/слотов должна остаться защитной нормализацией, но текущий скрин с двумя `22 июня` не считаем подтвержденным runtime-багом;
- `сегодня вечером` в записи должно сверяться с реальными окнами, а не просто спрашивать удобное время.

### 3.7 Price Search Шумит

`общий анализ крови` находит ОАК, но вместе с ним могут попадать `группа крови`, `медь в крови`, `хром в крови`.

Нужны не исключения под слова, а нормальное ранжирование:

- exact/canonical match;
- family match;
- token overlap;
- weak related matches ниже порога.

## 4. Архитектурный Принцип Рефакторинга

Новый верхний порядок должен быть таким:

```text
1. load session/context
2. classify user turn
3. handle global control/handoff/reset
4. handle context guard only if applicable
5. handle active flow continuation/repair/new-topic
6. route source: clinical/contact/web/local/general
7. ground entities
8. merge/rewrite memory focus and constraints
9. plan tools
10. execute tools
11. post-filter payload
12. post-tool verify
13. render
14. update/reset memory
```

Ключевое изменение:

```text
сначала тип хода пользователя, потом intent/tool/slot.
```

## 5. Целевая Модель Хода Пользователя

Нужно ввести явный объект, условно `TurnDecision`.

Предлагаемая структура:

```python
TurnDecision(
    kind: str,
    confidence: float,
    control_action: str = "",
    source_mode: str = "",
    flow_relation: str = "",
    repair_target: str = "",
    pending_message: str = "",
    continue_message: str = "",
)
```

Канонические значения `kind`:

- `empty`;
- `hard_reset`;
- `handoff_operator`;
- `negative_feedback`;
- `context_guard_answer`;
- `same_flow_slot`;
- `slot_repair`;
- `mixed_flow_and_switch`;
- `new_topic`;
- `clinical`;
- `contact`;
- `web`;
- `local_knowledge`;
- `general`;
- `unknown`.

Канонические значения `flow_relation`:

- `none`;
- `continue`;
- `repair`;
- `switch`;
- `interrupt`;
- `ambiguous`.

Важно:

- это не должен быть LLM-only слой;
- deterministic signals должны закрывать очевидные случаи;
- LLM arbiter нужен только для неоднозначных реплик.

## 6. Целевая Модель Памяти

Сейчас память - это flat dict. Для текущей итерации не обязательно сразу вводить полноценную модель, но нужно приблизиться к такой структуре:

```text
focus:
  type: doctor | specialty | service | result_lookup | appointment | none
  value: canonical value
  source: explicit_user | confirmed_candidate | single_tool_result | broad_tool_result | memory
  confidence: high | medium | low

constraints:
  branch_name
  city
  date/date_from/date_to
  time/time_from/time_to

appointment_context:
  appointment_action
  appointment_windows
  appointment_branch_options

result_context:
  result_surname
  result_year_of_birth
  result_analysis_code
  result_analysis_number
```

Для первой итерации можно реализовать это без миграции Redis-схемы:

- добавить policy-функции поверх текущего dict;
- хранить дополнительные meta-поля только если они реально нужны;
- не ломать текущий `CLINICAL_ENTITY_MEMORY_KEY`.

## 7. Правила Замены И Очистки Сущностей

Нужно оформить dependency policy.

### 7.1 Новый Врач

Если пользователь явно называет нового врача:

- заменить `doctor_name`;
- очистить `doctor_id`;
- очистить `appointment_windows`;
- очистить старые schedule-derived `date/time`, если они пришли от прошлого врача;
- не очищать `branch_name`, если пользователь явно задал филиал в этой же реплике.

### 7.2 Новая Специальность

Если пользователь явно переходит к специальности:

- заменить `specialty`;
- очистить `doctor_name`, если фраза не содержит конкретного врача;
- очистить `doctor_id`;
- очистить `appointment_windows`;
- сохранить branch/date/time constraints, если они явно заданы или релевантны.

### 7.3 Новый Филиал

Если пользователь задает другой филиал:

- заменить `branch_name`;
- очистить `appointment_windows`, если они были получены до смены филиала;
- сохранить doctor/specialty.

### 7.4 Новая Дата Или Время

Если пользователь задает дату/время:

- заменить date/time constraints;
- не выбирать произвольное окно без проверки availability;
- если подходящих окон нет, дать ближайшие альтернативы или честный no-slots.

### 7.5 Any/No Preference

Если пользователь говорит:

- `любой врач`;
- `без разницы`;
- `любой филиал`;
- `когда есть`;

то это не сброс всей памяти, а снятие конкретного constraint.

### 7.6 Негативное Исправление

Фразы:

- `не Дразнин`;
- `Панина не уролог`;
- `не этот врач`;
- `не на Ленина`;

должны приводить к targeted repair, а не к полному reset.

## 8. Пошаговый План Работ

## Этап 0. Baseline И Регрессии

Статус: pending  
Приоритет: P0

Задачи:

1. Зафиксировать набор регрессионных кейсов из скриншотов.
2. Добавить тесты на чистые policy-функции без LLM:
   - operator/handoff;
   - stop;
   - negative feedback;
   - result lookup slot completion;
   - phone/contact query during active flow;
   - doctor anaphora;
   - specialty vs doctor focus;
   - branch/time follow-up;
   - weekend filters.
3. Добавить минимальные integration tests через existing FT test helpers.

Критерий готовности:

- тесты воспроизводят текущие проблемы;
- текущий baseline понятен;
- нет изменений runtime-логики.

Затрагиваемые файлы:

- `tests/test_freetalk_interrupts.py`;
- `tests/test_freetalk_dialog_state.py`;
- `tests/test_freetalk_signal_parsers.py`;
- `tests/test_freetalk_medical_toolloop.py`;
- возможно новый `tests/test_freetalk_turn_policy.py`.

## Этап 1. Turn Classifier / Global Control Layer

Статус: pending  
Приоритет: P0

Задачи:

1. Ввести новый policy-модуль:
   - `src/localragagent/freetalk/turn_policy.py`
   - или `turn_classifier.py`.
2. Описать `TurnDecision` dataclass.
3. Перенести/обобщить туда:
   - hard reset;
   - stop/reset/full session reset;
   - handoff/operator;
   - negative feedback;
   - topic switch;
   - explicit web/local/general source signals.
4. Сделать handoff/operator верхнеуровневым control action.
5. Сохранить LLM interrupt arbiter только для неоднозначных случаев.

Критерий готовности:

- `оператор`, `переведи на оператора`, `дай оператора` не попадают в slot filling;
- `ты несешь бред` не зависит от active flow;
- `оператор` делает terminal handoff и сбрасывает session;
- `стоп`, `сбрось диалог`, `reset`, `hard reset` делают полный session reset без сохранения clinical entity memory;
- old interrupt tests green.

Затрагиваемые файлы:

- `interrupt_policy.py`;
- новый `turn_policy.py`;
- `orchestrator.py`;
- `signal_parsers.py`;
- tests.

## Этап 2. Перестройка Верхнего Orchestrator-Порядка

Статус: pending  
Приоритет: P0

Задачи:

1. В `chat()` сначала классифицировать turn.
2. Global control выполнять до context guard.
3. Context guard обрабатывать только если turn не является control/handoff/hard_reset.
4. Active flow запускать только если `TurnDecision.flow_relation` разрешает continuation/repair.
5. Не позволять active dialog state автоматически делать route clinical для явной новой темы.

Критерий готовности:

- active flow больше не перехватывает очевидные новые темы;
- `найди в сети телефоны клиники...` во время active flow идет в clinic contact/address policy, а не превращается в название услуги;
- context guard не блокирует stop/operator.

Затрагиваемые файлы:

- `orchestrator.py`;
- `routing_policy.py`;
- `flow_local_policy.py`;
- `interrupt_policy.py`;
- tests.

## Этап 3. Result Lookup Terminal Flow

Статус: pending  
Приоритет: P0

Задачи:

1. Убрать `reprocess_current_message=True` после полного сбора result slots.
2. После сбора:
   - `result_surname`;
   - `result_year_of_birth`;
   - `result_analysis_code`;
   - `result_analysis_number`;
   запускать `test_result_status` напрямую через обычный tool loop или отдельный reentry с собранными entities.
3. Разделить missing slots для `test_result_status` и `test_assist`.
4. `service_or_analysis_name` не должен требоваться для проверки результата, если result credentials уже собраны.

Критерий готовности:

- `Проверь результат анализа` -> уточнение result fields;
- `Иванов, 1990, код 123, номер 45678` -> tool result / not_found / tech_unavailable, но не уточнение названия услуги.

Затрагиваемые файлы:

- `flow_local_policy.py`;
- `routing_contract.py`;
- `medical_pretool_policy.py`;
- `medical_toolloop.py`;
- `adapter.py`;
- tests.

## Этап 4. Source Routing: Web / Contact / Local / Clinical / General

Статус: completed 2026-06-18
Приоритет: P1

Задачи:

1. Явный web intent должен иметь приоритет над clinical fallback для внешних, не-контактных запросов:
   - `в интернете`;
   - `в сети`;
   - `погугли`;
   - `актуальные`;
   - `свежие`;
   - `последние`.
2. Ввести/усилить contact/address class:
   - телефоны клиники;
   - как связаться;
   - контакты филиала.
3. Развести:
   - clinic contact/address from API;
   - external web search;
   - no-service external alternative search.
4. Contact/address class должен выигрывать у explicit web markers, если объект запроса - сама клиника.
5. Local Meili/docs/news оставить отдельным `local_knowledge`, но отметить как legacy-контур.
6. Добавить отдельный follow-up:
   - если клиника не оказывает услугу, бот может предложить поискать вне клиники по Самаре;
   - `да`, `давай`, `да поищи`, `действуй` после такого предложения запускают web search;
   - без такого предложения простое `да` не должно запускать внешний поиск.

Критерий готовности:

- `найди в сети телефоны клиники Наука Самара` идет в clinic contact/address policy, не в web и не в service slot;
- `телефон клиники на Ленина` идет в clinic address/contact;
- contact renderer показывает адрес + телефон + график, если они есть в payload;
- `актуальные рекомендации по гипертонии` запускает web/general medical answer, а не clinic tools;
- после ответа `в клинике эта услуга не найдена` бот может предложить внешний поиск по Самаре, а подтверждение запускает web search.

Затрагиваемые файлы:

- `tool_planning.py`;
- `routing_policy.py`;
- `orchestrator.py`;
- `routing_prompting.py`;
- `rendering.py`;
- tests.

## Этап 5. Memory Focus И Правила Обновления Памяти

Статус: pending  
Приоритет: P1

Задачи:

1. Добавить policy-функции:
   - `classify_entity_update`;
   - `apply_entity_update_policy`;
   - `clear_dependent_entities`;
   - `should_remember_doctor_from_payload`.
2. Не перезаписывать `last_doctor_name` из широкого списка врачей.
3. Запоминать врача как focus только если:
   - врач явно назван пользователем;
   - кандидат подтвержден;
   - tool result содержит одного конкретного врача;
   - appointment confirmation закрепила врача.
4. Для специальности не выбирать одного врача как current focus без подтверждения.
5. Ввести targeted reset зависимых ключей.
6. Если пользователь просит `его расписание`, а в фокусе только специальность и список врачей, спросить выбор конкретного врача.

Критерий готовности:

- после списка дерматовенерологов `его расписание` не ссылается на случайного врача;
- бот просит выбрать конкретного врача и показывает короткий список: ФИО, специализация/роль, филиалы, ключевые услуги/направления при наличии данных;
- `расписание Дразнина` -> `а на Ленина утром?` сохраняет Дразнина;
- `теперь уролог` очищает старого врача, но может сохранить филиал/время.

Затрагиваемые файлы:

- `memory_policy.py`;
- `dialog_state.py`;
- `followup_policy.py`;
- `medical_toolloop.py`;
- `appointment_policy.py`;
- tests.

## Этап 6. Appointment Repair И Availability-Aware Flow

Статус: pending  
Приоритет: P1

Задачи:

1. Добавить slot-level repair в appointment:
   - wrong doctor;
   - wrong specialty;
   - wrong branch;
   - wrong date;
   - wrong time;
   - wrong patient name.
2. `Нет` на appointment confirmation не должно всегда превращаться в generic "что изменить".
3. Фразы вида `Панина не уролог` должны менять doctor/specialty constraints.
4. `сегодня вечером` сверять с `appointment_windows`.
5. Если окон нет:
   - сказать, что на выбранный период нет записи;
   - предложить ближайшие альтернативы.

Критерий готовности:

- пользователь может исправить только врача без сброса всего сценария;
- пользователь может изменить только время/филиал;
- appointment flow не выбирает несуществующее окно;
- appointment handoff пока достаточно завершать по ФИО пациента, телефон не обязателен.

Затрагиваемые файлы:

- `appointment_policy.py`;
- tests.

## Этап 7. Schedule Filters / Dedup / Follow-Up

Статус: completed 2026-06-18
Приоритет: P1

Задачи:

1. Проверить прохождение `date_from/date_to/time_from/time_to`:
   - parser;
   - entities;
   - adapter;
   - backend;
   - payload;
   - renderer.
2. Оставить защитную дедупликацию дней и слотов перед рендером.
3. `на выходных` должен давать только ближайшую субботу/воскресенье или честно объяснять отсутствие окон.
4. `А на Ленина утром?` должен merge-ить remembered doctor + branch + time.
5. Unsupported city policy не должна срабатывать на филиал `Ленина`.

Критерий готовности:

- выходные не показывают будни;
- branch follow-up не уходит в handoff.

Затрагиваемые файлы:

- `rendering.py`;
- tests.

## Этап 8. Price Relevance

Статус: completed 2026-06-18
Приоритет: P2

Задачи:

1. Ввести post-ranking для price payload.
2. Развести:
   - exact canonical;
   - family variants;
   - weak token matches.
3. Слабые совпадения показывать ниже или скрывать, если есть сильные.
4. Не делать ручные исключения вроде `медь/хром`.

Критерий готовности:

- `общий анализ крови` показывает ОАК и близкие варианты;
- слабые совпадения не засоряют первые позиции.

Затрагиваемые файлы:

- `adapter.py`;
- `rendering.py`;
- возможно legacy service payload normalization;
- tests.

## Этап 9. Observability И Debug

Статус: pending  
Приоритет: P2

Задачи:

1. Добавить события:
   - `turn_classified`;
   - `global_control_handled`;
   - `active_flow_relation`;
   - `entity_memory_updated`;
   - `entity_memory_cleared`;
   - `focus_changed`;
   - `source_mode_selected`.
2. В debug API добавить ключевые поля, если это не ломает контракт:
   - `turn_kind`;
   - `flow_relation`;
   - `source_mode`;
   - `memory_updates`.

Критерий готовности:

- по логам можно понять, почему бот продолжил flow или сменил тему;
- remote eval failures диагностируются без чтения всего Redis state.

Затрагиваемые файлы:

- `observability.py`;
- `orchestrator.py`;
- `agent_api.py`;
- tests.

## Этап 10. Regression / Deploy / Manual QA

Статус: pending  
Приоритет: P0 для каждого merge-пакета

Локальные проверки:

```bash
./.agent_venv/bin/python -m pytest tests/test_freetalk_*.py -q
./.agent_venv/bin/python tools/check_localragagent_architecture.py
./.agent_venv/bin/python tools/freetalk_quality_gate.py --profile deterministic
git diff --check
```

Remote checks после деплоя:

```bash
./.agent_venv/bin/python tools/freetalk_eval_suite/eval_remote_cases.py \
  --url http://172.16.0.16/v1/freetalk/generate-once \
  --timeout-sec 90
```

Manual Gradio smoke:

1. result lookup;
2. operator during active flow;
3. stop/reset;
4. doctor anaphora;
5. specialty schedule;
6. branch/time follow-up;
7. weekend schedule;
8. appointment repair;
9. contact query;
10. general medical question.

## 9. Предлагаемый Порядок На Сегодня

Если план согласован, начинать лучше так:

1. Этап 0: добавить регрессионные тесты под найденные проблемы.
2. Этап 1: ввести `TurnDecision` и global control layer.
3. Этап 2: переставить верхний порядок в `orchestrator.py`.
4. Этап 3: исправить result lookup terminal flow.
5. Прогнать все локальные проверки.
6. Сделать первый deploy/smoke.
7. После этого переходить к memory focus и appointment repair.

Причина:

- P0-проблемы сейчас ломают базовую отзывчивость;
- пока global turn handling не исправлен, memory/appointment fixes будут наслаиваться на неверный порядок обработки;
- result lookup дает очевидный пользовательский провал и относительно локален.

## 10. Что Не Делаем В Этой Итерации

1. Не переписываем legacy `messengers_router`.
2. Не меняем `agent_logic_2/config.ini`.
3. Не меняем внешний Gradio/UI contract без отдельного решения.
4. Не коммитим сырые реальные prod-диалоги.
5. Не делаем большой rename всего пакета.
6. Не заменяем deterministic policies LLM-промптами.
7. Не добавляем частные if-ы под каждый скриншот, если проблему можно решить уровнем policy.

## 11. Решения Перед Кодом

1. `оператор` в FT должен:
   - сразу terminal handoff + reset session.

2. `телефоны клиники` должны идти:
   - в clinic address/contact API;
   - при деградации допустим last-good snapshot/seed филиалов;
   - web search для телефонов самой клиники не используется.

3. Appointment handoff:
   - достаточно ФИО пациента;
   - телефон не является обязательным слотом на этой итерации.

4. При широком списке врачей по специальности:
   - сохранять specialty focus;
   - не сохранять конкретного врача, пока пользователь его не выбрал;
   - на анафору `его/ее расписание` просить выбрать врача из списка.

5. При `стоп`:
   - очищать все: dialog state, clinical entity memory, flow meta;
   - `сброс`, `reset`, `hard reset` работают так же.

6. Meili:
   - не использовать как источник клинических данных там, где есть API;
   - оставить `main_index_info/news_info` как legacy/local knowledge до отдельного решения;
   - отдельно разобрать `prepare.py`, где Meili еще является fallback для подготовки.

7. Внешний поиск:
   - не основной контур телеграм-бота;
   - допустим для явно внешних актуальных вопросов;
   - допустим как follow-up после отсутствующей услуги в клинике и явного согласия пользователя.

## 12. Definition Of Done Для Рефакторинга

Рефакторинг считается успешным, если:

1. Пользовательские control-команды не попадают в slot filling.
2. Новая тема не залипает в старом flow без подтверждения.
3. Result lookup не требует названия услуги после сбора result credentials.
4. Doctor/specialty memory не перезаписывается широкими списками.
5. Можно заменить одну сущность без полного reset.
6. Web/contact/general запросы маршрутизируются предсказуемо: контакты клиники остаются в clinic API, external web не перехватывает их.
7. Appointment repair работает на уровне слотов.
8. Schedule filters применяются к payload до renderer-а.
9. Все existing FT tests проходят.
10. Новые регрессии из ручных скринов проходят.
11. Remote smoke показывает HTTP 200 без exceptions.
12. Логи позволяют объяснить ключевые решения бота.
