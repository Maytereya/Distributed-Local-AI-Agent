# Playbook внедрения новых функций Free Talk

Статус: draft v0.2  
Дата: 2026-04-17

## 1. Назначение

Этот документ нужен как практическая инструкция для разработчиков FT при добавлении нового функционала.

Главная цель:

- стандартизовать путь внедрения новой возможности;
- не допускать хаотичного роста `agent.py` и связанных модулей;
- сделать каждую новую фичу локализованной по слоям, тестируемой и совместимой с существующей логикой.

Связанные документы:

- [free_talk_runtime_architecture.md](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/docs/free_talk_runtime_architecture.md)
- [free_talk_refactor_plan.md](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/docs/free_talk_refactor_plan.md)
- [free_talk_data_registry.md](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/docs/free_talk_data_registry.md)
- [free_talk_translation_rules.md](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/docs/free_talk_translation_rules.md)
- [free_talk_processing_pipeline.md](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/docs/free_talk_processing_pipeline.md)
- [free_talk_capability_matrix.md](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/docs/free_talk_capability_matrix.md)

## 2. Базовый принцип

Новая функция не должна начинаться с вопроса:

`в какой кусок agent.py это вставить?`

Правильный вопрос:

`какому слою FT принадлежит эта функция?`

Новая логика должна попадать только в тот слой, которому она принадлежит:

1. `routing contract`
2. `policy / orchestration`
3. `adapter`
4. `external port`
5. `deterministic capability module`
6. `rendering`
7. `tests / eval`

## 3. Классы новых функций

Любая новая возможность FT сначала классифицируется.

### 3.1 Backend extension

Это новая возможность поверх уже существующего clinic backend.

Примеры:

- новый сценарий использования `doctors_info`;
- новый способ запроса `test_result_status`;
- новый route к уже существующему `service_bundle_info`.

Типовой путь внедрения:

`routing contract -> adapter -> LegacyServicesPort -> rendering`

### 3.2 External capability

Это новая возможность, для которой нужен новый внешний источник данных или отдельный вычислительный алгоритм.

Примеры:

- поиск ближайшего филиала по адресу пользователя;
- геокодинг;
- интеграция с картами;
- отдельный поиск по локальному хранилищу, не совпадающий с текущим backend.

Типовой путь внедрения:

`routing contract -> capability module -> external port -> rendering`

Важно:

- это не задача adapter-а к `messengers_router`, если capability не использует legacy clinic backend;
- для такого функционала нужен отдельный `port`.

### 3.3 Reasoning / policy feature

Это новая возможность без нового источника данных, но с новой логикой интерпретации или диалога.

Примеры:

- recommendation flow;
- compare flow;
- self-correction policy;
- новая логика topic shift;
- новая политика памяти.

Типовой путь внедрения:

`routing/task policy -> deterministic reasoning policy -> rendering`

## 4. Обязательный workflow внедрения

### 4.1 Шаг 1. Классификация фичи

Для каждой новой задачи разработчик обязан явно зафиксировать:

1. это `backend extension`, `external capability` или `reasoning/policy feature`;
2. какие новые `intent`, `missing_slots`, `entities`, `source_mode`, `task_mode` требуются;
3. нужен ли новый `port`;
4. нужен ли новый `adapter` mapping;
5. какие новые fallback cases появятся.

Без этой классификации код писать не следует.

### 4.2 Шаг 2. Обновление документации контракта

Перед кодом нужно обновить релевантные документы:

1. [free_talk_data_registry.md](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/docs/free_talk_data_registry.md)
2. [free_talk_capability_matrix.md](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/docs/free_talk_capability_matrix.md)
3. [free_talk_translation_rules.md](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/docs/free_talk_translation_rules.md), если есть mapping
4. [free_talk_processing_pipeline.md](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/docs/free_talk_processing_pipeline.md), если меняется flow

Если документация не обновлена, код считается несогласованным с архитектурой.

### 4.3 Шаг 3. Определение точки расширения

Нужно выбрать ровно один основной путь расширения.

#### Если это backend extension

Меняются:

1. routing contract
2. adapter
3. rendering
4. tests

#### Если это external capability

Меняются:

1. routing contract
2. new port
3. new deterministic capability module
4. rendering
5. tests

#### Если это reasoning/policy feature

Меняются:

1. routing/task policy
2. `signal_parsers`, если появляются новые typed user signals
3. deterministic policy module
4. prompts, если нужно
5. tests

### 4.4 Шаг 4. Определение code ownership

Для каждой новой функции нужно явно решить:

#### Что делает код

Кодом должны делаться:

1. state transitions
2. slot/entity canon
3. mapping
4. deterministic filtering
5. fallback interpretation
6. distance / scoring / sorting / nearest-match calculations
7. privacy-sensitive data handling
8. typed parsing и mixed-utterance resolution

#### Что делает LLM

LLM должна делать:

1. intent wording
2. clarify wording
3. synthesis
4. explanation / recommendation, если это разрешено задачей

LLM не должна:

1. считать расстояния;
2. выбирать backend aliases;
3. решать, какие поля хранить в памяти;
4. определять unsupported-city policy;
5. строить payload для внешних сервисов по свободной логике.
6. быть первым уровнем разбора для `interrupt/topic-switch/continuation/correction`, если это можно выразить deterministic parsers.

## 5. Standard extension points

## 5.1 Routing contract

Новая фича может добавить:

1. новый `intent`
2. новые `missing_slots`
3. новые normalized entities
4. новый `clarify_type`, если без него не обойтись

Правило:

- новые user-facing slot names сначала фиксируются в data registry;
- произвольные поля от LLM сразу в runtime не допускаются.

## 5.2 Adapter

Новый код идёт в adapter только если:

1. фича использует clinic backend;
2. нужен перевод FT canonical entities в backend-facing payload;
3. нужен обратный перевод legacy payload в FT-friendly shape.

Adapter не является местом для:

1. web search;
2. геокодинга;
3. карт;
4. произвольных внешних API;
5. общей policy логики диалога.

## 5.3 External port

Новый `port` создаётся, если:

1. нужна новая внешняя dependency;
2. у неё свой network contract;
3. её нужно изолировать от orchestrator и тестировать отдельно.

Примеры:

- `GeoPort`
- `MapsPort`
- `LocationResolvePort`

## 5.4 Deterministic capability module

Такой модуль нужен, если кроме внешнего источника есть отдельная прикладная логика.

Примеры:

- выбор ближайшего филиала;
- scoring нескольких кандидатов;
- фильтрация филиалов по типу доступности;
- ranking услуг по нескольким признакам.

Принцип:

- orchestration не должна считать доменную логику сама;
- она должна вызывать capability module как отдельный слой.

## 5.5 Rendering

Каждая новая функция обязана иметь определённый ответный слой:

1. success render
2. clarify render
3. not-found render
4. source attribution
5. fallback render

Если новая функция не имеет явного rendering strategy, она считается недоделанной.

## 5.6 Signal parser layer

Если новая функция вводит новые typed user signals, она обязана сначала проверить общий parser-layer.

Текущий source-of-truth:

- [signal_parsers.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/src/localragagent/freetalk/signal_parsers.py)

Правила:

1. новый flow не должен писать локальные regex/helper для уже существующих signal types;
2. общие parser-типы вроде `yes/no`, `date/time`, `branch/city`, `person-name`, doctor/service references и controlled mixed split живут только в общем parser-layer;
3. локальные parsers допустимы только для genuinely domain-specific сигналов.

## 5.7 Active flow contract

Если новая функция запускает активный многоходовый flow, она обязана заполнять единый `flow_descriptor`.

Текущий контракт:

1. `flow_active`
2. `flow_kind`
3. `flow_stage`
4. `flow_interruptible`
5. `flow_resume_question`
6. `expected_slots`
7. `flow_non_answer_count`
8. `flow_non_answer_kind`

Практический смысл:

1. `interrupt_policy.py` не должен знать имена всех будущих flow;
2. `flow_local_policy.py` должен понимать continuation/correction/non-answer по общему контракту;
3. новые flow не должны держать stop/topic-switch semantics в ad hoc флагах.

## 5.8 Handoff rule

Если новая функция может завершаться `handoff`-сценарием, это должно считаться terminal state для FT.

Обязательное правило:

1. любой `handoff` в FT завершает текущую FT-сессию;
2. при `handoff` память FT очищается полностью, а не частично;
3. после `handoff` должен выдаваться новый `session_id`;
4. следующее обращение к FT должно идти как новый диалог, без reuse старого summary/history/meta;
5. это правило не должно реализовываться локально внутри одной фичи как уникальная логика, если handoff может появиться и в других сценариях.

Практический смысл:

- если диалог передан оператору, значит FT больше не является владельцем этой сессии;
- если пользователь позже снова пишет боту, это уже новый диалог, а не продолжение старого handoff-кейса.

Следствие для реализации:

- `handoff` должен проходить через единый session-reset path;
- нельзя ограничиваться очисткой только `dialog_state` или только части session memory.
- source-of-truth для этого правила должен быть один общий finalizer в runtime core, а не локальная логика внутри конкретной фичи.

Текущее runtime-правило FT:

1. handoff может приходить как `reply.handoff=True`;
2. handoff может приходить из tool payload как `handoff_required=True`;
3. handoff может приходить из tool payload как непустой `handoff_message`;
4. все эти сигналы должны сводиться к одному общему handoff finalizer в `orchestrator.py`;
5. finalizer обязан вызвать полный `clear_session(session_id)` и выдать новый `next_session_id`.

## 5.9 Interrupt and reset rule

Если новая функция запускает многоходовый flow, она обязана быть совместима с общим FT interrupt/reset layer.

Обязательное различие:

1. `interrupt_current_flow`
   - останавливает только текущую процедуру;
   - очищает `dialog_state` и flow-scoped memory;
   - не очищает весь диалог и не ротирует `session_id`;
2. `hard_reset_session`
   - очищает всю FT-session;
   - выдаёт новый `next_session_id`;
3. `handoff_terminal_reset`
   - также очищает всю FT-session;
   - но является terminal состоянием, а не обычной пользовательской stop-командой.

Следствие для новых фич:

1. общие stop-фразы пользователя не должны обрабатываться локально внутри одной фичи, если в FT уже есть общий interrupt layer;
2. фича может иметь свои domain-specific cancel/reschedule intents, но не должна дублировать глобальный `stop/reset`;
3. новый flow должен иметь корректный `resume_question`, если interrupt-confirm был отклонён пользователем.
4. mixed utterances должны переиспользовать существующий `topic_switch_confirm` path, а не строить отдельный конкурентный router.

## 6. Обязательные запреты

При добавлении новых функций запрещено:

1. добавлять новую доменную логику напрямую в `agent.py`, если для неё можно выделить отдельный слой;
2. хранить новые произвольные поля в `missing_slots`;
3. использовать LLM как вычислительный движок для deterministic задач;
4. тащить новую внешнюю зависимость напрямую в orchestrator;
5. менять сразу routing, memory, adapter и renderer без локальных тестов по каждому слою;
6. вводить backend legacy names как user-facing names.

## 7. Чек-лист внедрения

Перед merge новой функции разработчик должен пройти checklist.

### 7.1 Contract checklist

1. Новый `intent` действительно нужен, а не покрывается существующим.
2. Новые `missing_slots` названы user-facing и канонически.
3. Новые normalized entities не дублируют старые по смыслу.
4. Для backend integration прописан явный mapping.
5. Если есть `handoff`, для него определён session-reset path.

### 7.2 Architecture checklist

1. Определён основной класс фичи.
2. Выбрана правильная точка расширения.
3. Новая логика не усиливает перегрузку orchestrator.
4. Sensitive data handling продумана.
5. Если добавлен новый active flow, он заполняет `flow_descriptor`.
6. Если добавлены новые typed signals, они либо переиспользуют общий parser-layer, либо явно обоснованы как domain-specific.

### 7.3 Testing checklist

1. Есть unit tests.
2. Есть multi-turn integration test.
3. Если фича user-visible и важная, есть remote eval case.
4. Есть regression-case на failure mode.
5. Протестирован topic shift рядом с новой фичей.
6. Если есть `handoff`, есть тест на полный reset FT-session и ротацию `session_id`.
7. Если фича многоходовая, есть тесты на `interrupt/reset`, `no_preference` и mixed utterance рядом с ней.

## 8. Пример: nearest branch by user address

Постановка:

- пользователь спрашивает, где ближайший к нему офис или заборный пункт;
- бот уточняет текущий адрес;
- дальше система находит ближайшую точку клиники и возвращает адрес.

### 8.1 Классификация

Это `external capability`, а не `adapter feature`.

Причина:

- здесь нужен географический расчёт;
- это не просто перевод FT entities в legacy clinic backend;
- для геокодинга или карт нужен отдельный внешний контракт.

### 8.2 Что надо добавить

В data contract:

1. `intent = nearest_branch`
2. `missing_slot = user_address`
3. normalized entities:
   - `user_address`
   - `user_lat`
   - `user_lon`

В коде:

1. новый `GeoPort`
2. новый capability module, например `branch_locator.py`
3. deterministic selection logic:
   - geocode user address
   - resolve branch coordinates
   - compute nearest branch
   - rank candidates

В rendering:

1. nearest branch answer
2. clarify when address is missing
3. fallback when geocoding failed

### 8.3 Что не надо делать

Не надо:

1. пихать вычисление расстояния в LLM prompt;
2. вшивать карты прямо в orchestrator;
3. добавлять географические вычисления в adapter к `messengers_router`;
4. сохранять полный адрес пользователя в долговременной памяти без явного основания.

## 9. Правило рутинного расширения

Если архитектура соблюдается, новая функция должна добавляться так:

1. документируется как capability
2. получает свой contract
3. получает свой boundary layer
4. получает deterministic module
5. получает renderer
6. получает tests

Если новая функция требует “ещё 200 строк в orchestrator”, значит точка расширения выбрана неверно.

## 10. Practical merge rule

Новый функционал FT допускается к merge только если на него можно ответить:

1. где живёт его contract;
2. где живёт его deterministic logic;
3. где живёт его external boundary;
4. где живёт его rendering;
5. какими тестами он защищён.

Если один из этих пунктов отсутствует, функция ещё не доведена до архитектурно безопасного состояния.
