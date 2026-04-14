# Playbook внедрения новых функций Free Talk

Статус: draft v0.1  
Дата: 2026-04-14

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
2. deterministic policy module
3. prompts, если нужно
4. tests

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

### 7.2 Architecture checklist

1. Определён основной класс фичи.
2. Выбрана правильная точка расширения.
3. Новая логика не усиливает перегрузку orchestrator.
4. Sensitive data handling продумана.

### 7.3 Testing checklist

1. Есть unit tests.
2. Есть multi-turn integration test.
3. Если фича user-visible и важная, есть remote eval case.
4. Есть regression-case на failure mode.
5. Протестирован topic shift рядом с новой фичей.

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
