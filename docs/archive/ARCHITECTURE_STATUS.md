# Messengers Router: Architecture Status (v5)

Обновлено: **2026-04-10**

Документ отражает текущее состояние `messengers_router` после серии правок по `APPOINTMENT`, `PRICE`, `ADDRESS`, `PREPARE` и интеграции `priceUnits`.

## 1) Кратко: что изменилось

Базовая архитектура не переписана, но логика работы бота заметно изменилась.

Что осталось прежним:
- основной каркас: `router -> services -> renderer/response_builder`;
- состояние по-прежнему хранится через `state.last_entities` и `memory.pending`;
- внешние API-контракты для основного чата не ломались.

Что реально изменилось:
- `APPOINTMENT` стал более строгим slot-flow и меньше выпадает в общий `NLU`;
- `PRICE` стал сильнее опираться на каталог и data-driven логику вместо слабых текстовых эвристик;
- `ADDRESS` по процедурам теперь использует `care setting` через `priceUnits`;
- появился bounded-flow для mixed `PRICE`-запросов с несколькими услугами;
- patient-facing ответы по цене стали группироваться по формату оказания и адресу;
- doctor/service matching стал строже, чтобы уменьшить ложные совпадения.

Итог: архитектура осталась эволюционной, но поведение бота стало более детерминированным и контекстно-устойчивым.

## 2) Актуальная схема работы

### 2.1 Router layer

`/Users/maxten/Dev/Distributed-Local-AI-Agent2/messengers_router/router.py`

Роутер сейчас работает в таком порядке:

1. читает `pending` и session context;
2. применяет узкие pre-NLU handlers;
3. запускает primary NLU / rule-based guards / topic registry;
4. строит `Plan`;
5. выполняет plan через `services`;
6. собирает patient-facing ответ через `response_builder` / `renderer`.

Важно:
- это уже не “каждый ход заново классифицируется без памяти”;
- несколько сценариев теперь удерживаются до общего NLU, если есть активный transaction/pending context.

### 2.2 Service layer

`/Users/maxten/Dev/Distributed-Local-AI-Agent2/messengers_router/services.py`

Это главный доменный слой. Здесь сейчас сосредоточена большая часть продуктовой логики:

- `price_info(...)`
- `service_bundle_info(...)`
- `address_info(...)`
- `doctors_info(...)`
- `doctors_schedule_week(...)`
- `prepare_info(...)`
- `test_result_status(...)`

Ключевая тенденция:
- больше логики переведено в deterministic/data-driven `services`;
- меньше reliance на случайный текстовый матч в верхнем слое.

### 2.3 Presentation layer

`/Users/maxten/Dev/Distributed-Local-AI-Agent2/messengers_router/renderer.py`  
`/Users/maxten/Dev/Distributed-Local-AI-Agent2/messengers_router/response_builder.py`

Здесь теперь не просто “форматирование текста”.

Фактически presentation layer отвечает за:
- финальную patient-facing сборку ответа;
- сохранение части transient context;
- компактный, но информативный вывод family-price, doctor list, care-setting и follow-up подсказок.

## 3) Что поменялось по интентам

### 3.1 APPOINTMENT

Подтвержденные изменения:

- active `APPOINTMENT` flow лучше удерживается как slot-driven сценарий;
- короткие ответы вроде фамилии врача, даты, `да/нет` меньше проваливаются в `OTHER`;
- reschedule-flow стабилизирован;
- hard reset / cancel confirm работает предсказуемее;
- ambiguous `appointment_action` (`отменить или перенести?`) теперь удерживается отдельным pending-handler’ом.

Практический смысл:
- сценарии записи стали меньше зависеть от случайной пере-классификации каждого нового сообщения;
- бот стал ближе к state machine внутри активной записи.

Связанные модули:
- `/Users/maxten/Dev/Distributed-Local-AI-Agent2/messengers_router/router.py`
- `/Users/maxten/Dev/Distributed-Local-AI-Agent2/messengers_router/flow_policy.py`
- `/Users/maxten/Dev/Distributed-Local-AI-Agent2/messengers_router/appointment_flow_guard.py`

### 3.2 PRICE

Это самый существенно изменившийся блок.

Подтвержденные изменения:

- catalog-grounded resolution для price-запросов;
- cleaner selection эффективного `service_name`;
- family-mode для запросов с несколькими родственными вариантами;
- группировка family-ответа по формату оказания и адресу;
- bounded-flow для compound/multi-service `PRICE`;
- использование `priceUnits` как доменного сигнала, а не просто поля из кэша.

Практический смысл:
- бот реже берет случайную строку прайса;
- бот может честно показать несколько допустимых вариантов;
- mixed query вида `УЗДГ + ЛПНП + филиал` больше не схлопывается в заведомо неверный single-answer.

Связанные модули:
- `/Users/maxten/Dev/Distributed-Local-AI-Agent2/messengers_router/services.py`
- `/Users/maxten/Dev/Distributed-Local-AI-Agent2/messengers_router/renderer.py`
- `/Users/maxten/Dev/Distributed-Local-AI-Agent2/messengers_router/response_builder.py`

### 3.3 ADDRESS

Подтвержденные изменения:

- для процедур/операций `ADDRESS` сначала пытается взять адреса из `care setting`;
- fallback в общий список филиалов теперь не является первым путем для процедурных кейсов;
- это убрало часть старых багов, когда бот выдавал “все филиалы”, хотя услуга относится к конкретному типу оказания.

Связанные модули:
- `/Users/maxten/Dev/Distributed-Local-AI-Agent2/messengers_router/services.py`

### 3.4 PREPARE

Подтвержденные изменения:

- short follow-up после `PREPARE` стал удерживаться лучше;
- кейсы вида “Как подготовиться к ...?” -> “Вульвоскопия” больше не должны уезжать в другой intent так легко.

Что важно:
- `prepare_wrap` по-прежнему partly LLM-dependent;
- жесткого архитектурного переписывания этого слоя пока не было.

## 4) Новый endpoint и новые данные

### 4.1 Добавлен `priceUnits`

Новый endpoint:
- `api/v1/site/priceUnits`

Реализация:
- `/Users/maxten/Dev/Distributed-Local-AI-Agent2/agent_logic_2/nayka_api/api_price.py`

Добавлено:
- `fetch_price_units()`
- `update_price_units()`
- `load_price_units()`
- построение индекса `priceUnitId -> node`
- резолвер контекста оказания услуги по `priceUnitId`

### 4.2 Как используется `priceUnits`

Через `priceUnits` бот теперь определяет корневой тип оказания услуги:

- `146` -> поликлиника
- `311` -> дневной стационар
- `312` -> круглосуточный стационар

Далее это маппится на адрес:

- `146/311` -> `г. Самара, пр. Ленина, 5`
- `312` -> `г. Самара, ул. Ново-Садовая, 106, кор. 82`

Это используется в:
- `PRICE`
- `ADDRESS`
- части procedure-related сценариев записи

### 4.3 Что это дало продуктово

- у стационарных процедур появился детерминированный адресный контекст;
- цена и адрес теперь можно показывать вместе с форматом оказания;
- бот начал отличать “поликлиника / дневной стационар / круглосуточный стационар” как доменные сущности.

## 5) Новые bounded-flow сценарии

### 5.1 Compound PRICE flow

Добавлен узкий сценарий для mixed `PRICE`-запросов с несколькими услугами.

Пример класса запросов:
- `УЗДГ сосудов шеи`
- `и сдать кровь на ЛПНП`
- `по адресу Победы 83`
- `какова стоимость`

Поведение:
- бот не пытается больше уверенно слепить одну услугу из нескольких;
- если видит надежный compound case, возвращает controlled clarify;
- follow-up `да`, `ЛПНП`, другой вопрос обрабатываются детерминированно.

Это не общий multi-service planner, а узкий безопасный flow.

### 5.2 Pending handlers до NLU

В роутере появились дополнительные narrow pre-NLU handlers:

- `appointment_action` pending
- compound price pending
- short prepare follow-up

Идея:
- не пускать короткие контекстные ответы сразу в общий классификатор;
- сначала попытаться трактовать их как ответ на текущий pending state.

## 6) Что изменилось в doctor matching

Подтвержденные изменения:

- stricter matching по фамилии врача;
- точнее фильтруются ложные совпадения;
- в price/service-bundle для врача может показываться короткая релевантная специальность в зависимости от запроса;
- doctor-facing и specialty-facing логика стала аккуратнее разделяться.

Практический смысл:
- меньше ложных совпадений вроде `Иванов` -> `Иванова`;
- меньше путаницы, когда у врача несколько ролей / подразделений;
- в patient-facing выдаче проще понять, кто именно из однофамильцев нужен.

## 7) Что НЕ поменялось архитектурно

Чтобы не было ложного впечатления о rewrite:

- scoped-state (`profile_scope / focus_scope / transaction_scope`) все еще не внедрен как отдельная модель;
- `state.last_entities` и `memory.pending` по-прежнему основа состояния;
- полного universal conflict-resolver слоя пока нет;
- `PREPARE-wrap` не переведен в fully deterministic режим;
- общий слой не разделен на новый набор доменных подагентов или orchestrator-подсистем.

То есть это все еще эволюция текущего `messengers_router`, а не новая архитектура с нуля.

## 8) Где смотреть коллеге в первую очередь

### 8.1 Router / flow

- `/Users/maxten/Dev/Distributed-Local-AI-Agent2/messengers_router/router.py`
- `/Users/maxten/Dev/Distributed-Local-AI-Agent2/messengers_router/flow_policy.py`
- `/Users/maxten/Dev/Distributed-Local-AI-Agent2/messengers_router/appointment_flow_guard.py`

### 8.2 Domain logic

- `/Users/maxten/Dev/Distributed-Local-AI-Agent2/messengers_router/services.py`

### 8.3 Presentation / response assembly

- `/Users/maxten/Dev/Distributed-Local-AI-Agent2/messengers_router/renderer.py`
- `/Users/maxten/Dev/Distributed-Local-AI-Agent2/messengers_router/response_builder.py`

### 8.4 Nayka API adapters and caches

- `/Users/maxten/Dev/Distributed-Local-AI-Agent2/agent_logic_2/nayka_api/api_price.py`
- `/Users/maxten/Dev/Distributed-Local-AI-Agent2/agent_logic_2/nayka_api/api_service_info.py`
- `/Users/maxten/Dev/Distributed-Local-AI-Agent2/agent_logic_2/nayka_api/api_nayka.py`

### 8.5 История price-дефектов и их фиксов

- `/Users/maxten/Dev/Distributed-Local-AI-Agent2/messengers_router/eval_suite/PRICE_BUG_TRACKER.md`

## 9) Подтвержденные результаты на последних ревизиях

По последним прогоном перед локальными незакоммиченными правками было подтверждено:

- `stage1` — зеленый;
- `stage3 appointment flow` — зеленый;
- `stage4 reliability` — зеленый;
- `stage5 golden corpus` — доведен до полного прохода;
- `PRICE` и `ADDRESS` стали значительно стабильнее на критичных кейсах.

Важно:
- часть самых последних точечных фиксов может находиться только в локальном workspace до следующего коммита/деплоя;
- при передаче работы коллеге нужно сверить `git status` и понять, какие изменения уже в `origin/release`, а какие только локальные.

## 10) Практический вывод для следующего разработчика

Если нужно быстро войти в систему, правильная ментальная модель такая:

1. Это не stateless classifier.
2. Это не полностью rule-only система.
3. Это не LLM-first orchestration.

Это гибридная система:

- `router` удерживает контекст и pending-flow;
- `services` принимают доменные решения по данным;
- `renderer` отвечает за patient-facing интерпретацию;
- `priceUnits` и кэши Nayka API стали важной частью source of truth.

Поэтому новые правки лучше делать так:

- сначала искать, можно ли усилить `services.py`;
- потом, если нужно, добавлять узкий pre-NLU handler;
- и только в последнюю очередь трогать общий classifier behavior.

## 11) Обязательные проверки перед следующим deploy

1. `venv/bin/python -m pytest -q`
2. `python3 /Users/maxten/Dev/Distributed-Local-AI-Agent2/messengers_router/scripts/check_architecture_imports.py`
3. `python3 /Users/maxten/Dev/Distributed-Local-AI-Agent2/messengers_router/scripts/check_eval_coverage.py`
4. `bash /Users/maxten/Dev/Distributed-Local-AI-Agent2/messengers_router/eval_suite/run_remote_eval.sh --url <server>/api/messenger-generate-once`
