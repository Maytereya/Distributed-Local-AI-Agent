# Реестр данных Free Talk

Статус: draft v0.1  
Дата: 12.04.2026

## 1. Зачем нужен этот документ

Этот документ фиксирует три разных слоя данных в FT (Free talk conversation mode):

1. `Missing Slots Layer`  
   Чего именно не хватает, чтобы продолжить диалог и задать корректное уточнение.
2. `Normalized Entities Layer`  
   Какие нормализованные данные уже собраны в рамках FT-диалога.
3. `Adapter/Backend Mapping Layer`  
   Как данные FT переводятся в контракт адаптера (является частью пакета FT) и дальше в legacy backend `messengers_router/services.py`.

Главная цель:

- не смешивать `missing_slots` с реальными `entities`;
- не пропускать в state произвольные названия слотов от LLM;
- явно описать перевод из human-facing терминов FT в legacy поля backend.

Связанные документы:

- [free_talk_capability_matrix.md](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/docs/free_talk_capability_matrix.md)
- [free_talk_translation_rules.md](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/docs/free_talk_translation_rules.md)
- [free_talk_processing_pipeline.md](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/docs/free_talk_processing_pipeline.md)
- [free_talk_runtime_architecture.md](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/docs/free_talk_runtime_architecture.md)
- [free_talk_refactor_plan.md](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/docs/free_talk_refactor_plan.md)

## 2. Принципы

### 2.1 Missing slots

`missing_slots` — это не контейнер данных, а только короткие ярлыки нехватки информации.

Пример:

```json
{
  "missing_slots": ["specialty"],
  "entities": {}
}
```

После ответа пользователя:

```json
{
  "missing_slots": [],
  "entities": {
    "specialty": "уролог"
  }
}
```

### 2.2 Normalized entities

`entities` — это уже собранные и нормализованные данные, которые можно передавать в tools.

### 2.3 Mapping

`mapping` — это не свободная динамическая структура.

Это детерминированный перевод:

- из user-facing имен FT;
- в adapter-facing поля;
- затем в legacy backend ключи.

То есть:

`FT entity -> adapter entity -> backend argument`

## 3. Missing Slots Layer

Утверждаемый базовый список:

1. `doctor_name`
2. `specialty`
3. `service_or_analysis_name`
4. `branch_or_city`
5. `date`
6. `time`
7. `result_surname`
8. `result_year_of_birth`
9. `result_analysis_code`
10. `result_analysis_number`

### 3.1 Семантика missing slots

#### `doctor_name`

Не хватает данных, чтобы однозначно выбрать конкретного врача.

Примеры:

- `Дай информацию по врачу Иванову`
- `Покажи расписание Дразнина`

Если фамилия неоднозначна, FT уточняет имя/отчество до уровня, достаточного для выбора врача.

#### `specialty`

Не хватает специальности врача.

Примеры:

- `Покажи всех врачей`
- `К кому обратиться с натоптышем?`

После уточнения slot закрывается значением вроде `уролог`, `дерматовенеролог`, `хирург`.

#### `service_or_analysis_name`

Не хватает названия услуги, процедуры или анализа.

Примеры:

- `Сколько стоит это исследование?`
- `Как подготовиться к анализу?`

#### `branch_or_city`

Не хватает локационного фильтра.

Используется для:

- расписания врача;
- адресов филиалов;
- доступности услуги по филиалам.

Особое правило FT:

- для данных врачей/записи clinic API считается доступным только по Самаре;
- если пользователь спрашивает про другой город, FT должен честно сообщить это, а не продолжать doctor-schedule flow как будто данные есть.

#### `date`

Не хватает даты или диапазона дат.

Примеры:

- `на следующей неделе`
- `на 15 апреля`
- `в ближайшие дни`

#### `time`

Не хватает времени суток или временного интервала.

Примеры:

- `утром`
- `после 18:00`
- `во второй половине дня`

#### `result_surname`

Не хватает фамилии пациента для получения результата анализа.

#### `result_year_of_birth`

Не хватает года рождения пациента для получения результата анализа.

#### `result_analysis_code`

Не хватает кода анализа в пользовательском формате.

Пример:

- `Бг`

Важно:

- в FT это называется именно `код анализа`, потому что так это выглядит в user-facing контракте;
- внутри legacy backend это сейчас уходит в поле `filial`.

#### `result_analysis_number`

Не хватает номера анализа в пользовательском формате.

Пример:

- `1234`

## 4. Normalized Entities Layer

Базовый список нормализованных сущностей FT:

### 4.1 Doctor domain

1. `doctor_name`
2. `doctor_surname`
3. `specialty`

### 4.2 Service / analysis domain

1. `service_name`
2. `test_name`
3. `service_variant`

#### `service_variant`

Это модификатор базовой услуги, который обычно появляется в коротком follow-up сообщении.

Примеры:

- `с наркозом`
- `под седацией`
- `без контраста`

Важно:

- `service_variant` не считается самостоятельной услугой;
- он дополняет уже выбранную `service_name`;
- на текущем этапе FT склеивает `remembered_service + service_variant` и пытается повторно разрешить итоговую формулировку через каталог.

### 4.3 Location / schedule filters

1. `branch_name`
2. `city`
3. `date`
4. `date_from`
5. `date_to`
6. `time`
7. `time_from`
8. `time_to`

#### Пояснение по локации

На уровне FT canonical entities intentionally храним только:

- `branch_name` — конкретный филиал/площадка;
- `city` — городской фильтр.

Низкоуровневые или legacy-варианты вроде `branch` и `region` остаются adapter/backend-слоем и не должны раздувать основной FT-контракт.

### 4.4 Test result domain

1. `result_surname`
2. `result_year_of_birth`
3. `result_analysis_code`
4. `result_analysis_number`

## 5. Source / Task Flags

Это не missing slots и не entities. Это отдельный слой управления ходом ответа.

### 5.1 `source_mode`

Возможные значения:

1. `clinic_api`
2. `local_indexes`
3. `web`
4. `self_knowledge`
5. `mixed`

### 5.2 `task_mode`

Возможные значения:

1. `lookup`
2. `clarify`
3. `explain`
4. `recommend`
5. `compare`
6. `analyze`

Примеры:

- `Поищи это в интернете` -> `source_mode=web`, `task_mode=lookup`
- `Что ты сам об этом думаешь?` -> `source_mode=self_knowledge`, `task_mode=explain|analyze`
- `К какому врачу лучше обратиться с натоптышем?` -> `source_mode=mixed`, `task_mode=recommend`
- `Какие анализы есть на холестерин?` -> `source_mode=clinic_api`, `task_mode=lookup`

## 6. Adapter / Backend Mapping Layer

### 6.1 Общий принцип

FT работает с user-facing именами.  
Адаптер переводит их в legacy-имена backend.

### 6.2 Mapping для результатов анализов

#### FT user-facing entities

1. `result_surname`
2. `result_year_of_birth`
3. `result_analysis_code`
4. `result_analysis_number`

#### Adapter-facing translation

```json
{
  "surname": "<result_surname>",
  "year": "<result_year_of_birth>",
  "filial": "<result_analysis_code>",
  "number": "<result_analysis_number>"
}
```

#### Legacy backend usage

Сейчас `messengers_router/services.py` вызывает:

- `test_result_status(query, entities)`
- внутри `_extract_result_query_fields(...)`
- затем `site_result_for_patient(surname, year, filial, number, ...)`

Ссылки:

- [services.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/messengers_router/services.py#L6926)
- [services.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/messengers_router/services.py#L2124)
- [api_nayka.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/agent_logic_2/nayka_api/api_nayka.py#L852)

### 6.3 Mapping для локации

#### FT user-facing slot

- `branch_or_city`

#### Нормализованные FT entities 
- `branch_name`
- `city`

#### Adapter behavior

1. Если пользователь указал филиал Самары:
   переводим в `branch_name`.
2. Если пользователь указал `Самара`:
   переводим в `city=Самара`.
3. Если пользователь город не указал:
   по doctor-related clinic API считаем рабочим default `city=Самара`, но не обязаны записывать это как явную user entity.
4. Если пользователь указал другой город:
   не пытаемся слепо звать doctor scheduling API;
   FT должен честно сообщить, что данные clinic API по врачам/записи ограничены Самарой.

## 7. Что НЕ должно попадать в Missing Slots

Следующие названия считаются слишком низкоуровневыми или legacy-специфичными для FT-слоя и не должны жить в `missing_slots`:

1. `full_name`
2. `doctor_id`
3. `branch_name`
4. `date_from`
5. `date_to`
6. `time_from`
7. `time_to`
8. `filial`
9. `number`

Они могут существовать:

- в normalized entities;
- в adapter mapping;
- в backend contract;

но не как user-facing clarifying slots.

## 8. Открытые вопросы

1. Стоит ли для doctor disambiguation в FT использовать отдельную сущность `doctor_surname`, а не только `surname`?
Решение: да. Используем `doctor_surname` и `result_surname` как разные доменные сущности.
2. Следует ли backend `test_result_status` со временем переименовать с legacy пары `filial/number` в человеко-понятные `analysis_code/analysis_number`?
Решение: да, но только после отдельной аккуратной миграции adapter/backend contract.
3. Нужен ли один boolean-флаг вроде `not_Samara_supported` для городовых ограничений?
Решение: нет. Нужна capability matrix по доменам/инструментам, потому что поддержка города зависит от сценария:
- doctor schedule;
- doctor info;
- address info;
- results;
- service lookup.
4. У нас пока нет полного и исчерпывающего списка возможностей извлечения бизнес-данных из `services.py` / `api_nayka.py` для построения полной карты FT.
Решение: да. Для этого введен отдельный документ:
- [free_talk_capability_matrix.md](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/docs/free_talk_capability_matrix.md)

## 9. Mapping examples

### 9.1 Результаты анализов

Пользователь пишет:

`Иванов, 1989, Бг, 1234`

FT normalized entities:

```json
{
  "result_surname": "Иванов",
  "result_year_of_birth": "1989",
  "result_analysis_code": "Бг",
  "result_analysis_number": "1234"
}
```

Adapter translation:

```json
{
  "surname": "Иванов",
  "year": "1989",
  "filial": "Бг",
  "number": "1234"
}
```

Legacy backend:

- `test_result_status(query, entities)`
- `site_result_for_patient(surname, year, filial, number, ...)`

### 9.2 Смена темы с врача на специальность

История:

1. `Покажи расписание Дразнина`
2. `А выведи всех урологов`

Корректное поведение FT:

- не тащить старый `doctor_name` в новый specialty-query;
- сбросить doctor-specific clarify-state;
- заполнить:

```json
{
  "missing_slots": [],
  "entities": {
    "specialty": "уролог"
  }
}
```

а затем вызвать `doctors_info`, а не `doctors_schedule_week`.
