# Реестр данных Free Talk

Статус: draft v0.2  
Дата: 17.04.2026

## 1. Зачем нужен этот документ

Этот документ фиксирует четыре разных слоя данных в FT:

1. `Missing Slots Layer`
2. `Normalized Entities Layer`
3. `Flow / Dialog State Layer`
4. `Adapter / Backend Mapping Layer`

Главная цель:

- не смешивать `missing_slots` с реальными `entities`;
- не смешивать runtime-state активного flow с пользовательскими сущностями;
- явно описать перевод из human-facing терминов FT в adapter/backend contract.

Связанные документы:

- [free_talk_capability_matrix.md](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/docs/free_talk_capability_matrix.md)
- [free_talk_translation_rules.md](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/docs/free_talk_translation_rules.md)
- [free_talk_processing_pipeline.md](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/docs/free_talk_processing_pipeline.md)
- [free_talk_runtime_architecture.md](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/docs/free_talk_runtime_architecture.md)
- [free_talk_refactor_plan.md](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/docs/free_talk_refactor_plan.md)

## 2. Принципы

### 2.1 Missing slots

`missing_slots` — это не контейнер данных, а только ярлыки нехватки информации для user-facing уточнения.

### 2.2 Normalized entities

`entities` — это уже собранные и нормализованные данные, которые можно передавать в tools или использовать в orchestration.

### 2.3 Flow state

`flow descriptor` — это runtime-state активного сценария. Это не `missing_slots` и не `entities`.

### 2.4 Mapping

`mapping` — это детерминированный перевод:

`FT entity -> adapter entity -> backend argument`

## 3. Missing Slots Layer

Утверждаемый базовый список:

1. `doctor_name`
2. `specialty`
3. `service_or_analysis_name`
4. `branch_or_city`
5. `date`
6. `time`
7. `appointment_action`
8. `patient_name`
9. `result_surname`
10. `result_year_of_birth`
11. `result_analysis_code`
12. `result_analysis_number`

### 3.1 Семантика missing slots

#### `doctor_name`

Не хватает данных, чтобы однозначно выбрать конкретного врача.

#### `specialty`

Не хватает специальности врача.

#### `service_or_analysis_name`

Не хватает названия услуги, процедуры или анализа.

#### `branch_or_city`

Не хватает локационного фильтра.

Используется для:

- расписания врача;
- адресов филиалов;
- доступности услуги по филиалам.

Особое правило FT:

- для doctor-related clinic API данные считаются поддерживаемыми только по Самаре;
- если пользователь спрашивает про другой город, FT должен честно сообщить это, а не продолжать doctor flow как будто данные есть.

#### `date`

Не хватает даты или диапазона дат.

#### `time`

Не хватает времени суток или временного интервала.

#### `appointment_action`

Не хватает типа действия в appointment-flow.

Примеры:

- `записаться`
- `перенести`
- `отменить`

Важно:

- это slot только для сценария записи;
- вне appointment-domain не должен появляться в `missing_slots`.

#### `patient_name`

Не хватает ФИО пациента для записи.

Важно:

- это user-facing slot appointment-domain;
- это не doctor name и не произвольная person entity.

#### `result_surname`

Не хватает фамилии пациента для получения результата анализа.

#### `result_year_of_birth`

Не хватает года рождения пациента для получения результата анализа.

#### `result_analysis_code`

Не хватает кода анализа в пользовательском формате.

Важно:

- в FT это называется именно `код анализа`;
- внутри legacy backend это сейчас уходит в поле `filial`.

#### `result_analysis_number`

Не хватает номера анализа в пользовательском формате.

## 4. Normalized Entities Layer

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

Важно:

- `service_variant` не считается самостоятельной услугой;
- он дополняет уже выбранную `service_name`.

### 4.3 Location / schedule filters

1. `branch_name`
2. `city`
3. `date`
4. `date_from`
5. `date_to`
6. `time`
7. `time_from`
8. `time_to`

На уровне FT canonical entities храним только:

- `branch_name`
- `city`

Legacy-варианты вроде `branch` и `region` остаются adapter/backend-слоем.

### 4.4 Test result domain

1. `result_surname`
2. `result_year_of_birth`
3. `result_analysis_code`
4. `result_analysis_number`

### 4.5 Appointment domain

1. `appointment_action`
2. `patient_name`
3. `appointment_windows`
4. `appointment_branch_options`

#### `appointment_windows`

Это нормализованный список доступных окон записи, который FT получает из schedule payload и удерживает в active flow.

#### `appointment_branch_options`

Это нормализованный список филиалов, подходящих для записи или связанного адресного уточнения.

## 5. Flow / Dialog State Layer

Это отдельный runtime-layer активного сценария. Он не должен смешиваться с `missing_slots` и `entities`.

Базовые поля:

1. `flow_active`
2. `flow_kind`
3. `flow_stage`
4. `flow_interruptible`
5. `flow_resume_question`
6. `expected_slots`
7. `flow_non_answer_count`
8. `flow_non_answer_kind`

### 5.1 Зачем нужен flow descriptor

Flow descriptor нужен, чтобы FT:

- одинаково обрабатывал active flows без списка `is_active_*` функций;
- умел deterministic различать `same_flow_continuation`, `slot_correction`, `topic_switch_candidate`;
- централизованно применял `interrupt`, `topic switch`, `no preference` и repeated non-answer policy.

### 5.2 Что важно

- owner этих полей находится в runtime/state слое FT;
- любой новый многоходовый flow обязан заполнять этот descriptor;
- `handoff` и `hard reset` должны очищать его полностью вместе с остальным session-state.

## 6. Source / Task Flags

### 6.1 `source_mode`

Возможные значения:

1. `clinic_api`
2. `local_indexes`
3. `web`
4. `self_knowledge`
5. `mixed`

### 6.2 `task_mode`

Возможные значения:

1. `lookup`
2. `clarify`
3. `explain`
4. `recommend`
5. `compare`
6. `analyze`

## 7. Adapter / Backend Mapping Layer

### 7.1 Общий принцип

FT работает с user-facing именами.  
Адаптер переводит их в backend-facing и legacy-имена.

### 7.2 Mapping для результатов анализов

FT user-facing entities:

1. `result_surname`
2. `result_year_of_birth`
3. `result_analysis_code`
4. `result_analysis_number`

Adapter-facing translation:

```json
{
  "surname": "<result_surname>",
  "year": "<result_year_of_birth>",
  "filial": "<result_analysis_code>",
  "number": "<result_analysis_number>"
}
```

### 7.3 Mapping для локации

FT user-facing slot:

- `branch_or_city`

Нормализованные FT entities:

- `branch_name`
- `city`

Adapter behavior:

1. если пользователь указал филиал Самары, переводим в `branch_name`;
2. если пользователь указал `Самара`, переводим в `city=Самара`;
3. если пользователь указал другой город, не пытаемся слепо звать doctor scheduling API.

### 7.4 Mapping для appointment-domain

FT user-facing entities:

1. `appointment_action`
2. `doctor_name`
3. `specialty`
4. `branch_name`
5. `city`
6. `date`
7. `time`
8. `patient_name`

Adapter behavior:

1. appointment-flow не создаёт отдельный booking backend tool;
2. FT использует существующие tools:
   - `doctors_schedule_week`
   - `doctors_info`
   - `address_info`
3. adapter нормализует:
   - `appointment_windows`
   - `appointment_branch_options`
4. terminal completion записи сейчас завершается `handoff`, а не прямым booking API.

## 8. Что НЕ должно попадать в Missing Slots

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

## 9. Открытые вопросы

1. Стоит ли для doctor disambiguation использовать отдельную сущность `doctor_surname`, а не только `surname`?  
Решение: да.
2. Следует ли backend `test_result_status` со временем переименовать legacy-пару `filial/number` в человеко-понятные `analysis_code/analysis_number`?  
Решение: да, но только после отдельной миграции adapter/backend contract.
3. Нужен ли один boolean-флаг для городовых ограничений?  
Решение: нет. Нужна capability matrix по доменам и инструментам.

## 10. Mapping examples

### 10.1 Результаты анализов

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

### 10.2 Смена темы с врача на специальность

История:

1. `Покажи расписание Дразнина`
2. `А выведи всех урологов`

Корректное поведение FT:

- не тащить старый `doctor_name` в новый specialty-query;
- сбросить doctor-specific clarify-state;
- вызвать `doctors_info`, а не `doctors_schedule_week`.

### 10.3 Appointment flow

Пользователь пишет:

`Хочу записаться к Трубину на 16 апреля, 09:00`

FT normalized entities:

```json
{
  "appointment_action": "book",
  "doctor_name": "Трубин Алексей Юрьевич",
  "date": "2026-04-16",
  "time": "09:00"
}
```

FT runtime state:

```json
{
  "flow_active": true,
  "flow_kind": "appointment",
  "flow_stage": "collecting",
  "flow_interruptible": true,
  "flow_resume_question": "Сообщите, пожалуйста, ваше ФИО для записи.",
  "expected_slots": ["patient_name"]
}
```

Дальше FT:

- удерживает `appointment_windows` и `appointment_branch_options`;
- дособирает `patient_name`;
- завершает сценарий через terminal `handoff`.
