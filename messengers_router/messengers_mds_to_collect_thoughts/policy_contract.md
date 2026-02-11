# Policy Contract для messenger router

## 1. Scope и источники

Этот контракт собран по текущим артефактам проекта:

- `/Users/maxten/Dev/Distributed-Local-AI-Agent2/messengers_router/messengers_mds_to_collect_thoughts/router_instruction.md`
- `/Users/maxten/Dev/Distributed-Local-AI-Agent2/messengers_router/messengers_mds_to_collect_thoughts/intent_question_flow.md`
- `/Users/maxten/Dev/Distributed-Local-AI-Agent2/messengers_router/messengers_mds_to_collect_thoughts/intents_review.md`
- `/Users/maxten/Dev/Distributed-Local-AI-Agent2/messengers_router/messengers_mds_to_collect_thoughts/messenger_intents_overview.md`
- `/Users/maxten/Dev/Distributed-Local-AI-Agent2/messengers_router/messengers_mds_to_collect_thoughts/messenger_intents_schema.md`
- `/Users/maxten/Dev/Distributed-Local-AI-Agent2/messengers_router/messengers_mds_to_collect_thoughts/messenger_trigger_phrases_raw.md`
- текущая реализация:
  - `/Users/maxten/Dev/Distributed-Local-AI-Agent2/messengers_router/classifier.py`
  - `/Users/maxten/Dev/Distributed-Local-AI-Agent2/messengers_router/router.py`
  - `/Users/maxten/Dev/Distributed-Local-AI-Agent2/messengers_router/services.py`
  - `/Users/maxten/Dev/Distributed-Local-AI-Agent2/messengers_router/memory.py`
  - `/Users/maxten/Dev/Distributed-Local-AI-Agent2/messengers_router/renderer.py`

Опорная статистика из `intent_question_flow.md`: 259 файлов, 252 корректно выделенных диалога.

## 2. Канонический словарь интентов (labels)

### 2.1 Канонические labels в коде

- `URGENT`
- `COMPLAINT`
- `MEDICAL_ADVICE`
- `APPOINTMENT`
- `TEST_ASSIST`
- `TEST_RESULT`
- `DOCTOR_INFO`
- `DOCTOR_SCHEDULE`
- `PRICE`
- `ADDRESS`
- `PREPARE`
- `NEWS`
- `OTHER`

### 2.2 Маппинг аналитических названий на labels роутера

- `LAB_TESTS_INFO` -> `TEST_ASSIST`
- `APPOINTMENT_BOOK` -> `APPOINTMENT`
- `DIAGNOSTICS_INFO` -> `APPOINTMENT` или `PRICE` (в зависимости от запроса)
- `ADDRESS_HOURS` -> `ADDRESS`
- `PRICE_QUERY` -> `PRICE`
- `DOCTOR_INFO` -> `DOCTOR_INFO`
- `RESULTS_DOCS` -> `TEST_RESULT`
- `APPOINTMENT_MOVE_CANCEL` -> `APPOINTMENT` с `appointment_action=reschedule|cancel`
- `DISCOUNT_PROMO` -> `NEWS` или `PRICE` (пока отдельного label нет)
- `INSURANCE_DMS_OMS` -> `OTHER` + handoff/уточнение (до выделения отдельного label)

## 3. Приоритет принятия решения (policy precedence)

Порядок обязателен:

1. Greeting/smalltalk (`OTHER`, без handoff).
2. Hard safety rules:
   - `URGENT`
   - `COMPLAINT`
   - `MEDICAL_ADVICE` (включая интерпретацию анализов)
3. Hard business rules (минимум `APPOINTMENT`-триггеры).
4. LLM classification (JSON schema).
5. Slot-policy и pending-уточнения.
6. План инструментов (`PlanStep`), сбор `Evidence`.
7. Renderer только упаковывает факты из evidence.

Правило: ошибки label не лечатся промптом рендера.

## 4. Slot policy (на основе чатов + текущего роутера)

### 4.1 Обязательные слоты по интентам

- `APPOINTMENT`:
  - `_any_of:doctor_id,doctor_name,specialty,service_name`
  - `_any_of:city,branch_name,branch_id`
  - в активном сценарии записи дополнительно собираем `date/time` и подтверждение.
- `TEST_ASSIST`:
  - `_any_of:city,branch_name,branch_id`
  - `_any_of:test_goal,test_name`
- `TEST_RESULT`:
  - `surname`
  - `year`
  - `filial`
  - `number`
- `DOCTOR_INFO`:
  - `_any_of:specialty,doctor_id,doctor_name`
- `DOCTOR_SCHEDULE`:
  - `_any_of:doctor_id,doctor_name`
- `PRICE`:
  - `_any_of:city,branch_name,branch_id`
  - `service_name`
- `ADDRESS`:
  - `_any_of:city,branch_name,branch_id`
- `PREPARE`:
  - `_any_of:test_name,service_name`

### 4.2 Порядок уточнений (из чатов)

- `PRICE`: чаще сначала город, затем филиал/детали услуги.
- `TEST_ASSIST`: чаще сначала город, затем филиал/дата/цель.
- `APPOINTMENT`: не всегда город первым; часто сначала услуга/врач/возраст, затем филиал, затем дата/время.

### 4.3 Целевой flow для записи (MVP)

1. Определить услугу/врача.
2. Запросить город (если отсутствует).
3. Показать релевантные адреса в городе.
4. Запросить филиал.
5. Сообщить стоимость (если есть), запросить дату/время.
6. Показать summary и запросить подтверждение.
7. После `да` -> `handoff=true` с подготовленным summary для оператора.

## 5. Handoff policy

### 5.1 Немедленный handoff

- `URGENT`
- `COMPLAINT`
- `MEDICAL_ADVICE`

### 5.2 Handoff после частичной автоматизации

- `APPOINTMENT`: после подтверждения заявки пользователем.
- `TEST_RESULT`: при технической ошибке источника данных (`resultForPatient`) или явном service fallback.
- недоступность критичных интеграций (timeout/API error) без рабочего fallback.

### 5.3 Не ставить handoff

- пока бот задает уточняющий вопрос по pending-слотам;
- когда можно продолжить сценарий без риска.

## 6. Service contract (инструменты)

- `APPOINTMENT`:
  - с врачом -> `doctors_schedule_week`
  - без врача -> `address_info` + `price_info`
- `DOCTOR_INFO` -> `doctors_info`
- `DOCTOR_SCHEDULE` -> `doctors_schedule_week`
- `PRICE` -> `price_info`
- `ADDRESS` -> `address_info`
- `TEST_ASSIST` -> `test_assist`
- `PREPARE` -> `test_prepare`
- `TEST_RESULT` -> `test_result_status`, `test_result_pdf`
  - текущий контракт: сначала сбор `surname/year/filial/number`, затем запрос в `resultForPatient`
- `NEWS` -> `news_info`

Правило источников:

1. Для адресов/расписания использовать live API приоритетно.
2. Кэш использовать как fallback.
3. Ошибки внешних систем не должны приводить к 500 в endpoint.

## 7. Observability contract (debug)

В debug-ответе обязательны:

- `decision`:
  - `label`, `confidence`, `flags`, `needs_handoff`, `entities`
- `plan`:
  - `label`, `steps`
- `evidence`:
  - `items` + `debug_trace`
- `pending`
- `history_tail`
- `last_entities`

Это минимальный контракт для диагностики "почему бот ответил так".

## 8. Найденные расхождения (docs vs code)

1. В prompt нужно поддерживать только канонические labels и не добавлять экспериментальные label-алиасы.
2. Policy частично централизован в `policies.py` (детекторы + slot/clarify/handoff), но часть логики еще остается в `router/services`.
3. Нет автоматизированного eval-контура на корпусе чатов (регрессы ловятся вручную).
4. По `TEST_RESULT` интеграция `resultForPatient` уже есть, но нужны стабильные тестовые данные и донастройка сценариев на реальных кейсах.
5. Future-кейсы вынесены отдельно: `future_intents_backlog.md`; до отдельной реализации они маршрутизируются как `OTHER + handoff`.

## 9. Roadmap работ (поэтапно)

### Этап 1. Нормализация контракта классификатора [DONE]

- Привести prompt few-shot к каноническим labels.
- Зафиксировать JSON schema для classifier как единственный контракт.
- Критерий готовности: нет неканонических labels в debug.
- Факт: выполнено, `eval_stage1_cases.py` = `20/20`.

### Этап 2. Централизация policy [DONE]

- Вынести правила handoff/slots/clarify в отдельный модуль policy (или policy map).
- Текущий статус: `REQUIRED_SLOTS`, `missing_slots`, `clarification_question`, `evidence_requires_handoff` перенесены в `policies.py`.
- Текущий статус: добавлены `HANDOFF_REASON_MATRIX` и `appointment_step_policy`, роутер использует их напрямую.
- Текущий статус: APPOINTMENT-тексты и переходы подтверждения (`yes/no/other`) вынесены в policy-функции.
- Текущий статус: вынесены точечные policy-тексты `DOCTOR_SCHEDULE`-уточнения и `decision.needs_handoff` fallback.
- Критерий готовности: все уточнения и handoff определяются таблицами, не "рассыпаны" по коду.
- Факт: выполнено.

### Этап 3. APPOINTMENT flow до стабильного MVP [DONE]

- Дожать сценарий `услуга -> город -> филиал -> дата/время -> подтверждение -> оператор`.
- Проверить фильтрацию адресов по городу и релевантности услуги.
- Текущий статус: добавлен stage-3 regression `scripts/eval_stage3_appointment_flow.py` (multi-turn сценарии).
- Текущий статус: `appointment_addresses` в policy теперь учитывает город (city-aware fallback).
- Критерий готовности: повторяемый сценарий без зацикливаний в simulator.
- Факт: выполнено, `eval_stage3_appointment_flow.py` = `10/10`.

### Этап 4. Надежность services [DONE]

- Добавить единый обработчик timeout/error для внешних источников.
- Добавить явный `handoff_reason` в evidence.
- Критерий готовности: отсутствие 500 при деградации API/Meili.
- Текущий статус: в `services.py` добавлен единый fallback-контракт (`handoff_required`, `handoff_reason`, `handoff_message`) для основных внешних вызовов.
- Текущий статус: в `router.py` добавлен fail-safe для `route_patient_message` (исключения переводятся в контролируемый handoff вместо 500).
- Текущий статус: выбор филиала в APPOINTMENT резолвится по ранее показанным пользователю адресам (без ухода в “чужие” адреса из общего справочника).
- Текущий статус: добавлен pre-deploy smoke `scripts/eval_stage4_reliability.py` (проверка `HTTP 200 + JSON envelope` на наборе сценариев).
- Факт: выполнено.

### Этап 5. Eval на корпусе чатов [IN PROGRESS]

- Собрать golden-набор сообщений (по 20-50 на ключевой интент).
- Метрики:
  - intent accuracy
  - slot fill rate
  - false handoff rate
  - unsafe miss rate (URGENT/COMPLAINT/MEDICAL_ADVICE)
- Критерий готовности: зафиксированные baseline и пороги релиза.
- Текущий статус: добавлен генератор golden-корпуса `scripts/build_stage5_golden_cases.py`.
- Текущий статус: добавлен корпусный eval `scripts/eval_stage5_corpus.py` (intent/handoff/slot/unsafe + gate).
- Текущий статус: добавлен preflight endpoint check в `eval_stage5_corpus.py`, чтобы сразу показывать проблему окружения вместо массовых `transport_or_json_error`.
- Текущий статус: добавлена инструкция запуска `stage5_corpus_eval.md`.
- Текущий статус: зафиксирован baseline в `stage5_baseline.md` и разбор mismatch в `stage5_mismatch_analysis_1770728996.md`.
- Последний run: `1770839513` -> `intent 89.8%`, `handoff 91.8%`, `false_handoff 2.0%`, `slot_fill 29.7%`.
- Оставшийся фокус: `APPOINTMENT` edge-cases, `TEST_RESULT` ожидания в golden и качество slot-fill.

### Этап 6. Промпты и стиль

- После стабилизации policy обновить classifier/renderer prompts под реальные формулировки из чатов.
- Критерий готовности: рендер не выдумывает факты, стиль стабильный, короткий, пациентский.

## 10. Что делаем следующим шагом

Следующий практический шаг: закрыть Этап 5 на рабочем окружении и зафиксировать baseline.

- синхронизировать golden-ожидания для `TEST_RESULT` с текущим контрактом сбора полей;
- поднять slot fill rate для `APPOINTMENT`/`TEST_RESULT`;
- перепрогнать `scripts/eval_stage5_corpus.py` и поднять показатели до release-threshold.

После закрытия Этапа 5: перейти к Этапу 6 (чистка classifier/renderer prompts под реальные формулировки из чатов без изменения policy-contract).
