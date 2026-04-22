# Pipeline обработки диалогов и запросов Free Talk

Статус: draft v0.2  
Дата: 2026-04-17

## 1. Назначение

Этот документ описывает сквозной pipeline FT в обе стороны:

1. `user -> FT -> adapter -> backend`
2. `backend -> adapter -> FT -> user`

Главная цель:

- явно разделить, что должно делаться детерминированным кодом;
- а что допустимо делегировать LLM.

Связанные документы:

- [free_talk_data_registry.md](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/docs/free_talk_data_registry.md)
- [free_talk_capability_matrix.md](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/docs/free_talk_capability_matrix.md)
- [free_talk_translation_rules.md](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/docs/free_talk_translation_rules.md)
- [free_talk_runtime_architecture.md](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/docs/free_talk_runtime_architecture.md)
- [free_talk_refactor_plan.md](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/docs/free_talk_refactor_plan.md)
- [free_talk_feature_integration_playbook.md](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/docs/free_talk_feature_integration_playbook.md)

## 2. Принцип разделения ответственности

### 2.1 Что должно быть в коде

В коде должны жить:

1. канон слотов и entities;
2. typed signal parsing;
3. flow descriptor и active-flow semantics;
4. state transitions;
5. topic shift detection;
6. stale-state reset;
7. mixed utterance resolve;
8. source/task gating;
9. adapter translation;
10. capability checks;
11. city support policy;
12. derived flags вроде `availability_scope`;
13. interpretation of backend fallback / unsupported / no-data.

### 2.2 Что можно делегировать LLM

LLM должна решать:

1. intent routing при неоднозначной формулировке;
2. wording уточняющего вопроса;
3. synthesis после получения payload;
4. recommendation / compare / analyze поверх уже собранных данных;
5. general knowledge ответы, если выбран соответствующий `source_mode`.

### 2.3 Что нельзя перекладывать на LLM

LLM не должна сама:

1. придумывать backend alias-и;
2. генерировать произвольные `missing_slots`;
3. определять city support policy;
4. тащить/сбрасывать stale state по своей интуиции;
5. решать, как `result_analysis_code` переводится в legacy `filial`.

## 3. Pipeline: user -> FT -> adapter -> backend

## 3.1 Stage A: Session load

Источник:

- Redis state
- Redis memory
- tail истории
- summary

Исполняется кодом.

Результат:

- `dialog_state`
- `session_entity_memory`
- `context`

## 3.2 Stage B: Pre-routing guards

Исполняется кодом.

Задачи:

1. context guard / rollover guard;
2. global interrupt / flow-stop / hard-reset guard;
3. flow-local deterministic handling для active flow:
   - `uncertainty`
   - `no_preference`
   - partial tuples
   - mixed utterances `flow_part -> switch_part`
4. appointment precheck и другие domain-specific active-flow hooks;
5. явные about-agent запросы;
6. topic shift detection;
7. stale-state reset, если новая реплика меняет домен;
8. deterministic hints:
   - doctor follow-up
   - specialty list query
   - city unsupported signals
   - short contextual follow-ups

Базовые runtime-owners этого этапа:

1. `signal_parsers.py` — typed parser layer;
2. `interrupt_policy.py` — global `interrupt/reset/topic_switch`;
3. `flow_local_policy.py` — active-flow deterministic behavior;
4. `appointment_policy.py` — domain-specific appointment precheck;
5. `LLM-arbiter` — только ambiguity fallback поверх deterministic interrupt/topic-switch слоя.

На этом этапе FT различает три разных reset-механизма:

1. `interrupt_current_flow`
   - очищает только текущий flow-state и flow-scoped memory;
   - история и `session_id` сохраняются;
2. `hard_reset_session`
   - очищает всю FT-session;
   - выдаёт новый `next_session_id`;
3. `handoff_terminal_reset`
   - также очищает всю FT-session;
   - но запускается не по пользовательской stop-команде, а по terminal `handoff`.

Отдельно:

1. если детерминированный слой обнаруживает `topic_switch_candidate`, FT поднимает `topic_switch_confirm`;
2. для нового вопроса сохраняется `pending_user_message`;
3. для mixed `clarify/confirmation` допускается и `continue_message`, чтобы при отклонении topic switch вернуться не к старому вопросу буквально, а к уже подтверждённой или скорректированной части той же реплики.

## 3.3 Stage C: Router decision

Исполняется LLM + кодом.

LLM:

- intent
- entities
- clarify question draft
- tool plan draft

Код:

1. валидирует JSON;
2. canonicalizes `missing_slots`;
3. canonicalizes entities;
4. merge-ит с активным state только по разрешенным правилам;
5. отбрасывает неканонические/legacy slot names.

Результат:

- `DialogAct`

## 3.4 Stage D: Entity grounding

Исполняется кодом.

Задачи:

1. `match_catalog_doctor`
2. `match_catalog_service`
3. enrichment из session memory
4. extraction of contextual filters
5. derived flags:
   - `availability_scope`

LLM здесь не нужна.

## 3.5 Stage E: Clarification gate

Исполняется кодом.

Если данных недостаточно:

1. сохраняем новый `DialogState`;
2. увеличиваем clarify count;
3. возвращаем уточнение.

Если flow остаётся активным, в `DialogState` дополнительно поддерживаются:

1. `flow_active`
2. `flow_kind`
3. `flow_stage`
4. `flow_interruptible`
5. `flow_resume_question`
6. `expected_slots`
7. `flow_non_answer_count`
8. `flow_non_answer_kind`

LLM может участвовать только в формулировке вопроса, но не в самом решении, что данных недостаточно.

## 3.6 Stage F: Adapter translation

Исполняется кодом.

Задачи:

1. перевести FT entities в adapter payload;
2. удалить user-facing slot names;
3. перевести human-facing поля в legacy aliases;
4. вычислить domain-specific payload;
5. проставить derived flags.

Примеры:

- `result_analysis_code -> filial`
- `result_analysis_number -> number`
- `branch_name -> branch` для `address_info`
- `doctor_surname -> doctor_last_name`

## 3.7 Stage G: Backend tool call

Исполняется кодом.

Задачи:

1. вызвать нужный tool;
2. интерпретировать `found / error / fallback / unsupported city`;
3. собрать trace.

## 4. Pipeline: backend -> adapter -> FT -> user

## 4.1 Stage H: Backend payload normalization

Исполняется кодом.

Задачи:

1. привести payload к FT-friendly shape;
2. извлечь `entities_used`;
3. обновить session memory;
4. выделить fallback signals:
   - `city_not_supported`
   - `no matches`
   - `missing result fields`
   - `source unavailable`

## 4.2 Stage I: Post-tool verification

Исполняется кодом + LLM.

Код:

1. базовая эвристическая проверка;
2. извлечение missing fields из payload;
3. решение `clarify / direct / not_found` по правилам.

LLM:

1. может уточнить wording final answer;
2. может помочь решить, хватает ли ответа для natural finalization;
3. не должна менять backend facts.

## 4.3 Stage J: Final rendering

Исполняется LLM + кодом.

Код:

1. source tagging;
2. selection of renderer by tool/domain;
3. no-hallucination guardrails;
4. annotation of `clinic_data / general_knowledge / web_search`.

LLM:

1. финальная естественная формулировка;
2. synthesis поверх нескольких payload fragments;
3. recommendation/analyze if это разрешено `task_mode`.

## 4.4 Stage K: Handoff finalization

Исполняется кодом.

Если текущий ответ помечен как `handoff`, FT обязан считать это terminal state.

Допустимые источники handoff-сигнала:

1. `AgentReply.handoff=True`
2. `tool_payload.handoff_required=True`
3. непустой `tool_payload.handoff_message`

Действия finalizer:

1. выполнить полный `clear_session(session_id)`;
2. не сохранять старую сессию обратно в history/summary/meta;
3. выдать новый `next_session_id`;
4. завершить текущий FT-run без reuse старого состояния.

Важно:

- handoff finalization не должен жить как частная логика отдельной фичи;
- это общий runtime-path для всех handoff-сценариев FT.

## 5. Domain-specific notes

## 5.1 Doctor availability

`availability_scope=doctor_visit`

Значит:

- ищем, где принимает врач;
- филиальный фильтр связан с расписанием/приёмом;
- `test_name` не должен втягиваться в этот сценарий.

## 5.2 Service availability

`availability_scope=service_delivery`

Значит:

- услуга/процедура может быть доступна не во всех офисах;
- доступность зависит от врача, оборудования, care-setting и service flags.

## 5.3 Lab availability

`availability_scope=lab_collection`

Значит:

- анализ может сдаваться шире, чем выполняется процедура;
- один и тот же `branch_or_city` в FT-контексте приводит к другому backend смыслу, чем для doctor/service.

## 6. Mini examples

## 6.1 Doctor -> specialty topic shift

История:

1. `Покажи расписание Дразнина`
2. `А выведи всех урологов`

Код должен:

1. распознать topic shift;
2. сбросить stale doctor state;
3. заполнить `specialty=уролог`;
4. выбрать `doctors_info`;
5. не продолжать doctor-schedule clarification loop.

## 6.2 Test result request

Пользователь:

`Иванов, 1989, Бг, 1234`

Код должен:

1. разобрать user-facing FT entities;
2. перевести их в legacy adapter payload;
3. вызвать backend;
4. если backend вернул missing/fallback, интерпретировать это кодом;
5. только затем дать LLM сформулировать финальный текст.

## 6.3 Recommendation question

Пользователь:

`К какому врачу лучше обратиться с натоптышем на пятке?`

Код должен:

1. выбрать правильный `source_mode/task_mode`;
2. собрать clinic-data candidates;
3. при необходимости добавить local indexes / self knowledge;
4. передать LLM уже собранные факты для recommendation synthesis.

## 7. Что надо реализовать дальше

1. Adapter module с явными domain translation rules.
2. Topic shift / stale-state policy как отдельный deterministic block.
3. Derived flags layer:
   - `availability_scope`
4. Unified debug trace:
   - router output
   - canonicalized state
   - adapter payload
   - backend payload
   - post-tool policy
5. После этого уже переписывать FT orchestration под новый канон документов.
