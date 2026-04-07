# Messengers Router: Architecture Status (v4)

Обновлено: **2026-04-07**

Документ отражает текущее состояние `messengers_router` после серии фиксов по flow-контролю, hard reset и очистке stale-контекста.

## 1) Подтверждено на текущей ревизии

1. Архитектурный guardrail:
   - Команда: `python3 messengers_router/scripts/check_architecture_imports.py`
   - Результат: `ARCHITECTURE CHECK PASSED`
   - Метрики: **30 модулей**, **95 внутренних ребер импортов**.
2. Локальные таргетные тесты по flow-контролю:
   - Команда:
     `./.agent_venv/bin/python -m pytest -q tests/test_router_flow_override.py -k "soft_pause_triggers_cancel_confirm or topic_switch_requests_confirmation or topic_switch_confirm_yes_clears_appointment_flow or confirm_pending_yes_handoff_resets_state"`
   - Результат: **9 passed**.
3. Stage3-ожидания приведены к текущему APPOINTMENT-flow:
   - Файл: `messengers_router/scripts/eval_stage3_appointment_flow.py`
   - Обновлены шаги в `APPT_HOLTER_RESCHEDULE`.

## 2) Что сделано по контролю состояния (state) сейчас

### 2.1 Hard reset / stop-фразы в APPOINTMENT

Расширен словарь soft-pause/hard-reset в `appointment_flow_guard.py`.

Поддерживаются в активном APPOINTMENT-flow:
- `стоп`, `отмена`, `передумал`, `не то`, `я не это имел в виду`
- `не туда`, `не так`, `остановись`
- `бред`, `ошибка`, `ты несешь бред`

Поведение: бот спрашивает подтверждение отмены текущего процесса (`да/нет`).

### 2.2 Handoff reset

После `handoff=true` выполняется `_reset_state_after_handoff(...)`:
- `memory.clear_pending(state)`
- `state.last_entities.clear()`
- сохраняется только профильный город (`city=Самара`, если был).

Это эквивалентно сбросу transaction+focus контекста с сохранением минимального profile.

### 2.3 Entity conflict cleanup (частично)

Уже есть точечные механизмы auto-drop stale-сущностей:
- при выдаче списка >1 врача без явного выбора — сброс `doctor_name/doctor_id` (в `response_builder.py`);
- при конфликте `service_name` и врача — `service_name` очищается (в `entity_grounder.py` / `router.py`);
- для price-консультаций снижено влияние stale doctor/service контекста (в `services.py`).

## 3) Сверка с согласованным планом (пункты 2.1 пользователя)

| Пункт | Статус | Комментарий |
|---|---|---|
| Разделить state на `profile_scope / focus_scope / transaction_scope` | **Не сделано** | Сейчас используется единый `state.last_entities` + `memory.pending`. |
| На `handoff=true` сбрасывать transaction+focus, оставлять profile | **Сделано (фактически)** | Реализовано через `_reset_state_after_handoff`; profile сейчас ограничен `city`. |
| Следующий ход после handoff начинать “с чистого листа” | **Сделано** | Подтверждается существующей reset-логикой. |
| Для активных транзакций добавить обязательный confirm при смене интента | **Частично** | Полноценно в APPOINTMENT; для TEST_RESULT / DOC_REQUEST требуется доработка. |
| Для `новый вопрос/другая тема/стоп` делать switch-with-confirm (`да/нет`) | **Частично** | Реализовано в APPOINTMENT precheck; не унифицировано для всех транзакций. |
| Auto-drop stale entities при конфликте | **Частично** | Есть набор точечных правил, но нет единого conflict-resolver слоя. |
| Обновить stage3/critical под строгую логику | **В работе** | Stage3 обновлен; critical расширен новыми hard-reset кейсами, нужен серверный прогон после деплоя. |

## 4) Реализационный план (следующий крупный шаг)

### Phase A: Scoped state (без ломки API)

1. Ввести внутренние helper-функции доступа к scope в `router`-слое:
   - `profile_scope`: `city`, `accepts_children`, `child_age`.
   - `focus_scope`: `specialty`, `service_name`, `doctor_name/doctor_id`, `branch`.
   - `transaction_scope`: `appointment_*`, pending confirm flags, `patient_name`, `date/time`, retry counters.
2. Реализовать это как адаптер поверх `last_entities` (без изменения внешнего контракта).

### Phase B: Unified flow transition policy

3. Вынести единое правило `switch_with_confirm` для всех транзакций:
   - APPOINTMENT, TEST_RESULT, DOC_REQUEST.
4. Для `yes` в switch-confirm:
   - полный reset `transaction_scope + focus_scope`,
   - сохранение `profile_scope`.
5. Для `no`:
   - возврат в текущую транзакцию с re-ask по отсутствующим слотам.

### Phase C: Central conflict resolver

6. Ввести единый post-merge резолвер конфликтов сущностей:
   - новый specialist/service -> drop `doctor_*`;
   - новый doctor -> drop конфликтный `service_name`;
   - `PRICE` по специальности -> игнор stale `doctor_name`, если в вопросе врач не указан.
7. Заменить разбросанные ad-hoc очистки на один реестр правил + debug flags.

### Phase D: Eval contract

8. Зафиксировать новые сценарии в eval:
   - Stage3: hard reset в активном APPOINTMENT-flow.
   - Critical: multi-turn reset и switch-with-confirm для конфликтующих тем.
9. Ввести отдельный отчёт по transition-метрикам:
   - `% topic_switch_with_confirm`,
   - `% stale_entity_dropped`,
   - `% unexpected_flow_carryover`.

## 5) Где сбрасывать флоу, а где нет (актуальная policy-цель)

### Сбрасывать

- после передачи оператору (`handoff=true`);
- при подтвержденной смене темы;
- при конфликте ядра темы (другой специалист/врач/тип задачи);
- при выходе из активной транзакции в другой интент.

### Не сбрасывать

- короткие слот-ответы внутри текущей транзакции (`Трубин`, `завтра 11:00`, `да/нет`);
- follow-up того же интента (`а у других урологов?`, `а в другом филиале?`) с точечной очисткой конфликтных сущностей.

## 6) Изменения в этой ревизии

1. `appointment_flow_guard.py`
   - расширен словарь soft pause / hard reset (добавлены `не туда`, `не так`, `остановись`, `бред`, `ошибка`).
2. `tests/test_router_flow_override.py`
   - добавлены параметризованные тесты на новые hard-reset фразы.
3. `scripts/eval_stage3_appointment_flow.py`
   - обновлены ожидания `APPT_HOLTER_RESCHEDULE`,
   - добавлены hard-reset flow-кейсы.
4. `eval_suite/critical_cases.jsonl`
   - добавлены multi-turn critical-кейсы на hard reset в активной записи.
5. `scripts/check_architecture_imports.py`
   - синхронизирован layer-map с новым модулем `llm_doesnt_work_fallback`.

## 7) Обязательные проверки перед merge/deploy

1. `./.agent_venv/bin/python -m pytest -q`
2. `python3 messengers_router/scripts/check_architecture_imports.py`
3. `python3 messengers_router/scripts/check_eval_coverage.py`
4. `bash messengers_router/eval_suite/run_remote_eval.sh --url <server>/api/messenger-generate-once`
