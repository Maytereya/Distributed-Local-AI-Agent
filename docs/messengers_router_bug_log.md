# `messengers_router` — журнал багов (anti-regression log)

**Зачем:** все присланные «живые» баги фиксируются здесь, чтобы код не регрессировал и баг не приходил повторно. Ключевой принцип — **записываем не только инстанс, но и обобщённый инвариант** (класс бага), и покрываем его **тестом на класс**, а не на конкретный случай.

**Почему не только тесты:** инстанс-тест (на «Турмухамбетова») ловит один случай — другой несамарский врач воспроизведёт баг, а тест пройдёт. Поэтому:
1. **Тест на КЛАСС/инвариант** — параметризованный, или сканирует live-кэш и выбирает отсутствующего, или использует синтетическое значение, гарантированно попадающее в класс.
2. **Локальные тесты держат детерминированную логику; remote eval держит live/LLM-поведение** (то, что локально не воспроизвести). Golden-кейсы в `eval_suite/` добавляются по решению владельца.
3. Запись здесь = «память проекта» по багам (версионируется, видна команде, переживает сессии).

**Формат записи:** симптом (как прислал пользователь) · корень · фикс-коммит · **инвариант** · покрытие (тест/eval).

---

## BUG-2026-06-02-01 — корень A′: грундер-дроп воскресает в `state.last_entities` мимо защиты

- **Симптом:** тот же класс, что BUG-2026-06-01-01 («ложные адреса» на запись к врачу не из самарского каталога), плюс родственный вариант: stale `specialty` из прошлой темы → запись «по специальности» вслепую на названного, но не существующего врача.
- **Корень (по live + детерминированному локальному трейсу с call-stack, НЕ угадан):** двухслойный обход грундера.
  - **A′-1 (точка утечки):** грундер [`ground_decision_entities`](../messengers_router/entity_grounder.py) корректно дропает `service_name` из `decision.entities` (флаг `entity_dropped_unverified_service_name`); грундированный decision мёржится в `state.last_entities` чистым ([router.py:2088](../messengers_router/router.py)). **Но сразу следом quick-fill** ([router.py:2106-2145](../messengers_router/router.py)) вызывает `_fill_appointment_entities` ([policies.py:1470-1473](../messengers_router/policies.py)) → `extract_service_phrase(user_text)` заново тащит ФИО из СЫРОГО текста и мёржит в `state.last_entities` **в обход грундера** (`_sanitize_doctor_in_entities` чинит только `doctor_name`). Планер читает `state.last_entities` → `service_name=ФИО`. (Предыдущая гипотеза «утечка через orchestrator:393 state-merge» — НЕВЕРНА: orchestrator:393 пишет в `state.dialog.entities`, которое планер не читает.)
  - **A′-2 (обход B′-гварда):** B′-гвард планера проверял `entities.get("specialty")` из **stale** `state.last_entities`, поэтому оставшаяся от прошлой темы специальность молча подменяла дропнутую цель → слепой оффер филиалов.
- **Инвариант (класс):**
  1. **Любой** слот, отвергнутый грундером в этом ходу (`entity_dropped_*`), НЕ имеет права заново попасть в `state.last_entities` через quick-fill/повторное извлечение из `user_text`. Защита, санирующая `decision.entities`, обязана покрывать и негрундированные пути записи в state, которые читает потребитель (планер).
  2. Гварды «нет валидной цели» на свежем ходу проверяют **грундированный `decision.entities` этого хода**, а не stale `state.last_entities` — чтобы сущность из прошлой темы не подменяла цель. В активном flow допустимо читать state (флоу-непрерывность).
  3. Критерий флаг-точечный (ловит любую цель: полное ФИО / 2-словное / фамилия / stale specialty), не инстанс.
- **Фикс-коммиты:** `f12096d` (A′-1 — `_suppress_grounder_rejected_slots` на 3 quick-fill merge-сайтах router), `b5795f2` (A′-2 — гвард планера вынесен ПЕРЕД missing-slots, читает грундированный decision, старый B′-гвард удалён).
- **Покрытие:** класс-инвариант-тесты в `tests/test_router_flow_override.py`: `test_quickfill_does_not_resurrect_grounder_rejected_service_name` (параметризован по drop-флагам и значениям), `test_quickfill_reextracts_fio_as_service_then_suppressed`, `test_appointment_dropped_target_with_stale_specialty_still_refuses` (параметризован по специальностям), `test_appointment_dropped_target_clean_state_refuses_not_clarifies`, `test_appointment_active_flow_specialty_survives_dropped_service` (анти-over-refuse). Локально: 848 passed, 1 xfailed, ruff clean. e2e-трейс: `service_name` больше не оседает в `last_entities`. **Live «после» — curl после деплоя** (рецепт в BUG-2026-06-01-01).
- **Отложенные сиблинги класса (записаны в чекап, не фиксились):** A′-3 (`test_goal` = сырой `user_text[:200]` в quick-fill), A′-disp (`appointment_service_display` показывает stale lab при отсутствии врача), B-1 (`_schedule_by_specialty` глотает `ScheduleSourceUnavailable`), B-2 (LLM-метка safety обходит детерминированный safety-шаблон).

---

## BUG-2026-06-01-01 — «ложные адреса» для врача не из самарского справочника

- **Симптом (live):** «Записаться к ТУРМУХАМБЕТОВА БАЛСЛУ ТУРМУРАДОВНА» → бот выдал «Есть возможность записи на Турмухамбетова… в городе Самара по адресам: [5 филиалов]. Какой филиал вам удобен? Или написать список врачей по этой услуге?» — врача нет в самарском онлайн-каталоге (`doctors_*.jsonl`: 0 совпадений).
- **Корень (по live-debug, не угадан):** двухслойно. (1) LLM классифицировал ФИО как `service_name` (не `doctor_name`). (2) Грундер дропнул его (`entity_dropped_unverified_service_name`) из `decision.entities` — но дропнутое значение **просочилось в план**: `nlu_route` (orchestrator:393) мёржит СЫРОЙ `decision.entities` (с `service_name=ФИО`) в state ДО грундинга; грундинг (router.py:2045-2057) дропает из `decision.entities`, но **не из state**; `build_plan` мёржит state → `plan.steps[0]`=`address_info` с `service_name=ФИО` → branch-selection → 5 филиалов. Т.е. дроп невалидной «услуги» не доходит до плана/исполнения.
- **Live-repro:** `curl -s -X POST http://172.16.0.16/api/messenger-generate-once -H 'Content-Type: application/json' -d '{"session_id":"dbg","text":"Записаться к <ФИО-не-из-Самары>","debug":true}'` → в `state_update.debug.decision.flags` есть `entity_dropped_unverified_service_name`, а `plan.steps[0].input.entities.service_name` = ФИО.
- **Инвариант (класс):** в APPOINTMENT-флоу, когда цель «дропнута как неверифицированная» (`entity_dropped_unverified_service_name`) и нет валидного `doctor_name`/`specialty`/реальной услуги — бот **никогда** не предлагает самарские филиалы; уточняет врача/услугу или честно отказывает (`_doctor_not_bookable_via_bot_offer`). Критерий точный (грундер уже отличает реальную услугу от мусора — реальные услуги НЕ дропаются), ловит и 2-словные имена. Критерий «врача нет в кэше = не самарский» сам по себе недостаточен: реальные услуги тоже не в doctors-кэше (`extract_doctor_name_candidate` ложно матчит УЗДГ/ОАК/УЗИ).
- **Фикс-коммит:** `03ecb0b` (B′ — planner honor-ит флаг дропа: `entity_dropped_unverified_service_name` + нет doctor/specialty → не идём в `address_info`, метим ход → `build_appointment_step_response` отдаёт existing honest-refuse). **A′ закрыт (Task 1, 2026-06-02):** корень оказался НЕ в state-merge `decision.entities` (как тут предполагалось), а в re-inject через quick-fill + обходе гварда stale-специальностью — см. **BUG-2026-06-02-01** (`f12096d` + `b5795f2`). Описание корня выше (про orchestrator:393) — историческая гипотеза, опровергнута трейсом.
- **Покрытие:** класс-инвариант-тесты (`test_appointment_dropped_unverified_target_does_not_offer_branches` — флаг-триггер, параметризован по ФИО/2-словным/фамилии; `test_appointment_unbookable_target_marker_yields_honest_refusal`). Локально: 843 passed, ruff clean. Live «до» подтверждён (флаг выдаётся, баг есть). **Live «после» — твой curl после деплоя** (рецепт выше).

---

> Шаблон новой записи копировать сверху. ID: `BUG-<дата>-NN`.
