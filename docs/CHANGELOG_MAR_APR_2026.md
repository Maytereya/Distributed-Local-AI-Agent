# Изменения в чат-боте — март–апрель 2026

**Период:** 3 марта → 22 апреля 2026 (≈7 недель) · **216 коммитов** · `release`-ветка

Короткая версия для коллеги. Полный разбор — в `docs/messengers_router_refactor_plan.md` и истории git.

---

## За что отвечает какой модуль (после рефакторинга)

```
messengers_router/
├── router.py               — точка входа, роутинг по интентам
├── orchestrator.py         — 5-stage pipeline (NLU → guards → execute → render → handoff)
├── classifier.py           — интент + entities (LLM-first, rule-донор)
├── nlu_pipeline.py         — pipeline координация
├── russian_nlu.py          — единая нормализация (normalize_ru, ENTITY_WHITELIST)
├── specialty_parser.py     — канонизация специальностей, PROCEDURE_TO_SPECIALTY
├── topic_registry.py       — реестр тем (443 строки, выделен из монолита)
├── appointment_flow_guard.py — охрана контекста записи
├── state_mutations.py      — безопасные мутации state
├── evidence_keys.py        — константы evidence-ключей
├── policies.py             — handoff-матрица, guardrails
└── services/
    ├── core.py             — Services dataclass + кэш-helpers (было services.py на 7 288 строк → 535)
    ├── _common.py          — базовые утилиты (normalise, runtime_*)
    ├── _prepare.py         — хелперы PREPARE (скоринг, relevance gate)
    ├── _regions.py         — regions/city
    ├── _addresses_helpers.py
    ├── _doctors_helpers.py — специальности, FIO-matching
    ├── _prices_helpers.py  — прайс, family-mode, модификаторы
    ├── prepare.py          — метод test_prepare + prepare flow
    ├── prices.py           — price_info, service_bundle_info
    ├── doctors.py          — doctors_info, doctors_schedule_week, match_catalog_*
    ├── addresses.py        — address_info
    ├── lab_tests.py        — test_assist, test_result_status
    ├── main_index.py       — main_index_info (+ налоговая справка)
    ├── appointments.py     — appointment_help
    └── news.py             — news_info
```

## Что сделано по каждому пункту ТЗ

| # | Пункт ТЗ | Ключевое изменение | Куда смотреть |
|---|---|---|---|
| 1 | Результаты анализов | Выделен домен + синхронизация с сервером №2 | `services/lab_tests.py`, `test_freetalk_labs_*` |
| 2 | Стоимость | Family-mode, OAK canonical, компаундные запросы, все модификаторы (cito/капилляр/повторный/дом/дет) | `services/_prices_helpers.py` (2 266 строк) |
| 3 | Услуга + 4 врача + подготовка | Новый PREPARE с скорингом + LLM-validation в серой зоне + LLM-wrap + serviceInfoAll fallback | `services/_prepare.py`, `services/prepare.py` |
| 4 | Конкретный врач | Устранена потеря ФИО на длинном контексте, doctor_name_port утилита | `messengers_router/doctor_name_port.py` |
| 5 | Врачи специальности | Deep specialty matching (compound + role-link), `_specialty_priority_rank`, лаб-услуги отключены от doctor-link | `services/_doctors_helpers.py` (1 415 строк) |
| 6 | Расписание | Единый `schedule_ttl_cache.py`, stale-fallback, различие «нет врача» vs «нет слотов», параллельный prefetch | `services/core.py::_fetch_schedule_source`, Stage 15 |
| 7 | Запись | Отдельный модуль + `AppointmentPhase` state-машина + reschedule/cancel + topic-switch detection | `appointment_flow_guard.py`, `mess_types.AppointmentPhase` |
| 8 | Справка в налоговую | `doc_request_kind` классификатор + tax direct-link + fallback queries | `services/main_index.py` |

## Архитектура

- **Тесты:** 0 → **653** unit-тестов в 39 файлах (+15 439 строк)
- **Регрессия:** 62 эталонных диалога в `messengers_router/eval_suite/` (critical / extended / server_parity / prepare_wrap)
- **Рефакторинг Stage 20–22:** монолит `services.py` (7 288 строк) распилен на 14 доменных модулей. `core.py` = 535 строк, только `Services` dataclass + lifecycle.
- **CI:** guardrails на циклические импорты и нарушения слоёв.
- **LLM-first NLU:** `_merge()` переписан, правила только донорствуют entities в LLM-ответ.
- **Performance:** параллельный doctor verify + speculative service-catalog prefetch (Stage 15), lru_cache на детерминированные helper'ы (Stage 16), per-stage latency instrumentation (Stage 14).
- **Free Talk AI:** отдельный подпроект с clinical router, tool registry, tool-loop архитектурой, сервером №2 (Llama).

## Ключевые багфиксы (регрессия-защищены)

| Баг | Где кейс |
|---|---|
| «Подготовиться к ФГДС» → искал врача с ФИО «ФГДС» | `CRIT_PREPARE_FGDS_001` |
| Вульвоскопия/любая процедура → «в Самаре доступны филиалы» | `CRIT_PREPARE_VULVOSCOPY_001` |
| Запись к Ким → зацикливание на расписании | `CRIT_APPT_KIM_LOOP_001` |
| Справка для налоговой → handoff к оператору | `CRIT_DOC_TAX_001` |
| Корректный флоу кардиолог → запись → подтверждение | `CRIT_CARDIO_TO_BOOKING_OK_001` |
| Хальметова, ЭКГ, холестерин, ревматолог, шунтирование желудка — закрыты персональными правками |

## Что ещё осталось по плану (не этот цикл)

- **Stage 17b** — speculative parallel rule+LLM NLU с early-cancel (нужно явное ОК заказчика, highest-risk)
- **3 eval-кейса по прайсу** — известны, скипнуты этим циклом
- `router.py` (2 400 строк) — следующая цель рефакторинга, но production-стабилен

## Как запускать локально

```bash
# гейт
venv/bin/ruff check messengers_router/
PYTHONPATH=. venv/bin/python -m pytest tests/ --ignore=tests/eval -q
# ожидается: ruff clean, 647 passed

# remote eval (к живому серверу)
./run_remote_eval.sh --url http://172.16.0.16/api/messenger-generate-once
```
