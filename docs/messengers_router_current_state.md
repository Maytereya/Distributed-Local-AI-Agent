# `messengers_router` — текущее состояние и дельта с 20 апреля 2026

**Для:** Владимир (Free Talk / LLM-first преемник).
**Срез:** 2026-06-02 · ветка `release` · модуль `messengers_router/` (~27k строк, ~60 файлов).

Ты знаешь, как MR работал **на 20 апреля** (по `messengers_router_refactor_plan.md`, Stages 0–22). С тех пор было ~75 коммитов. **Главное в этом доке — §1: что изменилось с 20 апреля.** §2–4 — сжатый текущий контракт/карта для справки (baseline ты в основном знаешь). Дальше сам решишь, что переносить в FT.

> Что НЕ изменилось с 20 апреля (твой baseline остаётся верен): 6-стадийный оркестратор `run_pipeline` (появился ~15.04, с тех пор — единственный путь роутинга; `route_patient_message` жив только для тестов), модуль `russian_nlu` (нормализация + `ENTITY_WHITELIST`), `entity_grounder`, `topic_registry`, `appointment_flow_guard`, дефолтный NLU-движок `legacy_v2` с LLM-first `_merge`. Контракт сущностей/инструментов стабилен с апреля.

---

## 1. Что изменилось с 20 апреля (главное)

Каждый пункт: суть + ключевой коммит (можно `git show <hash>`). Сквозная тема периода — **«честность и надёжность вместо блефа» + анти-зацикливание**.

### 1.1 Структура кода — большой рефакторинг ЗАВЕРШЁН
- Монолит `services.py` (~7.3k стр.) **распилен** в пакет `services/` (Stage 20–22, 20–21.04): `_common`, `_regions`, `_prepare`, `_addresses_helpers`, `_doctors_helpers`, `_prices_helpers` (хелперы) + домен-модули `prices`, `doctors`, `prepare`, `lab_tests`, `addresses`, `main_index`, `news` + `core` (фасад `Services`, бывший `services_legacy.py`). Shim сжат, домен-модули импортируют напрямую (`1f09e77`, `3ea86de`, `307dd1e`, `18bd02b`, …).
  > `refactor_plan.md` описывает этот распил как «ещё не сделано» — **уже сделано**. Самый свежий снимок модулей — §3 ниже.

### 1.2 Честность вместо блефа (доминирующая тема)
- **Несамарский врач** → честный отказ «запись через бота только по Самаре + оператор» вместо слепого списка из 5 филиалов (сигнал `doctor_lookup="unresolved"`) — `009fe7f`.
- **`BUG-2026-06-01-01` (A′/B′, новейшее, в `release`):** ФИО, ошибочно классифицированное LLM как `service_name`, давало «по адресам: [5 филиалов]» для несуществующего врача. Корень: грундер дропает невалидный `service_name` из `decision.entities`, но **quick-fill заново тащит его из сырого текста в `state.last_entities` мимо грундера** (планер читает `state.last_entities`). Починено: `f12096d` (A′-1 — `_suppress_grounder_rejected_slots`), `b5795f2` (A′-2 — гвард «нет цели» читает грундированный decision, не stale specialty), `03ecb0b` (B′ — honest-refuse). Детали и инвариант — `messengers_router_bug_log.md`.
- **Пустое расписание** в APPOINTMENT-превью → честный «свободных слотов нет → оператор» (хелпер `_no_free_slots_operator_offer`), раньше падало в «расписание не найдено» — `b54853a`.
- **Сбой CRM vs «врача нет»:** `_fetch_schedule_source` различает `ScheduleSourceUnavailable` (→ честный handoff / stale-данные) от пустого расписания (→ пусто без handoff); молчаливые сбои fetch теперь логируются WARNING — `39fb9cc`.
- **Stale lab-услуга** не утекает в карточку записи; guard на LLM-выдуманную услугу — `b57ba15`, `86b808e`.
- **Prompt-честность:** запрещено выдумывать цены/сроки для multi-item lab-запросов (`d448ac9`); не предлагать «записать» на лабораторные анализы (`3f9a5d1`).

### 1.3 Анти-зацикливание / эскалация на оператора
- Повтор одинакового ответа N раз → интерактивный **оффер оператора (O4)** — `ebaf794` + `ef354c5`.
- Внутри активного APPOINTMENT-флоу repeat-guard **пропускается** (там повтор уместен) — `acc8950`.
- Расширено распознавание явного запроса человека: «не бот» / «человек» / «оператор» — `a8b47bc`.
- После нерабочих часов — приписка о часах работы оператора в handoff — `fd03983`.

### 1.4 NLU
- **Rescue rule-PRICE:** явный ценовой вопрос с `rule.label=PRICE`, который LLM-merge уронил в `OTHER`, спасается обратно в `PRICE` (узко — строго из `OTHER`, не перехватывает уверенные APPOINTMENT/ADDRESS) — `1f0acd3`.
- Убран stale `@lru_cache` на `_extract_doctor_name` — кэш «врач не найден» больше не залипает после обновления индекса врачей — `1c609c7`.

### 1.5 Цены
- Care-setting адрес (на дому / в клинике) резолвится динамически из `doctor_prices` / через `priceUnit` override-карту — `aa771cc`, `b2f94f1`, `3838402`.
- Матчинг услуги: token-prefix вместо сырой подстроки, отсев `0₽`-строк врачей, падежные формы «стоимость», guard против медицинских кросс-хитов корня — `86b808e`, `d4efa73`.
- Сужены over-broad regex модификаторов (`ген`/`экспресс`/`дет`), скоринг по очищенным токенам (короткие аббревиатуры выживают стоп-слова) — `e832707`, `bf6f69b`.
- Приписка «Цены актуальны для г. Самара» к ответам с ценами — `4733391`.
- Compound-запрос: явный primary над вторичным lab-alias — `7bea980`. «<lab-abbrev> + срочно/cito» → PRICE — `114a1a6`.

### 1.6 Лаборатория / PREPARE
- Compound «ОАК, Ферритин, Витамин Д, …» резолвится **по каждому пункту** из каталога — `703e3e8`.
- «во сколько/когда сдать кровь» → памятка о заборе крови + адреса филиалов с часами; planner ставит `test_prepare` для запросов про тайминг визита — `0079851`, `462416d`, `e4a1a94`.
- «антитела на корь» → «Вирус кори Ig M/Ig G» (`4e14715`); сброс конфликтующего устаревшего анализа при разном биоматериале (`5d82d3e`); уточнение generic «правила подготовки» + clear summary при handoff (`cb9a280`).
- «Чекап» держится **широким** запросом (это категория/линейка, а не одна услуга) → `test_assist` возвращает всё семейство — `fca8cef`.

### 1.7 Расписание
- Ежедневный refresh кэша врачей + восстановлена ветка «no free slots» — `e6d8402`.
- **Демоция bare-specialty** запросов расписания: «расписание уролог» (без фамилии) → `DOCTOR_INFO` (список из JSONL-кэша) вместо загрузки 8 расписаний скопом (раньше 70+ сек / таймаут); fanout распараллелен — `02dd665`.
- Мульти-городские врачи: сохраняем самарское присутствие, узнаём самарские филиалы по region-slug — `90a37b9`, `37c9fd1`.
- Сброс stale `no_free_slots`-reason при матче следующего варианта фамилии (реальное расписание больше не затирается) — `4e3469d`.

### 1.8 Адреса / филиалы
- Восстановлены часы работы («График») из структурированных полей `/regions` — `063c306`.
- Override для фикс-оборудования (флюорография/маммография): корректные адреса + handoff в регистратуру — `3838402`, `e7b9ac9`.
- Multi-word несамарский город («Нижний Новгород», «Самарская область») ловится гео-гейтом «город не поддержан» — `978c57c`.

### 1.9 Специальности
- Канонический `unit→specialty` map, убраны substring-ложноположительные — `082b2b7`.
- Процедуры → специальности для запросов «кто делает X» — `421b3d9`.
- Маппинг unit-имён «Врач-дерматолог» / «Врач-челюстно-лицевой хирург» из live-кэша — `490a8e2`.

### 1.10 Прочее
- «выходные» парсятся как ближайшие сб+вс — `bdaad09`.
- `TEST_RESULT` разрешён для **любого** города (статус результата география не ограничивает — обход самарского гейта) — `277e561`.
- **Чистка мёртвого кода (01.06):** снесён недостижимый сервис `appointment_help` (`9c92658`), мёртвый PREPARE-скоринг (`f0b73e9`), осиротевшие state-сеттеры (`9802bfb`), dead `_static_nonbookable_branches` (`f8a3b76`); починен битый JSON-литерал в critic-prompt (`0675bff`).

### 1.11 Открытый долг / баги (на 02.06)
- **Открытый класс A′** (грундер-дроп воскресает через негрундированные пути записи в `state.last_entities`): починены инстансы service_name/specialty; **отложены** сиблинги — `test_goal` в quick-fill, `_schedule_by_specialty` глотает `ScheduleSourceUnavailable` (CRM-сбой → «не найдено»), LLM-метка safety обходит детерминированный safety-шаблон. Подробности — `messengers_router_checkup_2026-06-01.md` (раздел «Task 1»).
- M1 (`clarify_gate` отдаёт машинные коды) — дремлет под `legacy_v2`, оживёт при `llm_primary`.
- Полный список долга/мёртвого кода — тот же чекап.

---

## 2. Текущий контракт (сжато)

**Пайплайн** (`orchestrator.run_pipeline`): `early_guards` (safety-перехват URGENT/COMPLAINT/MEDICAL_ADVICE по regex, conf=1.0, short-circuit ДО LLM) → `pending_dispatch` (висящие состояния) → `nlu_route` (классификация + сущности) → `doctor_entity_guard` (верификация ФИО + спекулятивный prefetch каталога) → `clarify_gate` → `tool_loop` (`planner` → `executor` → `Evidence`) → `render`.

**NLU `_merge()`** (`nlu_pipeline.py`): для НЕ-safety меток метка LLM побеждает безусловно, правила лишь *донорствуют* сущности; safety-метки ловятся детерминированно. Узкие rescue-правила поверх (см. §1.4). Движок — `MR_NLU_ENGINE` (дефолт `legacy_v2`).

**Сущности** — `ENTITY_WHITELIST` (29 ключей, единый источник правды — `russian_nlu.py`). Всё вне whitelist отбрасывается.

**Инструменты (10)** — эмитит `planner.py`, исполняет `executor.py`: `price_info`, `service_bundle_info`, `doctors_info`, `doctors_schedule_week`, `address_info`, `test_assist`, `test_result_status`, `test_prepare`, `main_index_info`, `news_info`.

**Evidence** — `evidence_keys.py`: payload-ключи (`PRICE`, `DOCTORS_INFO`, `DOCTOR_SCHEDULE`, `TEST_ASSIST`, `PREPARE`, `ADDRESS`, …) + auth-гейт + handoff-гейт. Контракт «executor пишет → response_builder читает».

**Handoff** — единый `HANDOFF_REASON_MATRIX` в `policies.py` (`handoff_message("<reason>")`), без хардкода и без «дыр».

**Состояние записи** — `AppointmentPhase` (`mess_types.py`): `COLLECTING → CONFIRM → …` + reschedule/cancel. Нюанс: фаза `CANCEL_CONFIRM` объявлена, но cancel реально живёт во флаге `appointment_cancel_pending` (фазовая машина расходится с флаговой).

**Два хранилища сущностей** (важно — источник класса A′): `state.dialog.entities` (typed, фактически write-only) и `state.last_entities` (god-object dict, который **читает планер**). Грундер санирует только `decision.entities`.

**Рендер** — детерминированный (`renderer.py` / `response_builder.py`) ИЛИ LLM rich-renderer (`renderer_patient_rich`) + critic-гейт (`self_check.py`).

**Prompt'ы в двух локациях** (правило проекта): `messengers_router/prompts/*` И `app_data/prompts/mr_*` — менять синхронно.

---

## 3. Карта модулей (актуальная на 02.06)

```
messengers_router/
├── endpoint.py             — HTTP-вход, стриминг, DI сервисов, per-session lock
├── orchestrator.py         — 6-stage пайплайн + latency-инструментация
├── router.py               — хелперы стадий, prefetch, сборка ответов (хотспот ~2500 стр.)
├── planner.py              — intent → PlanStep[]
├── executor.py             — выполнение PlanStep → Evidence
├── classifier.py           — интент + сущности (LLM + rule), safety-правила
├── nlu_pipeline.py         — выбор движка, _merge(), rescue-правила
├── russian_nlu.py          — нормализация (normalize_ru), ENTITY_WHITELIST
├── specialty_parser.py     — канонизация специальностей
├── entity_grounder.py      — «приземление»/санитизация сущностей на каталог
├── topic_registry.py       — реестр тем (data/topic_registry.yaml)
├── doctor_name_port.py     — резолв ФИО по локальному индексу врачей
├── policies.py             — HANDOFF_REASON_MATRIX, guardrails, slot-fill, quick-fill (~2200 стр.)
├── flow_policy.py          — active-flow логика, quick_fill_entities_from_text, clear-хелперы
├── appointment_flow_guard.py — FSM записи (confirm/cancel/topic-switch)
├── state_mutations.py      — безопасные мутации SessionState
├── recovery_policy.py      — повторные уточнения / low-confidence
├── llm_mode_policy.py / llm_runtime.py / llm_doesnt_work_fallback.py — режимы LLM + деградация
├── dialog_graph.py         — диагностическая FSM (write-only, в trace)
├── memory.py               — pending/state-хранилище, merge_entities (правила сброса)
├── renderer.py             — рендер ответов (детерм. + LLM rich-renderer) (~980 стр.)
├── response_builder.py     — Evidence → структурированные ответы
├── text_templates.py / service_phrase.py — шаблоны и фразы
├── self_check.py           — critic-гейт (само-проверка LLM-ответа)
├── prompt_contracts.py / prompt_registry.py — контракты и реестр prompt'ов
├── evidence_keys.py / mess_types.py / city.py / runtime_config.py
├── prompts/                — prompt'ы (ДУБЛЬ с app_data/prompts/mr_*)
├── eval_suite/             — эталонные диалоги (remote eval)
└── services/               — доменный слой (бывший монолит, распилен — см. §1.1)
    ├── core.py             — Services dataclass + lifecycle + кеш-хелперы
    ├── _common.py          — базовые утилиты
    ├── prices.py / _prices_helpers.py     — цены, family-mode, модификаторы
    ├── doctors.py / _doctors_helpers.py   — врачи, расписание, специальности
    ├── prepare.py / _prepare.py           — PREPARE: скоринг + relevance-gate + LLM-wrap
    ├── lab_tests.py        — test_assist / test_result_status
    ├── addresses.py / _addresses_helpers.py / _regions.py — адреса, филиалы, часы
    ├── main_index.py       — индекс + налоговая справка
    └── news.py             — новости
```

Бэкенд — один **Nayka site API** (`agent_logic_2/nayka_api/api_nayka.py`), клиника в Самаре. `Services` — фасад с дневным кешем каталогов в `agent_logic_2/nayka_api/apidata/*.jsonl` (датированные файлы). Гео-граница: онлайн-запись только по самарским филиалам; врачи могут вести приём и в другом городе (тогда оставляем самарское присутствие).

---

## 4. Где правда

| Вопрос | Источник |
|--------|----------|
| Как MR работает сейчас | **этот файл** (§2–3) |
| Что изменилось с 20.04 | **этот файл §1** |
| Контракт сущностей | `russian_nlu.ENTITY_WHITELIST` (код — единственная правда) |
| Инструменты | `planner.py` + `executor.py` |
| Handoff-сообщения | `policies.HANDOFF_REASON_MATRIX` |
| История рефакторинга (Stages 0–22) | `messengers_router_refactor_plan.md` (журнал на 20.04, частично устарел) |
| Снимок модулей (апр.) | `CHANGELOG_MAR_APR_2026.md` (22.04) |
| Баги / долг на 01–02.06 | `messengers_router_checkup_2026-06-01.md` |
| Журнал живых багов + инварианты | `messengers_router_bug_log.md` |
| Бэкенд Nayka | `nayka_site_api_live_audit_latest.md` |
