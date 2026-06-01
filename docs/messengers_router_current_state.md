# `messengers_router` — как сейчас работает бот

**Аудитория:** Владимир (команда Free Talk / LLM-first).
**Дата среза:** 2026-06-01 · ветка `release` · модуль `messengers_router/` (~26.7k строк, ~60 файлов).
**Зачем:** ты делаешь Free Talk (FT) — LLM-first преемник. Этот документ — карта того, как работает текущий **детерминированный** продакшн-бот: его пайплайн, NLU-контракт, набор инструментов, что он умеет и что недавно поменялось. Чтобы FT мог (а) повторить контракт, (б) не наступить на те же грабли.

> **Самый свежий структурный док** до этого — [`messengers_router_refactor_plan.md`](messengers_router_refactor_plan.md) (20.04), но это **журнал рефакторинга** (Stages 0–22), и он уже частично устарел (описывает распил `services_legacy.py` как «ещё не сделано», хотя монолит уже распилен). Чистый снимок модулей — в [`CHANGELOG_MAR_APR_2026.md`](CHANGELOG_MAR_APR_2026.md) (22.04). **Этот файл — текущая правда на 01.06.**
>
> ⚠️ Параллельно лежит [`messengers_router_checkup_2026-06-01.md`](messengers_router_checkup_2026-06-01.md) — фул-чекап на баги/мёртвый код. Прочитай его раздел «HIGH», чтобы не перенести эти баги в FT.

---

## 1. MR vs FT в двух словах

| | `messengers_router` (MR) — этот проект | Free Talk (FT) — твой |
|---|---|---|
| Подход | **Детерминированный роутер**: правила + LLM только на узких участках | **LLM-first**: модель ведёт диалог, код — инструменты |
| Точка решения | жёсткий пайплайн стадий, LLM «донорствует» | LLM + tool-loop |
| Статус | **в проде** | разрабатывается |
| Бэкенд | один и тот же — Nayka site API (клиника, Самара) | тот же |

MR и FT целятся в одни и те же сценарии (цены, запись, врачи, адреса, результаты, подготовка) и один бэкенд. Поэтому **контракт сущностей/инструментов/handoff из MR — это то, что FT должен уметь воспроизвести** (раздел 4).

---

## 2. Высокоуровневая архитектура: 6-стадийный пайплайн

Точка входа — `endpoint.py` (стриминговый HTTP) → `orchestrator.run_pipeline()` ([`orchestrator.py:586`](../messengers_router/orchestrator.py)). Пайплайн (в коде назван «6-stage skeleton»):

```
текст пользователя
      │
 1. early_guards         — safety-перехват (URGENT/COMPLAINT/MEDICAL_ADVICE по детерм. правилам,
      │                     confidence=1.0) + явные команды → short-circuit в render
      │  (short-circuit?) ─────────────► render
 2. pending_dispatch     — обработка «висящих» состояний: подтверждение записи, оффер оператора,
      │                     вторичная очередь интентов, ожидание слота
      │  (short-circuit?) ─────────────► render
 3. nlu_route            — классификация интента + извлечение сущностей (см. §3)
      │
 4. doctor_entity_guard  — верификация ФИО врача по каталогу + спекулятивный prefetch каталога услуг
      │                     (Stage 15: verify ∥ match_catalog_service через asyncio.gather)
      │
 5. clarify_gate         — нужно ли уточнение перед вызовом инструментов
      │
 6. tool_loop            — planner строит PlanStep'ы → executor зовёт сервисы → evidence
      │
   render                — evidence → текст пациенту (детерм. рендер ИЛИ LLM rich-renderer + critic)
      │
   ответ (+ handoff на оператора, если нужно)
```

Ключевые модули пайплайна:
- `router.py` (~2480 стр.) — самый большой; хелперы стадий, prefetch-логика, сборка ответов. **Следующая цель рефакторинга**, но прод-стабилен.
- `orchestrator.py` — сам skeleton + per-stage latency-инструментация (`_timed_stage`).
- `planner.py` — `intent → list[PlanStep(tool=...)]`. Здесь определяется, какой инструмент дёрнуть.
- `executor.py` — выполняет PlanStep'ы, пишет в `Evidence`.
- `response_builder.py` — превращает `Evidence` в структурированные ответы.

> **Историческая заметка для FT:** `route_patient_message` в `router.py` — это **старый** монолитный путь. Прод его больше не зовёт (идёт через `run_pipeline`), он жив только ради тестов. Не бери его за образец — ориентируйся на `run_pipeline` + стадии.

---

## 3. NLU: LLM-first с rule-донорами (это важно для FT)

NLU — гибрид, и это самый поучительный для FT слой.

- **Движок** выбирается `MR_NLU_ENGINE` ([`nlu_pipeline.py:155`](../messengers_router/nlu_pipeline.py)): дефолт в проде — **`legacy_v2`**; альтернатива — `llm_primary` (LLM как первичный классификатор).
- **`_merge()`** ([`nlu_pipeline.py:71`](../messengers_router/nlu_pipeline.py)) — сердце гибрида. Принцип: **для не-safety меток метка LLM побеждает безусловно**, правила лишь *донорствуют* сущности в ответ LLM. Старый путь «высокая уверенность правила → пропустить LLM» был сознательно убран — он давал расхождение «зелёный eval / красный прод» (грубый regex молча перебивал верную LLM-классификацию).
- **Safety-метки — исключение:** `{URGENT, COMPLAINT, MEDICAL_ADVICE}` ловятся детерминированными правилами с `confidence=1.0` и короткозамыкают пайплайн **до** вызова LLM (`early_guards`). FT обязан сохранить этот приоритет безопасности.
- **Узкие rescue-правила** поверх merge — пример из свежих коммитов: явный ценовой вопрос с `rule.label=PRICE`, который LLM уронил в `OTHER`, спасается обратно в `PRICE` (`1f0acd3`). Это паттерн «LLM主, но узкий детерминированный предохранитель на дорогих ошибках».

**Вывод для FT:** даже в LLM-first мире MR держит (1) safety-перехват до LLM, (2) донорство сущностей правилами, (3) точечные rescue на классах ошибок, которые дорого стоят пациенту. Это не «или LLM, или правила», а «LLM + тонкая детерминированная страховка».

---

## 4. Контракт, который FT должен воспроизвести

### Интенты (метки классификатора)
`PRICE · APPOINTMENT · ADDRESS · DOCTOR_INFO · DOCTOR_SCHEDULE · TEST_RESULT · TEST_ASSIST · PREPARE · NEWS · MAIN_INDEX` (+ doc/справки) `· OTHER`
Safety: `URGENT · COMPLAINT · MEDICAL_ADVICE`.

### Сущности — `ENTITY_WHITELIST` (единый источник правды, [`russian_nlu.py:26`](../messengers_router/russian_nlu.py))
29 ключей; всё, что вне whitelist, отбрасывается (`prompt_contracts.ALLOWED_ENTITY_KEYS`):
```
doctor_name, doctor_id, specialty, branch_name, branch_id, city, service_name,
appointment_action, patient_name, test_name, test_goal, surname, year, filial,
number, order_id, result_action, lang, insurance_type, accepts_children,
child_age, date_hint, date_from, date_to, time_from, time_to,
secondary_intents, include_promos, time_flexible
```

### Инструменты (10) — что эмитит `planner.py`, что исполняет `executor.py`
| tool | назначение |
|------|-----------|
| `price_info` | стоимость услуги (+ модификаторы: cito / капиллярный / повторный / на дому / детский) |
| `service_bundle_info` | услуга + врачи + подготовка одним пакетом; compound-запросы |
| `doctors_info` | информация о враче / врачах специальности |
| `doctors_schedule_week` | расписание врача на 2 недели |
| `address_info` | адреса/филиалы + часы работы |
| `test_assist` | подбор анализов (чекап-семейства и т.п.) |
| `test_result_status` | статус готовности результата анализа |
| `test_prepare` | подготовка к анализу/процедуре (скоринг + relevance-gate + LLM-валидация) |
| `main_index_info` | общий индекс + справки (в т.ч. налоговая) |
| `news_info` | новости/акции |

### Evidence keys — контракт «executor пишет → response_builder читает»
Константы в [`evidence_keys.py`](../messengers_router/evidence_keys.py): payload-ключи (`PRICE`, `DOCTORS_INFO`, `DOCTOR_SCHEDULE`, `TEST_ASSIST`, `PREPARE`, `ADDRESS`, …), auth-гейт (`AUTH_REQUIRED/MESSAGE`), handoff-гейт (`HANDOFF_REQUIRED/REASON/MESSAGE`).

### Handoff на оператора
Все сообщения о передаче оператору централизованы через `HANDOFF_REASON_MATRIX` в `policies.py` (`handoff_message("<reason>")`). Если FT передаёт оператору — делай это через единый reason-словарь, а не хардкодом (в MR это закрыто Stage 6, и матрица сейчас без «дыр»).

### Состояние записи (appointment FSM)
`AppointmentPhase` ([`mess_types.py`](../messengers_router/mess_types.py)): `COLLECTING → CONFIRM → …`, плюс reschedule/cancel и detection переключения темы. Управляется через `state_mutations.py` (безопасные сеттеры) и `appointment_flow_guard.py`.
> Нюанс (см. чекап): фаза `CANCEL_CONFIRM` объявлена, но реально cancel живёт во флаге `appointment_cancel_pending`, а не в `dialog.phase`. Если FT делает свою FSM записи — заведи отмену как явное состояние сразу.

---

## 5. Карта модулей (актуальная на 01.06)

```
messengers_router/
├── endpoint.py             — HTTP-вход, стриминг, DI сервисов, per-session lock
├── orchestrator.py         — 6-stage пайплайн + latency-инструментация
├── router.py               — хелперы стадий, prefetch, сборка ответов (хотспот ~2480 стр.)
├── planner.py              — intent → PlanStep[]
├── executor.py             — выполнение PlanStep → Evidence
├── classifier.py           — интент + сущности (LLM + rule), safety-правила
├── nlu_pipeline.py         — выбор движка, _merge(), rescue-правила
├── russian_nlu.py          — единая нормализация (normalize_ru), ENTITY_WHITELIST
├── specialty_parser.py     — канонизация специальностей
├── entity_grounder.py      — «приземление» сущностей на каталог
├── topic_registry.py       — реестр тем (data/topic_registry.yaml)
├── doctor_name_port.py     — резолв ФИО по локальному индексу врачей
├── policies.py             — HANDOFF_REASON_MATRIX, guardrails, slot-fill (~2240 стр.)
├── flow_policy.py          — active-flow логика, clear-хелперы
├── appointment_flow_guard.py — FSM записи (confirm/cancel/topic-switch)
├── state_mutations.py      — безопасные мутации SessionState
├── recovery_policy.py      — повторные уточнения / low-confidence
├── llm_mode_policy.py / llm_runtime.py / llm_doesnt_work_fallback.py — режимы LLM + деградация
├── dialog_graph.py         — диагностическая FSM (сейчас write-only, в trace)
├── memory.py               — pending/state-хранилище
├── renderer.py             — рендер ответов (детерм. + LLM rich-renderer) (~980 стр.)
├── response_builder.py     — Evidence → структурированные ответы
├── text_templates.py / service_phrase.py — шаблоны и фразы
├── self_check.py           — critic-гейт (само-проверка LLM-ответа)
├── prompt_contracts.py / prompt_registry.py — контракты и реестр prompt'ов
├── evidence_keys.py / mess_types.py / city.py / runtime_config.py
├── prompts/                — prompt'ы (ДУБЛЬ: см. ниже)
├── eval_suite/             — 62 эталонных диалога (critical/extended/server_parity/prepare_wrap)
├── scripts/                — eval-стадии, аудиты, проверка архитектурных импортов
└── services/               — доменный слой (бывший монолит services.py 7288 стр., распилен)
    ├── core.py             — Services dataclass + lifecycle + кеш-хелперы
    ├── _common.py          — базовые утилиты (_normalise_input, runtime_*)
    ├── prices.py / _prices_helpers.py     — цены, family-mode, модификаторы (~2360 стр.)
    ├── doctors.py / _doctors_helpers.py   — врачи, расписание, специальности (~1470 стр.)
    ├── prepare.py / _prepare.py           — PREPARE: скоринг + relevance-gate + LLM-wrap
    ├── lab_tests.py        — test_assist / test_result_status
    ├── addresses.py / _addresses_helpers.py / _regions.py — адреса, филиалы, часы
    ├── main_index.py       — индекс + налоговая справка
    └── news.py             — новости
```

### ⚠️ Prompt'ы в двух локациях (правило проекта)
Шаблоны рендерера/критика живут в **двух** местах и должны быть синхронны:
`messengers_router/prompts/*` **и** `app_data/prompts/mr_*`.
На 01.06 копии `renderer_patient*` и `renderer_critic_*` **байт-в-байт идентичны** (дрейфа нет). Если FT/ты трогаешь эти prompt'ы — меняй обе копии.

---

## 6. Сервисный слой и бэкенд (Nayka)

- Один бэкенд — **Nayka site API** (`agent_logic_2/nayka_api/api_nayka.py`), клиника в Самаре.
- `Services` (в `core.py`) — фасад с дневным кешем каталогов (врачи, цены, service_info, regions) в `agent_logic_2/nayka_api/apidata/*.jsonl` (датированные файлы).
- **Расписание:** `_fetch_schedule_source` различает (свежий коммит `39fb9cc`) **сбой CRM** (`ScheduleSourceUnavailable` → честный handoff/stale-fallback) от **«врача нет»** (пусто без handoff). Молчаливые сбои fetch теперь логируются WARNING.
- **Гео-граница:** онлайн-запись — только самарские филиалы; врачи могут вести приём и в Самаре, и в другом городе (тогда оставляем самарское присутствие, обрезаем регионы).

> Свежий каркас аудита API — [`nayka_site_api_live_audit_latest.md`](nayka_site_api_live_audit_latest.md).

---

## 7. Рендеринг и self-check

- Два пути: **детерминированный** рендер (`renderer.py` / `response_builder.py`) и **LLM rich-renderer** (prompt `renderer_patient_rich`), у которого override синхронизирован с bundle-дефолтами (`0decce5`).
- **Critic-гейт** (`self_check.py`): LLM-ответ прогоняется через критика на безопасность/соответствие; при «unsafe» → регенерация. Схему критика см. в `prompts/renderer_critic_patient_alignment.txt` (там сейчас есть мелкий битый JSON-пример — поправить, см. чекап).

---

## 8. Что изменилось за последние коммиты (≥10)

Свежий слой — это тема **«честность и надёжность вместо блефа»** + **анти-зацикливание**. Самое полезное для FT — перенять *поведенческие принципы*, а не реализацию.

**Честность по расписанию/записи (не обещать то, чего нет):**
- `39fb9cc` — отличать **сбой CRM** от **«врача нет»**: при сбое — честный handoff или stale-данные, не вводящее в заблуждение «расписание не найдено». Молчаливые сбои → WARNING.
- `b54853a` — честный ответ «слотов нет → оператор» в APPOINTMENT-превью (раньше падал в «расписание не найдено» на пустом расписании). Общий хелпер `_no_free_slots_operator_offer`.
- `009fe7f` — честный отказ для врача **не из самарского** каталога: вместо слепого списка из 5 филиалов — «запись через бота только по Самаре + оператор» (сигнал `doctor_lookup="unresolved"`).
- `277e561` — `TEST_RESULT` разрешён для любого города (обход самарского гейта — статус результата география не ограничивает).

**Анти-зацикливание / эскалация на оператора:**
- `ebaf794` + `ef354c5` — если бот повторяет одинаковый ответ N раз → интерактивный оффер оператора (O4).
- `acc8950` — но внутри активного APPOINTMENT-флоу repeat-guard пропускается (там повтор уместен).

**NLU:**
- `1f0acd3` — rescue: явный `rule.label=PRICE` спасается, когда LLM-merge уронил в `OTHER` (узкий гейт строго по OTHER, не перехватывает уверенные APPOINTMENT/ADDRESS).
- `fca8cef` — «чекап» держится **широким** запросом: это категория (Ежегодный/Мужской/Женский/…), а не одна услуга → `test_assist` возвращает всё семейство, а не одну строку.

**Lab / PREPARE / цены (per-item точность, без выдумок):**
- `703e3e8` — compound «ОАК, Ферритин, Витамин Д, …» резолвится **по каждому пункту** из каталога.
- `d448ac9` — prompt: **запрещено выдумывать** цены/сроки для multi-item lab-запросов.
- `0079851` + `462416d` + `e4a1a94` — «во сколько/когда сдать кровь» → памятка о заборе крови + адреса филиалов с часами; planner ставит `test_prepare` для запросов про тайминг визита.
- `5d82d3e` — сбрасывать конфликтующий устаревший анализ, если биоматериал отличается.
- `4e14715` — «антитела на корь» → «Вирус кори Ig M/Ig G».
- `3f9a5d1` — prompt: не предлагать «записать» на лабораторные анализы (их не бронируют).

**Адреса/часы:**
- `063c306` — восстановлены часы работы филиалов («График») из структурированных полей `/regions`.

---

## 9. Известные проблемы (НЕ переноси в FT)

Полный список и статус — [`messengers_router_checkup_2026-06-01.md`](messengers_router_checkup_2026-06-01.md). Статус на 2026-06-01:
- ✅ Гейт `pytest` снова **зелёный** (`841 passed, 0 failed`, ruff clean) — красный гейт закрыт.
- ✅ **HIGH:** H1 (грундер «профиль 1»→«профиль 2», `a50b11a`), H2 (протухший `no_free_slots` выбрасывал реальное расписание, `4e3469d`), H3 (`@lru_cache` кешировал «врач не найден», `1c609c7`).
- ✅ **MEDIUM:** M2 (multi-word несамарский город в гео-гейте), M3 (over-broad regex модификаторов цены `ген`/`экспресс`/`дет`), M4 (scorer по очищенным токенам — короткие аббревиатуры со стоп-словом), M5 (доступность не того врача при FIO-промахе), M6 (через снос `appointment_help`).
- ✅ **Чистка:** снесён недостижимый сервис `appointment_help`; мёртвый PREPARE-скоринг, осиротевшие state-сеттеры, nonbookable-кластер; битый JSON-литерал в critic-prompt (U8).
- 🟡 **Ещё открыто (маргинальное):** M1 (clarify_gate коды, спит под `legacy_v2`), edge «где сдать <abbrev>», мелкий dead-code/дублирование — см. [чекап](messengers_router_checkup_2026-06-01.md).

**Уроки для FT-дизайна:** (1) не кешируй то, что зависит от изменяемого каталога; (2) при множественных кандидатах сбрасывай «негативные» причины при успешном матче; (3) сравнивай запрос с услугой по *очищенным* токенам, не по сырой строке; (4) гео-гейты должны понимать multi-word значения от LLM.

---

## 10. Как запускать локально

```bash
# гейт (должен быть зелёным; сейчас НЕ зелёный — см. чекап)
venv/bin/ruff check messengers_router/
PYTHONPATH=. venv/bin/python -m pytest tests/ --ignore=tests/eval -q

# remote eval (к живому серверу) — один раз перед/после деплоя, не на каждый шаг
./run_remote_eval.sh --url http://172.16.0.16/api/messenger-generate-once
```
Прод — через Docker rebuild (`--build`) на тест-сервере после `release`.

> Кеш-нюанс: локальный pytest берёт датированные Nayka-кеши из `agent_logic_2/nayka_api/apidata/`. Если день «перекатился» — гидрируй текущие `doctors_*/price_*/service_info_*.jsonl`, иначе live-зависимые тесты упадут по среде, а не по логике.

---

## 11. Где правда

| Вопрос | Источник |
|--------|----------|
| Как работает MR сейчас | **этот файл** |
| Контракт сущностей | `russian_nlu.ENTITY_WHITELIST` (код — единственная правда) |
| Инструменты | `planner.py` + `executor.py` |
| Handoff-сообщения | `policies.HANDOFF_REASON_MATRIX` |
| Снимок модулей (апр.) | `CHANGELOG_MAR_APR_2026.md` |
| История рефакторинга | `messengers_router_refactor_plan.md` (журнал, частично устарел) |
| Баги/долг на 01.06 | `messengers_router_checkup_2026-06-01.md` |
| Бэкенд Nayka | `nayka_site_api_live_audit_latest.md` |
