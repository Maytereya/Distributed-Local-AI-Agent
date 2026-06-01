# `messengers_router` — фул-чекап (2026-06-01)

**Что это:** сквозной аудит модуля `messengers_router/` (~26.7k строк, ~60 файлов) на баги, мёртвый / дублирующийся / бесполезный код. Делался ДО изменений, как точка отсчёта.

**Метод:** `ruff` + полный прогон `pytest tests/ --ignore=tests/eval` + 5 параллельных read-only ревью-агентов по доменам (routing core / policies+flow / pricing+doctors / prepare+lab+address / rendering+NLU), каждый с проверкой находок по всему репозиторию (`rg` + чтение исходников). HIGH-баги подтверждены чтением кода или воспроизведением.

**Главное:**
- 🔴 **Гейт `pytest` КРАСНЫЙ на `release`:** `3 failed, 831 passed, 1 xfailed`. Два падения — детерминированные регрессии (воспроизводятся за <2 c на мок-данных, не зависят от кеша), одно — data-drift smoke-тест.
- 🟠 **3 HIGH-бага** в проде (один из них = одно из падений теста).
- 🟡 ~6 MEDIUM-багов (часть «спящие» — открываются при смене флага или multi-word входе).
- 🧹 Заметный слой мёртвого кода после рефакторинга Stage 1–22 (целый недостижимый сервис `appointment_help`, дублирующиеся скоринг-функции PREPARE, осиротевшие сеттеры/обёртки).
- ✅ Хорошие новости: `ruff` чистый кроме 2 известных импортов; копии prompt'ов в двух локациях **байт-в-байт идентичны** (дрейфа нет); `HANDOFF_REASON_MATRIX` без «дыр»; russian_nlu-консолидация завершена.

---

## ✅ Статус исправлений — обновлено 2026-06-01 (эта сессия)

Красный гейт и **все 3 HIGH-бага закрыты.** Гейт: было `3 failed, 831 passed` + 2× ruff F401 → стало **`839 passed, 0 failed`, ruff clean.** 7 атомарных коммитов на `release` (каждый с red→green-тестом; после каждого прогонялся полный гейт):

| Находка | Коммит | Статус |
|---|---|---|
| Падающий тест #1 — compound `compound_clarify`→`lab` | `7bea980` | ✅ исправлено |
| H1 / падающий тест #2 — walk-in «профиль 1»→«профиль 2» | `a50b11a` | ✅ |
| Падающий тест #3 — unit-имена дерматолог/челюстно-лицевой из live-кэша | `490a8e2` | ✅ |
| 2 мёртвых импорта (ruff F401: `typing.Any`, `canonicalise_unit`) | `6ee5922` | ✅ |
| **H2** — протухший `no_free_slots` выбрасывает реальное расписание | `4e3469d` | ✅ |
| **H3** — `@lru_cache` на `_extract_doctor_name` кеширует «врач не найден» | `1c609c7` | ✅ |

**Следующий шаг:** one-shot remote eval / Docker rebuild на этой партии (по eval-workflow) — деплой-чекпойнт перед дальнейшими правками.

**Ещё открыто** (разделы 2–5 ниже не тронуты): MEDIUM-баги M1–M6, мёртвый код (`appointment_help` и пр.), дублирование, useless/obsolete.

> Разделы 0–1 ниже оставлены как исходная картина «до» (для контекста); актуальный статус — в таблице выше.

---

## 0. 🔴 Красный гейт — падающие тесты (чинить первым)

Документированная gate-команда (`PYTHONPATH=. venv/bin/python -m pytest tests/ --ignore=tests/eval -q`) на текущем `release` даёт **3 failed**:

| # | Тест | Ожидалось → факт | Природа |
|---|------|------------------|---------|
| 1 | `test_messenger_services.py::test_service_bundle_info_compound_price_query_returns_clarify` | `service_kind == "compound_clarify"` → `"lab"` | **Детерминированная регрессия.** Мок-данные, воспроизводится за 1.9 c. Тест правился 2026-04-09 (`5f80306`) — значит сломалось позже. Compound-запрос (УЗДГ шеи + кровь на ЛПНП), который должен просить уточнение, теперь молча резолвится в «lab». |
| 2 | `test_router_flow_override.py::test_route_message_promotes_profile_followup_to_nonbookable_address` | `service_name == "анализы"` → `"диабетический профиль 2"` | **Детерминированная регрессия** = HIGH-баг H1 ниже. Пользователь спросил «профиль 1», бот подставил «профиль **2**». |
| 3 | `test_unit_canonicalisation.py::test_smoke_against_live_doctors_cache` | unit_name из живого кеша отсутствует в карте канонизации | **Data-drift / env.** Тест читает живой `doctors_*.jsonl` с диска; карта unit→specialty отстала от текущего прод-кеша. Не логический баг, но сигнал, что карту надо пополнить. |

> Расхождение с историей: рефактор-план и CHANGELOG заявляли «всё зелёное / 577 passed». Сейчас собрано **835 тестов**, и 2 из них — реальные регрессии. Либо гейт давно не прогонялся целиком, либо регрессия приехала с последними ~12 коммитами (трогали `services/lab_tests.py`, `services/prepare.py`, `response_builder.py`). Рекомендация: `git bisect` по тесту #1 (он быстрый и детерминированный).

---

## 1. 🟠 HIGH — баги, видимые пациенту / падение

### H1 — Подставляется чужое имя услуги: «профиль 1» → «профиль 2»
`entity_grounder.py:240-290` (`_should_keep_nonbookable_service_name_as_is`) + `service_phrase.py:90-170` (`extract_service_phrase`).
Для «Диабетический профиль 1 где можно сдать?» `extract_service_phrase` возвращает `None` (порядок слов «<услуга> где можно сдать» не ловится ни одним маркером), грундер падает в fuzzy-catalog match «Диабетический профиль **2**», а prefix-guard принимает его, потому что общий 4-символьный префикс «диаб» совпадает. Итог — пациенту называют не ту услугу. **Подтверждено падающим тестом #2.**
*Фикс:* ужесточить prefix-guard (отвергать, если числовой/квалификатор-токен расходится с тем, что сказал пользователь, либо требовать, чтобы токены оставляемого значения были подмножеством пользовательских) и/или научить `extract_service_phrase` порядку «<услуга> где сдать».

### H2 — «Нет слотов» затирает реальное расписание со свободными слотами
`services/doctors.py:813-820` (цикл) → `response_builder.py:371-372` (потребитель). **Подтверждено чтением кода.**
В `doctors_schedule_week` при кандидате «нет слотов» ставится `schedule_unavailable_reason = "no_free_slots_2_weeks"` и `continue`. Если **следующий** кандидат (вариант фамилии/`query_name`) матчится с реальным расписанием и делает `break` (строка 818-820), reason **не сбрасывается**. `return` (строки 897-901) отдаёт и реальный `schedule`, и протухший reason. `build_doctor_schedule_response` читает reason ПЕРВЫМ → отдаёт «слотов нет → оператор» и **выбрасывает расписание, в котором слоты есть**.
*Фикс:* `schedule_unavailable_reason = None` прямо перед каждым `break` по успешному матчу (или присваивать reason только после цикла, если ни один кандидат не совпал).

### H3 — `@lru_cache` на извлечении ФИО кеширует «врач не найден» навсегда
`classifier.py:303-304` — `@lru_cache(maxsize=512)` на `_extract_doctor_name(text, *, mode)`. **Подтверждено чтением кода** (никакого `cache_clear` в репозитории нет).
Результат зависит от изменяемого локального индекса врачей (`resolve_cached_doctor_name_candidate` → `_ensure_local_doctors_index_loaded`, перезагружается по mtime). Кеш ключуется только по `(text, mode)` и не инвалидируется: запрос фамилии в момент пустого/непрогретого индекса (холодный старт; JSONL появляется после первого промаха; врач, добавленный среди дня) закешируется как `None` и так и останется `None` до перезапуска процесса.
*Фикс:* убрать `@lru_cache`, либо включить сигнатуру кеша врачей в ключ, либо чистить из хука фонового рефреша.

---

## 2. 🟡 MEDIUM — баги (часть «спящие»)

### M1 — `clarify_gate` показывает машинные коды вместо вопроса
`orchestrator.py:436-442` + `:532-534`. При `decision.clarify_needed` `ctx.clarify_text = decision.clarify_reason`, а это машинный код из `{"", "intent_disambiguation", "slot_request", "context_repair", "low_confidence"}` (`classifier.py:75`). Пользователь увидит буквально `"slot_request"` (или пустое сообщение). **Спит** при дефолтном `MR_NLU_ENGINE=legacy_v2` (там `clarify_needed` не выставляется), **оживает** при `llm_primary`.
*Фикс:* в `clarify_gate` маппить коды → человеческий текст или вызывать `clarification_question(...)`.

### M2 — Multi-word несамарский город проходит сквозь гейт «город не поддержан»
`services/addresses.py:90` + `services/_regions.py:54-57`. `_extract_city_token` возвращает `None` для любого значения из >1 слова, поэтому `_is_non_samara_city_value("Самарская область" / "Нижний Новгород" / "Набережные Челны")` = `False`, гейт не срабатывает → отдаются самарские адреса. Достижимо: `city` — LLM-извлекаемая сущность, модель может выдать multi-word. Одно-словные (Москва, Тольятти) обрабатываются верно.
*Фикс:* в `_extract_city_token` принимать multi-word (паттерн «<прил.> область/край/республика» или трактовать любое несамарское multi-word как несамарское).

### M3 — Слишком широкие regex модификаторов цены ловят не те услуги
`services/_prices_helpers.py:85,93` (`ген\w*`), `:78,86` (`экспресс\w*`), `:80,88` (`дет\w*`).
- `ген\w*` (генетика) на реальных данных флагует «Гентамицин», «Гениопластика», «Удаление генитальных образований» — **202 из 338** «генетических» строк не генетические → штраф −120 и lab-variant-фильтрация легитимных услуг.
- `экспресс\w*` (cito) метит «Экспресс-тест ВИЧ/гепатит…» (23 из 31) как cito-надбавку → −35 и подавление при запросе без слова «экспресс».
- `дет\w*` (детский) ловит «**Дет**екция мутации BRAF».
*Фикс:* сузить корни — `генетическ\w*|генотип\w*|мутац\w*|полиморф\w*|\bген\b`; `детск\w*|детям|для детей|ребён\w*`; «экспресс-тест» (продукт) отделить от «cito/срочно» (надбавка).

### M4 — Скорер сравнивает с «шумной» полной строкой → короткие аббревиатуры теряются
`services/_prices_helpers.py:889,985`. `query = _normalise_input(query_text)` оставляет стоп-слова; бонусы `query == name` (+150) и head-match (+90) считаются по шумной строке. `_select_patient_price_rows("стоимость ттг")` → **пусто**, тогда как `("ттг")` → ТТГ@380. Основной `price_info` защищён пре-резолвом `service_name`, но `address_info` (`addresses.py:142`) может прокинуть сырой запрос → «где сдать СОЭ» вернёт пусто.
*Фикс:* в `_score_price_rows` брать `query` из `" ".join(_price_query_tokens(query_text))` (очищенный), а не из `_normalise_input`.

### M5 — `_doctor_availability_snapshot` приписывает слоты не тому врачу
`services/doctors.py:373-374`. При промахе по ФИО — `chosen = next(первая строка payload)` и её расписание выдаётся как у запрошенного врача. Для омонимичных фамилий покажет доступность врача B рядом с именем/ценой врача A в PRICE-карточках `service_bundle_info`. Параллельный путь `_schedule_by_specialty` (272-288) на промахе корректно `continue`. Внутреннее расхождение.
*Фикс:* на промахе возвращать пустой `availability_unmatched`, а не первую строку.

### M6 — `appointment_help` вернул бы сырой sentinel «совпадений не найдено» пациенту
`services/appointments.py:27-52`. Гейтит только `_is_meili_error_text`, но не `_is_meili_no_matches_text`, и не делает `.strip()` (ср. корректный двойной гейт в `main_index.py:87-90`). MEDIUM только потому, что сам сервис сейчас **недостижим** (см. D1) — при ре-подключении станет HIGH.
*Фикс:* добавить `_is_meili_no_matches_text`-ветку и `.strip()`, зеркально `main_index_info`.

### (LOW-bug) availability_error без ветки рендера
`services/prices.py:585` может выставить `note="availability_error"`, но `renderer.py:306-328` не имеет такой ветки → ошибочная проверка расписания выдаётся как обычное «статус уточняется».
*Фикс:* добавить явную ветку «расписание временно недоступно».

---

## 3. 🧹 Мёртвый / неиспользуемый код

| Что | Где | Заметка / как проверено |
|-----|-----|-------------------------|
| **Весь сервис `appointment_help` недостижим** | `services/appointments.py`, `core.py:601-602`, `executor.py:56-61`, `evidence_keys.py:29` | Планировщик никогда не эмитит `tool="appointment_help"` (перечислены все литералы `tool=`); `ek.APPOINTMENT` только пишется, в `response_builder` нет потребителя. Удалить или подключить. |
| `_prepare_from_analysis_api_cache` | `services/prepare.py:644-661` (+ `core.py:621,631`) | Привязан к `Services`, но не вызывается; дублирует живой inline-путь `_prepare_candidates_from_analysis_api_cache`. |
| `_choose_service_info_preparation` + `_service_info_row_score` | `services/_prepare.py:587-643` | Мёртвый кластер; дублирует живой скоринг в `prepare.py:235-264`. |
| `_static_nonbookable_branches` | `services/_addresses_helpers.py:214-230` | 0 вызовов во всём репо. |
| `_PREPARE_SERVICE_INFO_SYNONYMS = {}` | `services/_prepare.py:115` | Всегда пустой → цикл-расширение `_prepare.py:327-333` — гарантированный no-op. |
| 4 сеттера состояния без вызовов | `state_mutations.py:50,65,122,130` | `deactivate_appointment_flow`, `clear_appointment_confirm_pending`, `clear_appointment_cancel_pending`, `clear_appointment_topic_switch_pending` — остатки Stage-11 pilot. |
| `MemoryStore.save/load_dialog_state` | `memory.py:286,305` | Только в тестах; **коллизия ключа** `_dialog_state` с `dialog_graph` (dict vs str). |
| `dialog_graph.GraphEngine` — write-only | `dialog_graph.py` ← `router.py:2148,2157-2165` | Вычисленный `DialogState` идёт только в `debug_trace`; в проде никто не читает. «Явная FSM» сейчас декоративна. |
| `AppointmentPhase.CANCEL_CONFIRM` — фантом | `mess_types.py:111` | Объявлен и в `is_active()`, но `dialog.phase` им никогда не выставляется (cancel живёт во флаге `appointment_cancel_pending`). Фазовая машина расходится с флаговой. |
| 3 мёртвые обёртки response-builder | `router.py:2283,2315,2319` (+ импорт-алиасы `:63,64,66`) | `_build_service_bundle_response/_build_news_response/_build_main_index_info_response` — 0 вызовов (прод рендерит через `response_builder.build_first_structured_response`). |
| `_specialty_terms` | `services/_doctors_helpers.py:365-372` | 0 вызовов; тонкая обёртка над `_shared_specialty_terms`. |
| `recovery_policy._build_recovery_text` грузит prompt и выбрасывает | `recovery_policy.py:73-85` | `_ = prompt`, всегда возвращает статику. Мёртвая работа на каждом 2-м clarify. |
| `ek.ATTACHMENTS` читается, но не пишется | `evidence_keys.py:47` ← `orchestrator.py:575` | `attachments` в LLM-пути всегда `[]`. |
| Мёртвая ветка `reschedule` внутри `cancel`-блока | `response_builder.py:496` | После сужения блока до `cancel` (`137074ad`) тернарник «Перенос записи» недостижим. |
| Мёртвый tax-fallback-блок | `services/main_index.py:61-73` | После безусловного `return` на 58-59 `fallback_queries` всегда `[]`. |
| `_merge(llm_mode=...)` — неиспользуемый параметр | `nlu_pipeline.py:71` | Передаётся, но не читается. |
| Опечатка-ключ `" филиал"` (ведущий пробел) | `services/doctors.py:512` | Никто не выставляет → ветка недостижима. |
| 2 импорта (ruff F401) | `services/_doctors_helpers.py:36`, `services/_unit_canonicalisation.py:39` | `canonicalise_unit`, `typing.Any` — `ruff --fix`. |

---

## 4. ♻️ Дублирование

| Что | Где |
|-----|-----|
| `route_patient_message` — полная параллельная реализация пайплайна оркестратора, живёт только ради тестов; может тихо разъезжаться с `run_pipeline` (напр. doctor-guard prefetch отличается). **Известный долг.** | `router.py:2173-2240` |
| Фильтр lab-variant-флагов повторён 4–5× | `_prices_helpers.py:1205,1238,1331,1433,1640` → извлечь `_filter_rows_by_requested_flags()` |
| Tuple dedup-ключа цены пересобирается inline в ~5 местах | `_prices_helpers.py:935,1392,1514,1671`, `core.py:460` → `_price_row_key(row)` |
| Наборы стоп-слов триплицированы | `policies.py:379,363`, `flow_policy.py:70`, `llm_doesnt_work_fallback.py:50` |
| Два разных типа с именем `DialogState` (Enum vs dataclass) | `dialog_graph.py:25` / `mess_types.py:125` → переименовать Enum в `GraphState` |
| Сборка «самарские филиалы + часы» продублирована | `services/prepare.py:484-510` vs `services/addresses.py:67-75,185-213` → общий `_samara_branches_with_hours()` |
| `_normalise_text` дублирует `_common._normalise_input` (с точностью до `.strip()`) | `services/doctors.py:77` |

---

## 5. 🗑 Бесполезное / устаревшее

- **Битый JSON-пример в critic-prompt:** `tr`⏎`  ue` (literal `true` разорван переносом) — `prompts/renderer_critic_patient_alignment.txt:9-10` и идентичная копия `app_data/prompts/mr_renderer_critic_patient_alignment.txt`. Не падение (sanitize → `False`), но деградирует JSON-adherence критика и тратит регенерации. *Фикс:* `"unsafe_or_policy_violation": true|false,` одной строкой.
- **`topic_registry` keyword-матчер не ловит словоформы:** «цена» ≠ «цены», «акции» ≠ «акция» (`topic_registry.py:85-125`) → `exclude_keywords` дырявые. *Фикс:* снизить порог стемминга / нормализовать exclude-ключи к стемам.
- **Лишний speculative-вызов каталога для «чекап»** (`router.py:1127-1144`): `_plan_service_catalog_prefetch` не исключает TEST_ASSIST-категории, хотя `_inject_catalog_candidates` (`:1230-1233`) их всё равно отбрасывает. Перф, не корректность.
- `_RU_SUFFIXES` — дубли элементов (`«ией»` 4×, `«ам»/«ям»` 2×) — `llm_doesnt_work_fallback.py:110-113,125-126,131-132`.
- `__main__`-демо-блок «Дразнин» — `services/core.py:634-640`.
- Устаревший комментарий «Пока заглушка» при полной реализации `get_branches` — `services/core.py:514-564`.
- Временные migration-bridge'ы помечены «remove once… Phase 2», Stage 1–22 завершены — `mess_types.py:193-204`, `memory.py:289` (проверить писателей legacy-флагов, потом снять).
- Устаревший комментарий-блок про удалённый `_ensure_procedure_rows_loaded` — `services/addresses.py:316-322`.

---

## 6. Рекомендуемый порядок правок

1. **Вернуть гейт в зелёное** — разобрать 3 падающих теста (#1 — bisect регрессии compound-clarify; #2 = H1; #3 — пополнить unit→specialty карту или сделать тест-skip при дрейфе кеша).
2. **HIGH-баги H1/H2/H3** — каждый отдельным атомарным коммитом с red→green тестом.
3. **MEDIUM** — M2 (multi-word город) и M3 (regex модификаторов) ближе всего к пациенту; M1 закрыть до любого перехода на `llm_primary`.
4. **Чистка** — мёртвый код (раздел 3) можно сносить пачкой под `ruff` + полным `pytest`; начать с `appointment_help`, дублей PREPARE-скоринга и осиротевших сеттеров.
5. **Дублирование** (раздел 4) — оппортунистически, «migrate as you touch»; `route_patient_message` снести вместе с миграцией тестов на `run_pipeline`.

> Перед сносом мёртвого кода: эти находки проверялись `rg` по всему репо, но прод-деплой — через Docker rebuild и one-shot remote eval. Сначала зелёный локальный гейт (ruff + полный pytest), затем remote eval, потом деплой.
