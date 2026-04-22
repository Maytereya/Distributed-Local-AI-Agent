# ТЗ: Free Talk Agent

Статус: v0.4 (синхронизировано с кодом)  
Дата: 2026-04-15  
Контекст: новый режим для вкладки `AI - ассистент` в Gradio.

## 1. Цель

Сделать новый режим агента `Свободное общение (Free talk)`, который:

1. Поддерживает естественный диалог на любые темы.
2. При вопросах про клинику вызывает функции доступа к данным клиники через сервисный слой `messengers_router/services.py` и встроенный `service_catalog` (агрегация `priceByRegion` + `doctorServicePricesByRegion`).
3. Имеет оперативную память в Redis.
4. Умеет компактизировать историю и сохранять ее в постоянную память.
5. Явно разделяет:
   - факты из данных клиники;
   - общие знания модели (с обязательной пометкой).

## 2. Что уже есть в проекте и переиспользуем

1. UI-переключатель режимов:  
   `agent_logic_2/gradio_ui/tabs/assistant_tab.py` + `agent_logic_2/gradio_ui/handlers/chat_modes.py`.
2. Готовый слой данных клиники:  
   `messengers_router/services.py`.
   Важно: отдельного модуля `clinic_catalog` в проекте нет, фактическая реализация каталога — это `service_catalog` внутри `Services`:
   - `_ensure_service_catalog_rows_loaded` (сбор объединенного каталога услуг);
   - `match_catalog_service`, `match_catalog_doctor`, `get_catalog_health`;
   - `resolve_price_service_name_from_catalog` (price-grounding).
3. Готовые функции, которые можно вызывать из Free Talk без дублирования API-логики:
   - `doctors_info`, `doctors_schedule_week`,
   - `price_info`, `service_bundle_info`,
   - `test_prepare`, `test_assist`, `test_result_status`,
   - `address_info`.
4. Источники и кэши уже реализованы в `agent_logic_2/nayka_api/*`:
   - `api_price.load_price_by_region`, `api_price.load_doctor_prices`;
   - `api_service_info.load_service_info`;
   - `api_nayka.get_cached_doctors_data`, `api_nayka.site_regions`.
5. Текущая in-memory память сессий (как reference):  
   `messengers_router/memory.py`.
6. Runtime вызовов LLM и очередь:  
   `messengers_router/llm_runtime.py`.

Вывод: новый режим строим поверх `Services`, не дублируем API-логику из `agent_logic_2/llama_func_call.py`.

## 3. Фактическая реализация (на 2026-04-15)

1. Основной package FT:
   - `src/localragagent/freetalk/agent.py` — публичный фасад `FreeTalkAgent`
   - `src/localragagent/freetalk/orchestrator.py` — основной runtime flow
   - `src/localragagent/freetalk/routing_contract.py`
   - `src/localragagent/freetalk/routing_prompting.py`
   - `src/localragagent/freetalk/tool_planning.py`
   - `src/localragagent/freetalk/adapter.py`
   - `src/localragagent/freetalk/adapter_contracts.py`
   - `src/localragagent/freetalk/tool_dispatcher.py`
   - `src/localragagent/freetalk/dialog_state.py`
   - `src/localragagent/freetalk/followup_policy.py`
   - `src/localragagent/freetalk/memory_policy.py`
   - `src/localragagent/freetalk/memory_redis.py`
   - `src/localragagent/freetalk/memory_persist.py`
   - `src/localragagent/freetalk/rendering.py`
   - `src/localragagent/freetalk/system_prompt.txt`
   - `src/localragagent/freetalk/config.ini`
2. Порты FT:
   - `src/localragagent/ports/freetalk_services_port.py` (к `messengers_router.Services`)
   - `src/localragagent/ports/freetalk_web_search_port.py` (SearXNG)
   - `src/localragagent/ports/freetalk_llm_port.py` (LLM runtime)
3. Введен `clarification-first` контур с явным routing-contract слоем:
   - `src/localragagent/freetalk/routing_contract.py`
   - `src/localragagent/freetalk/routing_prompting.py`
   - intent/confidence/entities/missing_slots/clarify_question/tool_plan
   - обязательный уточняющий шаг при недостатке данных
   - мягкий retry после пустого результата инструментов
4. Добавлено контекстное follow-up поведение:
   - `dialog_state` + `session_entity_memory`
   - doctor/service/address/result follow-up логика
   - корректная обработка коротких продолжений и topic shift
5. Введен source-tagging ответов:
   - внутренние `source_fragments` с тегами `clinic_data | general_knowledge | web_search`
   - запись тегов в trace/логи (`event=answer_source_trace`)
   - сохранение тегированных фрагментов в turn-историю Redis (UI при этом остается текстовым).
6. Добавлен eval quality-gate FT (in-process):
   - датасет: `tests/eval/freetalk_dialogs.jsonl`
   - пороги: `tests/eval/freetalk_quality_gate.json`
   - раннер: `tools/freetalk_quality_gate.py`
   - запуск в CI: `.github/workflows/messengers_router_arch_guardrails.yml`.

## 4. UX/функциональные требования

1. На вкладке `AI - ассистент` добавить режим:
   - `Free talk` (или `Свободное общение`)
   - в списке режимов перед `Call-Center-Ai` и `Messengers-Ai`.
2. Поведение режима:
   - если вопрос не про клинику: обычный разговорный ответ;
   - если вопрос про врачей/услуги/подготовку/цены/расписание/адреса: вызов нужной функции;
   - если в данных клиники ничего не найдено: честный ответ "в данных клиники не найдено";
   - опционально дать общее знание с меткой, что это не данные клиники.
3. Ответ должен быть коротко релевантным вопросу (без вываливания всего payload инструмента).

## 5. Архитектура (v1, актуальная)

### 5.1 Модули

1. `src/localragagent/freetalk/agent.py`  
   Публичный фасад `FreeTalkAgent`.
2. `src/localragagent/freetalk/orchestrator.py`  
   Оркестратор диалога: routing -> clarification -> tool loop -> финальный ответ.
3. `src/localragagent/freetalk/routing_contract.py`  
   Нормализация routing-решения (`intent/confidence/entities/missing_slots/tool_plan`).
4. `src/localragagent/freetalk/routing_prompting.py`  
   Router/verifier prompt builders.
5. `src/localragagent/freetalk/tool_planning.py`  
   Эвристический fallback-планировщик и маршрутизация meili/web-сигналов.
6. `src/localragagent/freetalk/adapter.py`  
   Явный FT -> backend adapter для clinic domains.
7. `src/localragagent/freetalk/memory_redis.py`  
   Оперативная память диалога в Redis.
8. `src/localragagent/freetalk/memory_persist.py`  
   Компактизация и запись long-term памяти.

### 5.2 Интеграция в существующий flow

1. Добавить режим в `assistant_tab.py` (`radio_type_of_search`).
2. Добавить ветку в `universal_echo` (`chat_modes.py`):
   - `if radio_value == "Free-talk-Ai": ...`
3. Для этого режима хранить отдельный `free_talk_session_state` (ID сессии) через `gr.State`.
4. Переиспользовать singleton `Services` (аналогично `messengers_router.endpoint.get_services`).

### 5.3 Алгоритм хода диалога (clarification-first)

1. Принять `message`, `session_id`.
2. Прочитать Redis-контекст (tail истории + summary + session meta/pending).
3. Сформировать input для LLM clinical-router и получить JSON-решение:
   - `intent`, `confidence`, `entities`,
   - `missing_slots`, `clarify_question`, `tool_plan`.
4. Если `missing_slots` не пустой или уверенность ниже порога:
   - задать уточняющий вопрос;
   - сохранить `clinical_dialog_state` в Redis;
   - завершить текущий ход.
5. Если данных достаточно:
   - выполнить tool-plan (до `MAX_TOOL_STEPS`);
   - рендерить ответ строго из tool payload.
6. Если tools не дали результата:
   - один мягкий уточняющий шаг;
   - при повторном пустом результате — честный `not found`.
7. Сохранить user/assistant turns в Redis.
8. При пороге compaction обновить summary и записать snapshot в постоянную память.

## 6. Tool-calling контракт (v1)

Контракт решения routing-contract слоя (JSON):

```json
{
  "intent": "doctor_schedule | doctor_info | price | prepare | tests | test_result | address | clinic_documents | clinic_news | service_info | unknown",
  "confidence": 0.0,
  "entities": {},
  "missing_slots": [],
  "clarify_question": "",
  "tool_plan": []
}
```

Правило маршрутизации для медицинских тем:
1. Если запрос про врачей, услуги, процедуры, подготовку, цены, анализы, расписание, адреса филиалов или результаты анализов - модель обязана сделать `tool_call`.
2. Если подходящий tool не выбран из-за нехватки данных - сначала задать уточняющий вопрос, а не отвечать «из головы».
3. Если tool вернул пусто/не найдено - явно сообщить, что в данных клиники нет результата.

Инструменты v1 (основной API/cache-first набор, без обязательного Meili):

1. `match_catalog_service(raw_text_or_name, current_service_name="")`
2. `match_catalog_doctor(raw_text_or_name)`
3. `get_catalog_health()`
4. `service_bundle_info(query, entities, top_n=None)`
5. `price_info(query, entities)`
6. `test_prepare(query, entities)`
7. `test_assist(query, entities)`
8. `doctors_info(query, entities)`
9. `doctors_schedule_week(query, entities)`
10. `address_info(query, entities)`
11. `test_result_status(query, entities)`

Опциональные инструменты (включаются отдельно, не часть мед-ядра v1):

1. `main_index_info(query, entities)` - основан на Meili.
2. `news_info(query, entities)` - основан на Meili.
3. `web_search(query, entities)` - интернет-поиск через SearXNG.

Примечание: `entities` заполняем минимально и безопасно (без попытки продублировать весь NLU `messengers_router`).

## 7. Память

### 7.1 Оперативная память (Redis)

Технически:

1. Библиотека: `redis` (async API, `redis.asyncio`).
2. Конфиг хранится явно в package-файле: `src/localragagent/freetalk/config.ini`.
3. Базовые параметры:
   - `redis_url = redis://redis:6379/0`
   - `redis_prefix = ft`
   - `session_ttl_sec = 86400`
   - `web_search_url = http://searxng:8080`
   - `enable_web_search_tool = true`
   - для SearXNG обязательно включить `search.formats: [html, json]` в `settings.yml`
   - для SearXNG обязательно задать `server.secret_key` (не `ultrasecretkey`)
   - `web_search_healthcheck_timeout_s = 3`
   - `web_search_healthcheck_ttl_s = 30`
   - `context_window_tokens = 24576`
   - `context_warn_ratio = 0.82`
   - `context_response_reserve_tokens = 2048`
4. Ключи:
   - `ft:session:{session_id}:turns` (list/json turns)
   - `ft:session:{session_id}:summary` (string)
   - `ft:session:{session_id}:meta` (hash/json)

### 7.2 Постоянная память (long-term)

v1 (простой и надежный):

1. Файл: `app_data/free_talk_memory/dialog_summaries.jsonl`
2. Запись snapshot:
   - `session_id`
   - `ts`
   - `summary`
   - `key_facts`
   - `open_loops`
3. Запись атомарным append (через temp + rename при необходимости).

v2 (опционально): индексировать summary в отдельную коллекцию Chroma для retrieval между долгими диалогами.

## 8. Системный промпт (черновик)

```text
Ты разговорный AI-ассистент клиники.

Правила:
1) Отвечай свободно и дружелюбно на любые темы, если вопрос не требует данных клиники.
2) Если вопрос относится к медицинским данным клиники (врачи, услуги, процедуры, подготовка, прайс, анализы, расписание, филиалы, результаты) —
   сначала ОБЯЗАТЕЛЬНО вызови подходящий tool и отвечай по его результату.
3) Если пользователь просит поиск в интернете/актуальные внешние данные, используй tool `web_search`.
4) Не придумывай данные клиники и не отвечай по памяти модели там, где нужен tool.
5) Используй только релевантные фрагменты результата tool, не выводи служебный JSON целиком.
6) Если данных клиники не найдено, явно сообщи об этом.
7) Если после этого даешь общий ответ из собственных знаний, обязательно пометь:
   "Это общая информация, не из данных клиники."
8) Для ответов из `web_search` обязательно помечай, что это данные интернет-поиска.
9) Если уверенности недостаточно и tool не помог, предложи уточняющий вопрос.
10) Если пользователь спрашивает о возможностях агента ("кто ты", "расскажи о себе"),
    дай краткую инструкцию с примерами формулировок запросов.
11) При близком заполнении контекста запускай сценарий подтверждения:
    сначала "завершить сейчас? Да/Нет", при "Нет" разрешить еще одно сообщение,
    затем финальный выбор: удалить диалог или сохранить compact summary и начать новую сессию.
```

## 9. Изменения по файлам (актуально)

1. `agent_logic_2/gradio_ui/tabs/assistant_tab.py`
2. `agent_logic_2/gradio_ui/handlers/chat_modes.py`
3. `gradio_interface.py` (список доступных режимов для ролей)
4. `src/localragagent/freetalk/config.ini` + `src/localragagent/freetalk/config.py` (Redis и runtime конфиги)
5. `requirements.txt` (добавить `redis`, если отсутствует)
6. Новые файлы `src/localragagent/freetalk/*`
7. `agent_logic_2/data/prompts/` + `app_data/prompts/` (seed prompt файлов)

## 10. Логи и наблюдаемость

Минимум логов для `Free talk`:

1. `session_id`, `mode`, `tool_called`, `tool_status`, `latency_ms`.
2. `memory_compaction_triggered`, `summary_chars`, `persist_status`.
3. Флаг источника ответа:
   - `source=clinic_data`
   - `source=general_knowledge`
   - `source=mixed`
4. Формат событий:
   - префикс: `FreeTalkAI`
   - `component=freetalk|ports`
   - ключи вида `event=... session_id=... error_type=...`
5. Для расписания добавлена диагностика payload:
   - `event=medical_schedule_payload_stats`
   - поля: `doctors_count`, `regions_count`, `days_count`, `slots_count`.

## 11. Тесты (минимальный DoD)

1. Unit:
   - выбор/валидация `tool_call`;
   - fallback при пустых данных инструмента;
   - Redis read/write/TTL;
   - compaction + persist snapshot.
2. Integration:
   - режим `Free talk` работает через `universal_echo`;
   - переключение mode не ломает `Call-Center-Ai` и `Messengers-Ai`.
3. Contract:
   - если ответ не из клиники, обязательная метка в тексте.
4. Router:
   - тесты `routing_contract/routing_prompting` на intent mapping/missing slots/clarification.
5. Eval профили:
   - `deterministic` (без внешней LLM, фиксированные `router_decision` из датасета):
     - `python3 tools/freetalk_quality_gate.py --profile deterministic`
   - `production-like` (live LLM-router/verifier, без фиксированных `router_decision`):
     - `python3 tools/freetalk_quality_gate.py --profile production-like`
   - для `production-like` требование: рабочий `messengers_router.llm_runtime` с доступом к Ollama.

## 12. Следующие шаги (v1.1)

1. Стабилизировать intent-router на реальных диалогах (doctor/service follow-up).
2. Уточнить пороги confidence/clarify-retries для выбранной модели.
3. Добавить A/B-профили параметров Ollama для разных моделей.
4. После стабилизации — рассмотреть retrieval-слой Chroma (v2).

## 13. Зафиксированные решения и открытые вопросы

1. UI mode key: `Free-talk-Ai` (фиксируем).
2. UI label для пользователя: `Свободное общение` (англ. `Free talk` можно оставить в скобках).
3. Доступен отдельный FT API endpoint для eval/debug:
   - `/v1/freetalk/generate-once`.
4. FT поддерживает `handoff` как terminal state: при handoff FT полностью очищает память текущей сессии и выдаёт новый `session_id`.
5. v1: Chroma не используем; только запись summary в постоянную память.
6. `main_index_info/news_info` поддерживаются как опциональные и управляются через `include_meili_tools`.
7. Открытый вопрос: оставить ли LLM-router + heuristic fallback или перейти на pure LLM-router без regex-предмаршрутизации.

## 14. План улучшений (приоритетный, согласованный)

### 14.1 Единый контракт `dialog_act` на каждый ход

Цель: убрать разрозненные решения по ходу и централизовать исполнение такта диалога.

Базовый контракт:

```json
{
  "route": "general | clinical | web",
  "intent": "doctor_schedule | doctor_info | price | prepare | tests | service_info | clinic_news | clinic_documents | unknown",
  "entities": {},
  "confidence": 0.0,
  "missing_slots": [],
  "clarify_question": "",
  "tool_plan": [],
  "response_policy": "tool_only | mixed | general_only"
}
```

Правила исполнения:

1. Один ход -> один `dialog_act`.
2. Исполнитель (`dialog_act executor`) является единственной точкой, где решается:
   - задаем уточнение;
   - вызываем tools;
   - отвечаем сразу;
   - делаем fallback.
3. Никаких параллельных "скрытых" веток принятия решения вне исполнителя.

### 14.2 Regex-эвристики только как fallback

Цель: снизить ложные срабатывания и конфликт между эвристикой и LLM-router.

Правило:

1. Основное решение: LLM-router (`dialog_act`).
2. Regex/keyword слой включается только если:
   - LLM вернула `unknown`;
   - confidence ниже порога;
   - JSON невалиден.
3. Любая эвристическая коррекция должна логироваться отдельным событием:
   - `event=router_fallback_applied`
   - `reason=low_confidence|invalid_json|unknown_intent`.

### 14.3 Обязательный `post-tool verifier`

Цель: после tool-вызовов гарантировать корректный тип ответа: финальный ответ vs уточнение.

Контракт verifier (короткая проверка):

```json
{
  "enough_data": true,
  "should_clarify": false,
  "clarify_question": "",
  "answer_policy": "direct | clarify | not_found"
}
```

Правила:

1. Если `enough_data=true` -> прямой ответ только по payload.
2. Если `should_clarify=true` -> один уточняющий вопрос.
3. Если данных нет и уточнять нечего -> честный `not_found` без галлюцинаций.

### 14.4 Eval-набор FT и quality gate в CI

Цель: оценивать качество FT на реальных диалогах до merge.

Решение v1:

1. Собрать набор анонимизированных реальных FT-диалогов (`tests/eval/freetalk_dialogs.jsonl`).
2. Ввести метрики:
   - `tool_call_recall` для клинических интентов;
   - `hallucination_rate_clinic`;
   - `clarification_appropriateness`;
   - `answer_grounded_rate`.
3. Добавить CI-job, который прогоняет eval и фейлит pipeline при деградации выше порога.

Примечание: отдельный HTTP endpoint для этого не обязателен на первом этапе; достаточно in-process harness в тестах/скрипте CI.

### 14.5 Source-tagging на уровне фрагментов ответа

Цель: структурно разделить источники, чтобы снизить смешение "данные клиники" и "общие знания".

Внутренний формат фрагментов:

```json
[
  {"text": "...", "source": "clinic_data"},
  {"text": "...", "source": "general_knowledge"},
  {"text": "...", "source": "web_search"}
]
```

Правила:

1. Рендер в UI может оставаться текстовым, но tags сохраняются в лог/trace.
2. Для `general_knowledge` и `web_search` обязательна явная пометка в тексте ответа.
3. Source-tagging повышает управляемость и проверяемость ответов, но не заменяет NLU модели; это механизм контроля качества, а не "ускоритель понимания".
