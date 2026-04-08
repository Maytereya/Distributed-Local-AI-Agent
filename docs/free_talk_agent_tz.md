# ТЗ Draft: Free Talk Agent

Статус: draft v0.2  
Дата: 2026-04-08  
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

## 3. UX/функциональные требования

1. На вкладке `AI - ассистент` добавить режим:
   - `Free talk` (или `Свободное общение`)
   - в списке режимов перед `Call-Center-Ai` и `Messengers-Ai`.
2. Поведение режима:
   - если вопрос не про клинику: обычный разговорный ответ;
   - если вопрос про врачей/услуги/подготовку/цены/расписание/адреса: вызов нужной функции;
   - если в данных клиники ничего не найдено: честный ответ "в данных клиники не найдено";
   - опционально дать общее знание с меткой, что это не данные клиники.
3. Ответ должен быть коротко релевантным вопросу (без вываливания всего payload инструмента).

## 4. Предлагаемая архитектура (v1)

### 4.1 Новые модули

1. `agent_logic_2/free_talk/agent.py`  
   Оркестратор диалога: prompt -> tool loop -> финальный ответ.
2. `agent_logic_2/free_talk/tool_registry.py`  
   JSON-схемы инструментов + диспетчер вызовов в `Services`.
3. `agent_logic_2/free_talk/memory_redis.py`  
   Оперативная память диалога в Redis.
4. `agent_logic_2/free_talk/memory_persist.py`  
   Компактизация и запись long-term памяти.
5. `agent_logic_2/free_talk/prompts.py`  
   Загрузка системного промпта и prompt-шаблонов для summary.

### 4.2 Интеграция в существующий flow

1. Добавить режим в `assistant_tab.py` (`radio_type_of_search`).
2. Добавить ветку в `universal_echo` (`chat_modes.py`):
   - `if radio_value == "Free-talk-Ai": ...`
3. Для этого режима хранить отдельный `free_talk_session_state` (ID сессии) через `gr.State`.
4. Переиспользовать singleton `Services` (аналогично `messengers_router.endpoint.get_services`).

### 4.3 Алгоритм хода диалога

1. Принять `message`, `session_id`.
2. Прочитать Redis-контекст (tail истории + summary + user profile минимально).
3. Сформировать input для LLM: system prompt + context + user message.
4. Запросить у LLM JSON-действие:
   - `answer`
   - `tool_call` (tool + arguments)
5. Если `tool_call`:
   - провалидировать аргументы;
   - вызвать `Services.<tool>`;
   - положить tool result в контекст;
   - повторить цикл (не более `MAX_TOOL_STEPS`, например 3).
6. Сгенерировать финальный ответ пользователю.
7. Сохранить user/assistant turns в Redis.
8. Если достигнут порог compaction (по числу ходов/символов) - обновить summary и записать snapshot в постоянную память.

## 5. Tool-calling контракт (v1)

Контракт действия LLM в JSON:

```json
{
  "action": "answer | tool_call",
  "answer": "string",
  "tool": "string",
  "arguments": {}
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

## 6. Память

### 6.1 Оперативная память (Redis)

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

### 6.2 Постоянная память (long-term)

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

## 7. Системный промпт (черновик)

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

## 8. Изменения по файлам (план)

1. `agent_logic_2/gradio_ui/tabs/assistant_tab.py`
2. `agent_logic_2/gradio_ui/handlers/chat_modes.py`
3. `gradio_interface.py` (список доступных режимов для ролей)
4. `src/localragagent/freetalk/config.ini` + `src/localragagent/freetalk/config.py` (Redis и runtime конфиги)
5. `requirements.txt` (добавить `redis`, если отсутствует)
6. Новые файлы `agent_logic_2/free_talk/*`
7. `agent_logic_2/data/prompts/` + `app_data/prompts/` (seed prompt файлов)

## 9. Логи и наблюдаемость

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

## 10. Тесты (минимальный DoD)

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

## 11. План внедрения

1. Этап 1: skeleton + mode switch + базовый разговор без tools.
2. Этап 2: tool loop + интеграция `Services` (API/cache-first инструменты).
3. Этап 3: Redis memory + compaction + persist summary в jsonl.
4. Этап 4: тесты + стабилизация + эксплуатационные логи.
5. Этап 5 (после v1): опциональные инструменты на Meili и retrieval по Chroma.

## 12. Зафиксированные решения и открытые вопросы

1. UI mode key: `Free-talk-Ai` (фиксируем).
2. UI label для пользователя: `Свободное общение` (англ. `Free talk` можно оставить в скобках).
3. v1: только Gradio-режим, без отдельного API endpoint.
4. v1: handoff на оператора не требуется.
5. v1: Chroma не используем; только запись summary в постоянную память.
6. Открытый вопрос: включать ли в v1 `main_index_info/news_info` как опциональные Meili-инструменты или оставить их полностью выключенными до v2.
