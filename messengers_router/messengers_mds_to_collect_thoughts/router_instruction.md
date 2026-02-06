Контур в двух словах
Есть два независимых контура:
1. Call-center (старый): /v1/agent/stream →
agent_logic_2.router_preprocessor.routing(...) (SSE)
2. Messengers (новый):
• /api/messenger-generate (NDJSON stream)
• /api/messenger-generate-once (debug JSON)
→ messengers_router.patient_routing_stream(...)
Мессенджерный контур не должен тянуть промпты/логику сотрудников, чтобы не было
шума и побочных зависимостей.
ПОДРОБНОСТИ
1) Модули мессенджерного роутера и их
ответственность
1.1 messengers_router/endpoint.py
Задача: API-слой.
• Принимает {session_id, text} (Pydantic)
• Достаёт SessionState из MemoryStore
• Запускает patient_routing_stream(...)
• Возвращает:
• NDJSON stream: /api/messenger-generate
• Обычный JSON: /api/messenger-generate-once (для Swagger/ручных тестов)
Править тут, если:
• интеграторы жалуются на формат ответа/контент-тайп
• нужно изменить протокол (например: добавлять trace_id, event, final, и т.п.)
• нужно склеивание чанков / throttling (если слишком много мелких токенов)
• нужно включить auth на endpoint (API key/headers)
Не править тут:
правила — всё это ниже.
• бизнес-логику, определение label, поиск врачей/цен/расписаний, ургент-
1.2 messengers_router/router.py
Задача: главный оркестратор (маршрутизация).
Ключевые функции:
• route_patient_message(...) — принимает текст + состояние, возвращает:
• RouteDecision (label, flags, confidence, needs_handoﬀ)
• Plan (список шагов/инструментов)
• Evidence (данные, которые нашли)
• patient_routing_stream(...) — конвертирует решение в поток ResponseEnvelope:
• early exits: URGENT / COMPLAINT / MEDICAL_ADVICE
• pending-логика (чего не хватает — задаём уточняющий вопрос)
• auth_required — просим авторизацию
• иначе: render_stream(...) → поток текста
Править тут, если:
• неверно выбирается сценарий (label) или неверный “план действий”
• бот задаёт не те уточняющие вопросы (pending/missing)
• слишком часто ставится handoﬀ=true (не там где надо)
• надо изменить приоритет: “сначала ургент → потом жалобы → потом
остальное”
• надо добавлять “жёсткие правила” до LLM (например urgent keywords)
Логика правки при неверной реакции:
1. Проверяем label/flags/needs_handoﬀ (что решил классификатор)
2. Проверяем Plan (какие шаги должны выполниться)
3. Проверяем Evidence (какие данные реально нашли)
4. Смотрим, почему пошли в LLM (render_stream), а не в pending/early exit.
1.3 messengers_router/classifier.py
Задача: классификация запроса в один из Label, плюс флаги.
Есть два “уровня”:
• Hard rules (правила безопасности): URGENT/COMPLAINT/MEDICAL_ADVICE
должны отрабатывать до LLM.
• LLM classification (если нужно): для “мягких” кейсов.
Править тут, если:
• “у меня кровь” не попадает в URGENT
• жалобы не попадают в COMPLAINT
• любые медицинские советы попадают в LLM вместо безопасного отказа
• вопросы “как записаться” попадают в OTHER и вызывают handoﬀ
Логика правки:
• Добавить/изменить keyword rules для urgent/complaint/medical_advice
• Отрегулировать needs_handoﬀ:
• TRUE только для URGENT/COMPLAINT/MEDICAL_ADVICE/OTHER (и/или
низкого confidence)
• FALSE для “нормальных” справочных/запись/цены/адрес и т.п.
1.4 messengers_router/policies.py
Задача: “политики” безопасности и постобработка текста.
• sanitize_for_patient() — очищает служебные маркеры (важно: стрим-
безопасно, без .strip()).
• (опционально) правила отказов: medical advice, unsafe topics и т.п.
Править тут, если:
• “слипаются пробелы”, ломается форматирование, пропадают переносы
• нужно убрать из текста внутренние маркеры (“RAW_FULL”,
“SEGMENT_SEPARATOR”, “Заметка…”)
• нужно ограничить длину/тон/подачу
Важно:
• Любая “сильная чистка” в стриме ломает текст.
В стриме допустимо только удаление явных маркеров, без trim по краям.
1.5 messengers_router/renderer.py
Задача: генерация финального текста (LLM-wrap).
• Формирует _final_prompt(user_text, decision, evidence)
• Вызывает Ollama streaming (ollama_call)
• Отдаёт поток delta → sanitize_for_patient(delta)
• render_urgent/render_complaint/render_medical_advice — шаблонные ответы
Править тут, если:
• LLM придумывает факты (нужен более жёсткий промпт)
• нужно изменить стиль ответа (короче, дружелюбнее, списки и т.п.)
• нужно убрать “галлюцинации”: усилить правило “только из evidence”
• нужно менять стратегию стрима (дельты/частота/коалесинг — но лучше в
endpoint)
1.6 messengers_router/services.py
Задача: “инструменты/интеграции” (доступ к данным клиники).
Это слой, где должны быть методы вида:
• search_doctors(...)
• get_doctor_schedule(...)
• get_prices(...)
• get_addresses(...)
• get_promotions(...)
• get_test_result_pdf(patient_token, ...) (авторизация)
Править тут, если:
• роутер правильно понял intent, но данные не находятся
• надо поменять, из какой БД/индекса искать (Meili/Chroma/CRM)
• надо подключить реальные источники
1.7 messengers_router/memory.py
Задача: состояние диалога.
• хранит SessionState по session_id
• TTL, pending TTL
• история и last_entities (чтобы не спрашивать одно и то же)
Править тут, если:
• “забывает” контекст слишком быстро / слишком долго
• pending не сохраняется / сохраняется неправильно
• нужно добавить очистку персональных данных (PII)
1.8 messengers_router/mess_types.py
Задача: типы домена (dataclass).
• SessionState, RouteDecision, Plan, Evidence, ResponseEnvelope
Править тут, если:
• добавляем новые сущности (например trace_id, event_type, final)
• надо сделать attachments строгими (типизированными)
• меняется контракт между модулей
2) Как дебажить “неверную реакцию” (пошагово)
Шаг A — воспроизведение
Самый быстрый путь:
• Swagger debug endpoint: /api/messenger-generate-once
• или messenger_simulator.py (консоль)
Шаг B — смотреть decision/plan/evidence (нужен trace)
Сейчас Evidence.debug_trace есть, но он не отдается.
Рекомендую добавить временный режим debug=true в request:
• сохранять “решения” в debug_trace
• отдавать в state_update (только для debug endpoint)
Минимальная практика:
• если label неверный → правим classifier/router
• если label верный, но данные пустые → правим services
• если данные есть, но LLM врёт → правим renderer prompt / политики
3) Что сейчас нужно подключить
Сейчас работает “скелет”. Чтобы он стал “боевым”, нужно подключить именно Services-
слой:
Обязательно (для реальной пользы)
1. Расписание врачей (DOCTOR_SCHEDULE)
2. Список врачей/специализации (DOCTOR_INFO)
3. Прайс (PRICE)
4. Адреса/филиалы (ADDRESS)
5. Новости/акции (NEWS)
6. Подготовка к анализам (PREPARE) — если есть база
Высокий приоритет, но сложнее
7. TEST_RESULT — выдача PDF результатов:
• нужен patient token / авторизация
• нужен сервис генерации ссылки/временного URL (или отдача файла)
• нужен юридически корректный флоу “мед. тайна”
Опционально
8. APPOINTMENT — запись:
• интеграция с CRM/регистратурой
• или handoﬀ на оператора
4) Как править
“карту симптомов”
Симптом: URGENT не срабатывает
→ classifier/router: hard-rule до LLM
Симптом: бот ставит handoﬀ в обычных кейсах
→ classifier: needs_handoﬀ только для нужных label или по confidence
→ router: не отправлять handoﬀ, если идёт уточняющий вопрос
Симптом: “нет информации” там, где должна быть
→ services: поиск не подключён / возвращает пусто
→ router: возможно, pending не срабатывает (не спросили дату/филиал)
Симптом: LLM придумывает врачей/цены/адреса
→ renderer: усилить промпт (“только факты из evidence”)
→ policies: жёстче чистить “выдумки” нельзя, лучше предотвращать промптом
Симптом: текст ломается/слипается
→ policies.sanitize_for_patient: убрать strip/squeeze на дельтах
Суммарно
1. Любая правка начинается с того, чтобы понять:
• какой label выставился
• были ли missing fields
• какие данные реально найдены
2. Не лечить “ошибочный label” промптом.
Промпт — это только упаковка данных, не логика.
3. Hard-safety (urgent, medical advice) — только правилами, не LLM.