# Architecture Contract (Messenger Router)

## 1. Что делает каждый модуль

- `/Users/maxten/Dev/Distributed-Local-AI-Agent2/messengers_router/endpoint.py`
  - HTTP-слой (`/api/messenger-generate`, `/api/messenger-generate-once`), без бизнес-логики.
- `/Users/maxten/Dev/Distributed-Local-AI-Agent2/messengers_router/router.py`
  - Оркестратор: decision -> plan -> evidence -> pending/handoff -> renderer.
- `/Users/maxten/Dev/Distributed-Local-AI-Agent2/messengers_router/classifier.py`
  - Классификация и первичное извлечение entities, hard-rules + LLM fallback.
- `/Users/maxten/Dev/Distributed-Local-AI-Agent2/messengers_router/policies.py`
  - Централизованные policy: intent-детекторы, slot policy, clarify-тексты, handoff policy, appointment state transitions.
- `/Users/maxten/Dev/Distributed-Local-AI-Agent2/messengers_router/services.py`
  - Интеграции (Nayka API, price, meili), кэши, fallback-контракт ошибок.
- `/Users/maxten/Dev/Distributed-Local-AI-Agent2/messengers_router/renderer.py`
  - Упаковка финального ответа пациенту на основе evidence (без смены маршрута).
- `/Users/maxten/Dev/Distributed-Local-AI-Agent2/messengers_router/memory.py`
  - Состояние диалога и pending по `session_id`.
- `/Users/maxten/Dev/Distributed-Local-AI-Agent2/messengers_router/mess_types.py`
  - Доменные типы/контракт между слоями.

## 2. Границы ответственности (чтобы не размывать логику)

- Label и маршрут выбираются только в `classifier.py` + `router.py`.
- Вопросы вида "чего не хватает" и handoff-тексты задаются policy-функциями из `policies.py`.
- `renderer.py` не исправляет ошибочный label и не подменяет бизнес-решения.
- `services.py` не должен решать intent; он только получает данные и отдает унифицированный payload.
- Все внешние ошибки переводятся в контролируемый fallback (`handoff_required`, `handoff_reason`, `handoff_message`), без `500`.

## 3. Контракт TEST_RESULT (текущий)

- Сценарий сбора полей: `surname`, `year`, `filial`, `number`.
- Пока не собраны все поля: бот задает уточняющие вопросы (без немедленного handoff).
- После сбора: запрос в `resultForPatient`.
- Handoff только если источник данных недоступен/ошибка интеграции.

## 4. Контракт prompts

- `/Users/maxten/Dev/Distributed-Local-AI-Agent2/messengers_router/prompts/classifier_patient.txt`
  - Только канонические labels.
  - Явный `context_action`.
  - Few-shot на реальных пациентских формулировках.
- `/Users/maxten/Dev/Distributed-Local-AI-Agent2/messengers_router/prompts/renderer_patient.txt`
  - Только стиль и ограничения ответа, без маршрутизации.
