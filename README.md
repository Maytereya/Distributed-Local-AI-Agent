# LocalRAGagent / Neiry Agent API

Текущая версия проекта: production-ориентированный API-контур для клиники с двумя независимыми сценариями:

- `for-messengers` — пациентский роутер (`messengers_router`) с stateful flow-логикой.
- `for-call-center` — SSE-стрим для операторского интерфейса.

Основной фокус текущего цикла: стабильность `messengers_router` v2, архитектурные guardrails и воспроизводимый remote eval.

## Что сейчас умеет система

### 1) Контур мессенджеров (`messengers_router`)

- Интенты: `APPOINTMENT`, `DOCTOR_SCHEDULE`, `DOCTOR_INFO`, `PRICE`, `ADDRESS`, `TEST_ASSIST`, `TEST_RESULT`, `PREPARE`, `NEWS`, `OTHER`, `URGENT`, `COMPLAINT`, `MEDICAL_ADVICE`.
- Stateful-диалог: память сессии, pending-слоты, подтверждение/отмена записи, topic-switch guardrails.
- NLU v2 pipeline с feature flags (`legacy_v2` / `llm_primary`) и debug trace.
- Entity grounding перед merge в state.
- Детерминированная сборка ответов через `response_builder.py`.
- Handoff-политики для high-risk сценариев.
- Архитектурный gate (циклы/границы импортов) + CI workflow.

### 2) Контур колл-центра (`/v1/agent/stream`)

- SSE-стрим ответа для операторов.
- Отдельный протокол от мессенджеров.
- Защита `X-API-Key`.

## HTTP API (актуально)

### For Messengers

- `POST /api/messenger-generate`
  - NDJSON stream (`application/x-ndjson`)
  - для Telegram/WhatsApp/Web-chat интеграций

- `POST /api/messenger-generate-once`
  - единый JSON-ответ
  - поддерживает `debug=true` и возвращает `state_update.debug`

### For Call Center

- `POST /v1/agent/stream`
  - SSE (`text/event-stream`)
  - требует `X-API-Key`

## Быстрый старт (локально)

### 1) Установка зависимостей

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Конфигурация

- Основная конфигурация: `agent_logic_2/config.ini`
- API-ключ для call-center endpoint: `AGENT_API_KEY` (используется в `agent_api.py`)
- Prompt override каталог: `app_data/prompts/` (опционально)

### 3) Запуск API

```bash
uvicorn agent_api:app --host 0.0.0.0 --port 8000 --reload
```

### 4) Smoke-проверка мессенджерного endpoint

```bash
curl -s http://localhost:8000/api/messenger-generate-once \
  -H 'Content-Type: application/json' \
  -d '{
    "session_id":"s_local_smoke",
    "text":"покажи расписание уролога",
    "debug":true,
    "llm_mode":"hybrid"
  }' | jq .
```

## Качество и регрессия

### Локальные тесты

```bash
PYTHONPATH=. pytest -q tests
```

### Архитектурный guardrail

```bash
python3 messengers_router/scripts/check_architecture_imports.py
```

### Remote eval (целевой gate: 100%)

```bash
bash messengers_router/eval_suite/run_remote_eval.sh \
  --url http://<server>/api/messenger-generate-once
```

`run_remote_eval.sh` запускает `stage1`, `stage3`, `stage4`, `stage5`, `critical` и затем `coverage_ext`.
Логи складываются в `messengers_router/eval_suite/logs/<run_id>/`.

## Текущий статус качества (2026-03-21)

- Локальный `pytest`: `150 passed`
- Последний server remote eval: `run_id=1774094099`
  - `stage1`: 20/20
  - `stage3`: 9/9
  - `stage4`: 15/15
  - `stage5`: 49/49
  - `critical`: 21/21
  - `coverage_ext`: passed

## Репозиторий: что важно читать первым

- API вход: `agent_api.py`
- Мессенджерные endpoint: `messengers_router/endpoint.py`
- Оркестратор: `messengers_router/router.py`
- Политики/слоты: `messengers_router/policies.py`
- Интеграции: `messengers_router/services.py`
- Архитектурный статус: `messengers_router/ARCHITECTURE_STATUS.md`
- Remote eval docs: `messengers_router/eval_suite/README.md`

## Docker / Infra

В `docker-compose.yml` описаны основные сервисы:

- `bookworm-agent`
- `agent-api`
- `nginx`
- `certbot`

Перед запуском docker-контура проверьте mount-paths и внешнюю сеть `local_net` под ваш хост.

## Важно для разработки

- Не правьте только `app_data/prompts/*`, если изменение должно жить в git.
  Канонический набор prompt-файлов для ревью: `messengers_router/prompts/*`.
- Для проверок поведения в мессенджерах используйте `.../api/messenger-generate-once` с `debug=true`.
- Для серверной репрезентативности используйте только remote eval с той машины, где развернут текущий коммит.
