# Remote Eval Suite (Messengers Router)

Этот пакет нужен для быстрого и повторяемого прогона регрессии на удаленном сервере.

## Что внутри

- `run_remote_eval.sh` — единый запуск stage-скриптов + critical checks.
- `eval_critical_cases.py` — строгая проверка критичных сценариев (turn-by-turn).
- `critical_cases.jsonl` — список критичных кейсов, которые нельзя ломать.
- `logs/` — сюда складываются логи прогонов.

## Быстрый старт

Из корня проекта:

```bash
bash messengers_router/eval_suite/run_remote_eval.sh \
  --url http://172.16.0.16/api/messenger-generate-once \
  --session-prefix s_test
```

Опционально указать golden-версию:

```bash
bash messengers_router/eval_suite/run_remote_eval.sh \
  --url http://172.16.0.16/api/messenger-generate-once \
  --session-prefix s_test \
  --golden-version v2
```

Подменить набор критичных кейсов:

```bash
bash messengers_router/eval_suite/run_remote_eval.sh \
  --url http://172.16.0.16/api/messenger-generate-once \
  --session-prefix s_test \
  --cases messengers_router/eval_suite/critical_cases.jsonl
```

## Как редактировать критичные проверки

Файл: `critical_cases.jsonl` (один JSON-объект на строку).

Поддерживаются 2 формата:

1) Одноходовый кейс:

```json
{"case_id":"C1","text":"...","expected_label":"PREPARE","expected_handoff":false,"required_any":["подготов"],"forbidden_any":["из какого города"]}
```

2) Многоходовый кейс:

```json
{
  "case_id":"C2",
  "turns":[
    {"text":"...","expected_label":"PREPARE","forbidden_any":["доступны филиалы"]},
    {"text":"...","expected_label":"PREPARE","forbidden_any":["из какого города"]}
  ]
}
```

### Поля проверки

- `expected_label` — ожидаемый интент (из debug decision.label).
- `expected_handoff` — ожидаемый handoff.
- `required_any` — хотя бы один паттерн должен встретиться в тексте ответа.
- `forbidden_any` — ни один паттерн не должен встретиться в тексте ответа.

Паттерны проверяются как подстроки, регистронезависимо.

## Важно

Для label-проверок нужен endpoint `.../api/messenger-generate-once` (с `debug=true`).
Если использовать streaming endpoint `.../api/messenger-generate`, label-check не будет полным.

Падение stage в отчете означает не "сервер недоступен", а "quality gate не пройден" (ответы не соответствуют ожидаемым правилам/метрикам).

## Человеческий коммент по запуску:

### В терминале (прямо в IDE), в корне проекта запуск:

bash messengers_router/eval_suite/run_remote_eval.sh \
  --url http://172.16.0.16/api/messenger-generate-once \
  --session-prefix s_test \
  --golden-version v2

### Если кастомный (или свой) набор критичных кейсов:

bash messengers_router/eval_suite/run_remote_eval.sh \
  --url http://172.16.0.16/api/messenger-generate-once \
  --session-prefix s_test \
  --cases messengers_router/eval_suite/critical_cases.jsonl

### Где оценить логи после запуска: 
messengers_router/eval_suite/logs/<timestamp>/

### Подсказка по параметрам:
bash messengers_router/eval_suite/run_remote_eval.sh --help
