# Remote Eval Suite (Messengers Router)

Этот пакет нужен для быстрого и повторяемого прогона регрессии на удаленном сервере.

## Что внутри

- `run_remote_eval.sh` — единый запуск stage-скриптов + critical checks + coverage_ext.
- `eval_critical_cases.py` — строгая проверка критичных сценариев (turn-by-turn).
- `critical_cases.jsonl` — серверный default-набор critical-кейсов для `run_remote_eval.sh`.
- `prepare_wrap_cases.jsonl` — отдельный quality-gate для PREPARE LLM-wrapper (релевантность + компактность).
- `critical_cases_server_parity.jsonl` — канонический локальный parity-набор на 21 кейс, запускается явно через `--cases`.
- `critical_cases_extended.jsonl` — расширенный набор критичных сценариев (P2 coverage, optional).
- `EVAL_EXPANSION_DOD.md` — критерии готовности и команды для расширенного eval-покрытия.
- `logs/` — сюда складываются логи прогонов.

## Быстрый старт

Из корня проекта:

```bash
bash messengers_router/eval_suite/run_remote_eval.sh \
  --url http://172.16.0.16/api/messenger-generate-once \
  --session-prefix s_test
```

Порядок этапов в логе:

- `01_stage1_intent_smoke`
- `02_stage3_appointment_flow`
- `03_stage4_reliability`
- `04_stage5_golden_corpus`
- `05_critical_safety_gate`
- `06_prepare_wrap_quality`
- `07_coverage_ext_assets`

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
  --cases messengers_router/eval_suite/critical_cases_server_parity.jsonl
```

Подменить отдельный набор кейсов для `prepare_wrap` stage:

```bash
bash messengers_router/eval_suite/run_remote_eval.sh \
  --url http://172.16.0.16/api/messenger-generate-once \
  --session-prefix s_test \
  --prepare-wrap-cases messengers_router/eval_suite/prepare_wrap_cases.jsonl
```

Отключить только coverage_ext этап (если нужен временный обход):

```bash
bash messengers_router/eval_suite/run_remote_eval.sh \
  --url http://172.16.0.16/api/messenger-generate-once \
  --session-prefix s_test \
  --skip-coverage-check
```

Запуск расширенного critical-набора:

```bash
python3 messengers_router/eval_suite/eval_critical_cases.py \
  --url http://172.16.0.16/api/messenger-generate-once \
  --cases messengers_router/eval_suite/critical_cases_extended.jsonl \
  --session-prefix s_test_ext \
  --llm-mode hybrid

Локальный pre-push parity прогон:

```bash
bash messengers_router/eval_suite/run_remote_eval.sh \
  --url http://localhost:8000/api/messenger-generate-once \
  --cases messengers_router/eval_suite/critical_cases_server_parity.jsonl
```
```

Запуск расширенного stage5 golden:

```bash
python3 messengers_router/scripts/eval_stage5_corpus.py \
  --url http://172.16.0.16/api/messenger-generate-once \
  --golden messengers_router/messengers_mds_to_collect_thoughts/analysis/golden_versions/stage5_golden_extension_v3.jsonl \
  --session-prefix s_test_stage5_ext
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
- `required_all` — все паттерны должны встретиться в тексте ответа.
- `forbidden_any` — ни один паттерн не должен встретиться в тексте ответа.
- `max_chars` — верхняя граница длины ответа в символах (для контроля избыточных ответов).

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
```bash
bash messengers_router/eval_suite/run_remote_eval.sh \
  --url http://172.16.0.16/api/messenger-generate-once \
  --session-prefix s_test \
  --cases messengers_router/eval_suite/critical_cases.jsonl
```
### Где оценить логи после запуска: 
```bash
messengers_router/eval_suite/logs/<timestamp>/
```
### Подсказка по параметрам:
```bash
bash messengers_router/eval_suite/run_remote_eval.sh --help
```

### run_remote_eval.sh берет тесты из локального checkout на той машине, где запущен.

Что именно:

• stage1/3/4 — кейсы зашиты прямо в скриптах: eval_stage1_cases.py, eval_stage3_appointment_flow.py, eval_stage4_reliability.py.

• critical — по умолчанию используется critical_cases.jsonl, для локального parity можно явно передать `--cases messengers_router/eval_suite/critical_cases_server_parity.jsonl`.

• stage5 — по умолчанию stage5_golden_cases.jsonl, либо версия через --golden-version vN из analysis/golden_versions.

• coverage_ext — проверка полноты extension-наборов (`check_eval_coverage.py`), запускается автоматически после `critical`.

То есть “последняя версия тестов” = последний код/файлы в текущем коммите на сервере.
