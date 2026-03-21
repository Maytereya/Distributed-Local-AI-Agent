## Eval Expansion DoD (P2 scope)

Цель: зафиксировать минимально достаточное покрытие для расширенного eval-контурa по направлениям:
- `DOCTOR_INFO`
- `DOCTOR_SCHEDULE`
- `PREPARE`
- `OTHER`
- non-Samara
- follow-up turns

Важно: это отдельный расширенный набор, который **не подменяет** текущий default gate (`run_remote_eval.sh` без дополнительных флагов).

### Артефакты

- Stage5 extension golden:
  - `messengers_router/messengers_mds_to_collect_thoughts/analysis/golden_versions/stage5_golden_extension_v3.jsonl`
- Critical extension (turn-by-turn):
  - `messengers_router/eval_suite/critical_cases_extended.jsonl`
- Coverage-check script:
  - `messengers_router/scripts/check_eval_coverage.py`

### Критерии DoD

1. В extension-наборах присутствуют и валидируются кейсы по `DOCTOR_INFO`, `DOCTOR_SCHEDULE`, `PREPARE`, `OTHER`.
2. Есть non-Samara покрытие (минимум 3 кейса/turn-а).
3. Есть follow-up multi-turn покрытие (минимум 4 сценария).
4. Проверка покрытия проходит:
   - `python3 messengers_router/scripts/check_eval_coverage.py`

### Как запускать расширенный eval

1. Stage5 extension:

```bash
python3 messengers_router/scripts/eval_stage5_corpus.py \
  --url http://<endpoint>/api/messenger-generate-once \
  --golden messengers_router/messengers_mds_to_collect_thoughts/analysis/golden_versions/stage5_golden_extension_v3.jsonl \
  --session-prefix s_eval_stage5_ext
```

2. Critical extension:

```bash
python3 messengers_router/eval_suite/eval_critical_cases.py \
  --url http://<endpoint>/api/messenger-generate-once \
  --cases messengers_router/eval_suite/critical_cases_extended.jsonl \
  --session-prefix s_eval_critical_ext \
  --llm-mode hybrid
```

3. Быстрая проверка структуры покрытия:

```bash
python3 messengers_router/scripts/check_eval_coverage.py
```

### Статус включения в gate

- На текущем этапе расширенный набор подготовлен и хранится отдельно.
- В default `run_remote_eval.sh` подключается только после отдельного решения команды (чтобы не рисковать стабильностью текущего production gate во время API-неопределенности).
