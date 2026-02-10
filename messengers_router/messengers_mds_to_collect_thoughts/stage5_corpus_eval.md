# Stage 5 Corpus Eval

Цель: регулярная регрессия на корпусе реальных формулировок пациентов из чатов.

## 1) Построение golden-набора

Источник:  
`/Users/maxten/Dev/Distributed-Local-AI-Agent2/messengers_router/messengers_mds_to_collect_thoughts/analysis/_tmp_intent_samples.json`

Генератор:

```bash
venv/bin/python /Users/maxten/Dev/Distributed-Local-AI-Agent2/messengers_router/scripts/build_stage5_golden_cases.py \
  --source /Users/maxten/Dev/Distributed-Local-AI-Agent2/messengers_router/messengers_mds_to_collect_thoughts/analysis/_tmp_intent_samples.json \
  --out /Users/maxten/Dev/Distributed-Local-AI-Agent2/messengers_router/messengers_mds_to_collect_thoughts/analysis/stage5_golden_cases.jsonl \
  --per-intent 12
```

Примечания:

- Канонический маппинг:
  - `RESULTS -> TEST_RESULT`
  - `DISCOUNT -> NEWS`
  - `CERTIFICATE -> OTHER`
- Для `TEST_RESULT` и `OTHER(=doc request)` в golden по умолчанию `expected_handoff=true`.
- Из выборки автоматически исключаются пустые/тривиальные приветствия.

## 2) Прогон eval

1. Запустить API:

```bash
python -m uvicorn agent_api:app --host 0.0.0.0 --port 8000 --reload
```

2. Запустить stage-5 eval:

```bash
venv/bin/python /Users/maxten/Dev/Distributed-Local-AI-Agent2/messengers_router/scripts/eval_stage5_corpus.py \
  --url http://localhost:8000/api/messenger-generate-once \
  --golden /Users/maxten/Dev/Distributed-Local-AI-Agent2/messengers_router/messengers_mds_to_collect_thoughts/analysis/stage5_golden_cases.jsonl
```

## 3) Метрики

Скрипт считает:

- `intent_accuracy`
- `handoff_accuracy`
- `false_handoff_rate`
- `slot_fill_rate` (proxy по `last_entities` vs required slots)
- `unsafe_miss_rate` (для `URGENT/COMPLAINT/MEDICAL_ADVICE`, если такие кейсы есть в golden)

## 4) Gate для релиза

Базовый gate (внутри скрипта):

- `intent_accuracy >= 85%`
- `handoff_accuracy >= 85%`
- `transport/json errors == 0`

Рекомендуемый target (после стабилизации):

- `intent_accuracy >= 90%`
- `handoff_accuracy >= 92%`
- `false_handoff_rate <= 8%`

