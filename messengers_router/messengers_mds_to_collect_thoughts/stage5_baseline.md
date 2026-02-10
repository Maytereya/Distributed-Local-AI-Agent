# Stage 5 Baseline Log

## 1. Цель

Единый журнал прогонов `eval_stage5_corpus.py`:

- фиксировать метрики качества маршрутизации;
- быстро видеть динамику после изменений;
- иметь прозрачный go/no-go перед деплоем.

## 2. Gate (release candidate)

- `intent_accuracy >= 85%` (минимум), target: `>= 90%`
- `handoff_accuracy >= 85%` (минимум), target: `>= 92%`
- `false_handoff_rate <= 15%` (минимум), target: `<= 8%`
- `slot_fill_rate >= 60%` (минимум), target: `>= 75%`

## 3. Шаблон записи прогона

```
Дата/время:
Commit:
Run ID:
Golden cases (N):
Endpoint URL:

Итоги:
- intent_accuracy:
- handoff_accuracy:
- false_handoff_rate:
- slot_fill_rate:
- unsafe_miss_rate:

By-label accuracy:
- ADDRESS:
- APPOINTMENT:
- NEWS:
- OTHER:
- PRICE:
- TEST_ASSIST:
- TEST_RESULT:

Failure reasons:
- label_mismatch:
- handoff_mismatch:
- ...

Вывод:
- [ ] Gate PASS
- [ ] Gate FAIL

Следующие действия:
1.
2.
3.
```

## 4. Реестр прогонов

| Date | Commit | Run ID | N | Intent | Handoff | False Handoff | Slot Fill | Gate |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 2026-02-09 | local | 1770666293 | 49 | 49.0% | 65.3% | 34.7% | 21.2% | FAIL |
| 2026-02-10 | local | 1770728996 | 49 | 49.0% | 65.3% | 34.7% | 21.2% | FAIL |

## 5. Текущий baseline (снимок)

Источник: запуск  
`venv/bin/python messengers_router/scripts/eval_stage5_corpus.py --url http://localhost:8000/api/messenger-generate-once`

Метрики:

- `run_id: 1770728996`
- `intent_accuracy: 24/49 = 49.0%`
- `handoff_accuracy: 32/49 = 65.3%`
- `false_handoff_rate: 17/49 = 34.7%`
- `slot_fill_rate: 17/80 = 21.2%`
- `unsafe_miss_rate: n/a`

By-label:

- `ADDRESS: 4/4 (100.0%)`
- `APPOINTMENT: 5/12 (41.7%)`
- `NEWS: 0/1 (0.0%)`
- `OTHER: 4/4 (100.0%)`
- `PRICE: 11/12 (91.7%)`
- `TEST_ASSIST: 0/12 (0.0%)`
- `TEST_RESULT: 0/4 (0.0%)`

Failure reasons:

- `label_mismatch: 25`
- `handoff_mismatch: 17`
- `main transport issue in stage1: 2 timeout cases (A1, P1)`

## 6. Фокус на следующий день

1. Снизить `OTHER + low_confidence` для `APPOINTMENT`, `TEST_ASSIST`, `TEST_RESULT`.
2. Убрать ложный `rule_appointment` на price-кейсах (пример `G011`).
3. Уточнить rules для `TEST_ASSIST` и `TEST_RESULT`, чтобы снять `0/12` и `0/4`.
4. Перепроверить gate после каждого изменения на том же golden-наборе.
