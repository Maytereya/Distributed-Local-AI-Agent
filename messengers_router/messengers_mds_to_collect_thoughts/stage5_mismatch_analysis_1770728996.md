# Stage 5 mismatch analysis (run 1770728996)

Дата: 2026-02-10  
Источник: `eval_stage5_corpus.py`  
Итог: `intent 49.0%`, `handoff 65.3%`, `false_handoff 34.7%`, `slot_fill 21.2%`

## 1. Сводка проблем

1. `APPOINTMENT -> OTHER` на части кейсов (`G014..G017`, `G020..G022`)  
Причина: `low_confidence + handoff_recommended`, правило записи не срабатывает на слабо структурированных фразах.

2. `TEST_ASSIST` почти полностью распадается (`G025..G036`)  
Причины:
- часть фраз уходит в `OTHER` из-за `low_confidence`;
- часть ошибочно ловится как `PRICE` (`G031`, `G033`) и `ADDRESS` (`G034`);
- один кейс ошибочно попал в `TEST_RESULT` из-за email/готовности (`G027`).

3. `TEST_RESULT -> OTHER` (`G037..G040`)  
Причина: правила результата слишком узкие для части формулировок из корпуса.

4. `PRICE -> APPOINTMENT` (`G011`)  
Причина: пересечение формулировок про прием и запись, appointment-rule перехватывает ценовой кейс.

5. `NEWS -> OTHER` (`G045`)  
Причина: нет устойчивого rules-покрытия новостей/акций, LLM fallback уходит в low confidence.

6. `handoff_mismatch` (17 кейсов)  
Причина: у non-critical интентов при `low_confidence` ставится `handoff=true` слишком рано.

7. Отдельно по Stage 1 (`run eval_stage1_cases`): `A1`, `P1`  
Причина: `TimeoutError` (transport), не логическая ошибка роутера.

## 2. Что считаем known limitations перед пушем

1. Корпусный Stage 5 пока не проходит gate, это ожидаемое ограничение релиза.  
2. Основной MVP flow записи стабилен (`stage3: 10/10`).  
3. Базовая маршрутизация Stage 1 держит KPI (`stage1: 18/20`, 2 падения из-за timeout).

## 3. Приоритет доработок (следующий цикл)

1. Расширить rule-покрытие `TEST_ASSIST` и добавить negative-guard от `PRICE/ADDRESS` перехвата.  
2. Расширить rule-покрытие `TEST_RESULT` (реальные формулировки “готовность/не пришли/на почту/смс”).  
3. Ослабить авто-handoff на `low_confidence` для non-critical (`PRICE/TEST_ASSIST/ADDRESS/APPOINTMENT`) в пользу уточняющего вопроса.  
4. Добавить rules для `NEWS` (скидки, акции, промокоды).  
5. Для `PRICE` усилить guard: “стоимость приема” не переводить в `APPOINTMENT`, если нет явного действия записи.
