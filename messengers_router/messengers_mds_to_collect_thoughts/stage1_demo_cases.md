# Stage 1 Demo Cases (20)

Цель: измеримо показать, что бот "живой", понимает запрос и стабильно маршрутизирует интенты.

KPI этапа: `>= 85%` кейсов с корректными `label` и `handoff`.

Проверка:

1. Запустить API:
   - `python -m uvicorn agent_api:app --host 0.0.0.0 --port 8000 --reload`
2. Для каждого кейса отправить:
   - `curl -s http://localhost:8000/api/messenger-generate-once -H "Content-Type: application/json" -d '{"session_id":"s_eval_X","text":"...","debug":true}'`
3. Сверить:
   - `state_update.debug.decision.label`
   - `handoff`

Автопроверка (рекомендуется):

- `python messengers_router/scripts/eval_stage1_cases.py --url http://localhost:8000/api/messenger-generate-once`

## APPOINTMENT (6)

| ID | Вход | Ожидаемый label | Ожидаемый handoff |
|---|---|---|---|
| A1 | Хочу записаться на ЭКГ в Самаре | APPOINTMENT | false |
| A2 | Можно перенести запись на холтер на завтра? | APPOINTMENT | false |
| A3 | Отмените прием у невролога | APPOINTMENT | false |
| A4 | Нужна запись к кардиологу ребенку 4 года | APPOINTMENT | false |
| A5 | Запишите на УЗИ печени на Ленина 5 | APPOINTMENT | false |
| A6 | Запись к Стрежневу на следующей неделе | APPOINTMENT | false |

## PRICE (4)

| ID | Вход | Ожидаемый label | Ожидаемый handoff |
|---|---|---|---|
| P1 | Сколько стоит ЭКГ в Самаре? | PRICE | false |
| P2 | Подскажите цену на ФГДС | PRICE | false |
| P3 | Какая стоимость приема уролога? | PRICE | false |
| P4 | Сколько стоит анализ на витамин Д? | PRICE | false |

## TEST_ASSIST (3)

| ID | Вход | Ожидаемый label | Ожидаемый handoff |
|---|---|---|---|
| T1 | Какие анализы сдать на щитовидку? | TEST_ASSIST | false |
| T2 | Нужен чекап по анемии | TEST_ASSIST | false |
| T3 | Можно сдать ОАК и ферритин завтра? | TEST_ASSIST | false |

## ADDRESS (2)

| ID | Вход | Ожидаемый label | Ожидаемый handoff |
|---|---|---|---|
| D1 | Адрес филиала на Победы 83 и режим работы | ADDRESS | false |
| D2 | Где вы находитесь в Оренбурге? | ADDRESS | false |

## TEST_RESULT fallback (2)

| ID | Вход | Ожидаемый label | Ожидаемый handoff |
|---|---|---|---|
| R1 | Результаты анализов готовы? | TEST_RESULT | true |
| R2 | Пришлите PDF бланк анализов | TEST_RESULT | true |

## DOC_REQUEST -> OTHER + handoff (3)

| ID | Вход | Ожидаемый label | Ожидаемый handoff |
|---|---|---|---|
| C1 | Мне нужна справка для налогового вычета | OTHER | true |
| C2 | Нужна копия договора с печатью | OTHER | true |
| C3 | Как получить копию амбулаторной карты? | OTHER | true |

## Шаблон фиксации результатов

| ID | Факт label | Факт handoff | Pass/Fail |
|---|---|---|---|
| A1 |  |  |  |
| A2 |  |  |  |
| A3 |  |  |  |
| A4 |  |  |  |
| A5 |  |  |  |
| A6 |  |  |  |
| P1 |  |  |  |
| P2 |  |  |  |
| P3 |  |  |  |
| P4 |  |  |  |
| T1 |  |  |  |
| T2 |  |  |  |
| T3 |  |  |  |
| D1 |  |  |  |
| D2 |  |  |  |
| R1 |  |  |  |
| R2 |  |  |  |
| C1 |  |  |  |
| C2 |  |  |  |
| C3 |  |  |  |
