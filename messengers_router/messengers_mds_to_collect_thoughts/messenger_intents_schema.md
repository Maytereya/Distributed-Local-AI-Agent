# Таблица интентов: слоты и уточняющие вопросы

| Интент | Обязательные слоты | Уточняющий вопрос |
|---|---|---|
| LAB_TESTS_INFO | service, city | Подскажите, какой именно анализ и в каком городе сдавать? |
| APPOINTMENT_BOOK | doctor_or_service, city, date, time | К какому врачу/услуге хотите записаться и в каком городе? Есть желаемая дата/время? |
| DIAGNOSTICS_INFO | service, city | Уточните вид исследования и город, пожалуйста. |
| ADDRESS_HOURS | city, branch | В каком городе/филиале нужен адрес и режим работы? |
| PRICE_QUERY | service, city | Уточните услугу/анализ и город — посчитаю стоимость. |
| DOCTOR_INFO | doctor_or_specialty, city | Уточните специальность или фамилию врача и город. |
| RESULTS_DOCS | patient_name, date, channel | Уточните дату сдачи и как вам удобнее получить результат (почта/лично). |
| APPOINTMENT_MOVE_CANCEL | doctor_or_service, date, time, patient_name | Кого и на какую дату/время нужно перенести/отменить? Назовите ФИО. |
| CALLBACK_OPERATOR | patient_name, phone | Как можно к вам обратиться и по какому телефону передать заявку? |
| DISCOUNT_PROMO | service, city, period | На какую услугу и в каком городе интересует акция? |
| INSURANCE_DMS_OMS | insurer, city | Уточните страховую компанию и город, пожалуйста. |