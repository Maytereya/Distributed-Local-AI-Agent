# Сценарная развилка интентов (по чатам WhatsApp)

Анализ папки: `agent_logic_2/чаты из ватсап для ии/*.txt`

- Файлов: **259**
- Диалогов (корректно прочитано): **252**
- Сообщений всего: **3900**
- Сообщений пациентов: **1920**
- Сообщений операторов: **1980**

## Частотность интентов пациентов

| Интент | Кол-во | Доля |
|---|---:|---:|
| LAB_TESTS_INFO | 255 | 13.3% |
| APPOINTMENT_BOOK | 199 | 10.4% |
| DIAGNOSTICS_INFO | 88 | 4.6% |
| ADDRESS_HOURS | 37 | 1.9% |
| PRICE_QUERY | 54 | 2.8% |
| DOCTOR_INFO | 36 | 1.9% |
| RESULTS_DOCS | 24 | 1.2% |
| APPOINTMENT_MOVE_CANCEL | 9 | 0.5% |
| CALLBACK_OPERATOR | 16 | 0.8% |
| DISCOUNT_PROMO | 12 | 0.6% |
| INSURANCE_DMS_OMS | 5 | 0.3% |

## Сценарная развилка (дерево решений)

```
Входящее сообщение
│
├─ НЕ пациент? → Staff/Internal
│
└─ Пациент
   │
   ├─ URGENT/COMPLAINT/MEDICAL_ADVICE? → handoff
   │
   ├─ APPOINTMENT_BOOK → уточнить город, врач/услуга, дата/время
   ├─ LAB_TESTS_INFO → уточнить анализ и город
   ├─ DIAGNOSTICS_INFO → уточнить исследование и город
   ├─ ADDRESS_HOURS → уточнить город/филиал
   ├─ PRICE_QUERY → уточнить услугу и город
   ├─ DOCTOR_INFO → уточнить специальность/ФИО и город
   ├─ RESULTS_DOCS → уточнить дату и канал получения
   ├─ APPOINTMENT_MOVE_CANCEL → уточнить дату/врача/ФИО
   ├─ DISCOUNT_PROMO → уточнить услугу и город
   ├─ INSURANCE_DMS_OMS → уточнить страховую и город
   └─ CALLBACK_OPERATOR → запрос контактов
```