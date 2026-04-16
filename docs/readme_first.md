# Docs: Readme First

Макс, этот файл - входная точка в раздел `docs/`.

Ниже кратко описано назначение каждого документа и порядок чтения.

## Рекомендуемый порядок чтения

1. `free_talk_agent_tz.md`
2. `free_talk_runtime_architecture.md`
3. `free_talk_data_registry.md`
4. `free_talk_processing_pipeline.md`
5. `free_talk_translation_rules.md`
6. `free_talk_capability_matrix.md`
7. `free_talk_feature_integration_playbook.md`
8. `free_talk_refactor_plan.md`
9. `localragagent_architecture.md`
10. `nayka_site_api_live_audit_latest.md`

## FreeTalk docs

### `free_talk_agent_tz.md`
- Исходное ТЗ на FT.
- Описывает продуктовую цель режима, ожидаемое поведение, ограничения и требования к качеству ответов.
- Это основной документ, если нужно понять, что FT вообще должен уметь.

### `free_talk_runtime_architecture.md`
- Карта текущей архитектуры FT.
- Объясняет, из каких модулей состоит FT runtime и за что отвечает каждый слой.
- Нужен, когда разработчик пытается понять структуру кода до чтения самих модулей.

### `free_talk_data_registry.md`
- Реестр основных типов данных FT.
- Фиксирует `missing slots`, нормализованные сущности, флаги и смысл полей.
- Это основной источник истины по названиям и семантике данных внутри FT.

### `free_talk_processing_pipeline.md`
- Описание потока обработки запроса в обе стороны.
- Показывает, что делает код детерминированно, а что делегируется LLM.
- Здесь же зафиксирован общий `handoff finalization` path: полный reset FT-session и выдача нового `session_id`.
- Нужен, когда нужно понять, где должна жить логика: в коде, в prompt или в adapter.

### `free_talk_translation_rules.md`
- Таблица перевода данных между FT, adapter и backend.
- Описывает mapping (или: картирование, направления связей - синонимы слова mapping в данном случае) по доменам: врачи, расписание, услуги, адреса, результаты анализов и т.д.
- Нужен при изменении adapter layer и контрактов tools.

### `free_talk_capability_matrix.md`
- Матрица возможностей FT.
- Показывает, какие сценарии поддержаны, какие источники используются и где есть ограничения.
- Нужна, когда надо понять, что система реально умеет, а что пока только планируется.

### `free_talk_feature_integration_playbook.md`
- Правила внедрения нового функционала в FT. Важнейший документ. Показывай его агенту, когда планируешь добавить новый функционал.
- Объясняет, как классифицировать новую фичу, куда её встраивать и какие документы/тесты обновлять.
- В том числе фиксирует обязательное правило: любой `handoff` в FT должен проходить через общий session-reset path.
- Это operational guide для разработчиков.

### `free_talk_refactor_plan.md`
- План структурного рефакторинга FT.
- Нужен для понимания, как проект переходил от старой структуры к новой и какие этапы были запланированы.
- Сейчас это документ о траектории изменений, а не о runtime-контракте. То есть исторический.

## Общие документы по пакету и backend

### `localragagent_architecture.md`
- Общая архитектура package `localragagent`.
- Определяет package boundary, dependency rules и место FT внутри общей структуры проекта.
- Нужен, если разработчик работает не только с FT, но и с package layout в целом.

### `nayka_site_api_live_audit_latest.md`
- Результаты живого аудита Nayka site API.
- Это справочный документ по фактическому поведению backend endpoint’ов и их данным.
- Нужен при проверке ограничений backend и при адаптации FT/adapter к реальным данным API.
- Может устаревать при выходе обновлений в нашем *пока фронтальном проекте messengers_router. 

## Как пользоваться этой папкой на практике

- Если нужно понять продуктовую цель FT: читать `free_talk_agent_tz.md`.
- Если нужно понять кодовую структуру FT: читать `free_talk_runtime_architecture.md`.
- Если нужно понять какие данные и слоты под данные используются: читать `free_talk_data_registry.md`.
- Если нужно менять adapter к messengers_router или tool contracts: читать `free_talk_translation_rules.md`.
- Если нужно добавлять любую новую фичу: читать `free_talk_feature_integration_playbook.md`.
- Если нужно свериться с реальными ограничениями backend: читать `nayka_site_api_live_audit_latest.md`.

## Что считать главным источником истины

- По целям и поведению FT: `free_talk_agent_tz.md`
- По внутренним данным FT: `free_talk_data_registry.md`
- По pipeline обработки запроса пользователя: `free_talk_processing_pipeline.md`
- По mapping FT -> adapter -> backend: `free_talk_translation_rules.md`
- По структуре runtime-кода: `free_talk_runtime_architecture.md`


Пока все.
