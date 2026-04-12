# Архитектура runtime Free Talk

Статус: draft v0.1  
Дата: 2026-04-12

## 1. Назначение

Этот документ фиксирует текущую runtime-архитектуру FT:

- какие модули реально участвуют в обработке диалога;
- где находятся основные зоны ответственности;
- какие части уже выглядят как устойчивые слои;
- какие части перегружены и требуют разукрупнения.

Документ нужен как опора перед рефакторингом adapter layer и разрезанием перегруженного `agent.py`.

Связанные документы:

- [free_talk_data_registry.md](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/docs/free_talk_data_registry.md)
- [free_talk_capability_matrix.md](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/docs/free_talk_capability_matrix.md)
- [free_talk_translation_rules.md](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/docs/free_talk_translation_rules.md)
- [free_talk_processing_pipeline.md](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/docs/free_talk_processing_pipeline.md)
- [free_talk_refactor_plan.md](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/docs/free_talk_refactor_plan.md)

## 2. Текущий runtime-контур

В упрощённом виде FT сейчас работает так:

```text
UI / API
  -> runner.py
  -> FreeTalkAgent in agent.py
  -> clinical router + heuristic planning
  -> state / memory merge
  -> tool dispatch
  -> legacy services port / web search port
  -> payload verification + rendering
  -> Redis / summary persistence
```

Важно:

- формального отдельного adapter layer в коде сейчас ещё нет;
- request/response translation размазана по `agent.py`;
- `freetalk_services_port.py` является gateway к legacy backend, а не полноценным адаптером.

## 3. Основные слои

### 3.1 Entry / integration layer

Назначение:

- принять сообщение из UI или API;
- создать инфраструктурные зависимости;
- выдать текстовый ответ и `session_id`.

Текущие файлы:

- [runner.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/src/localragagent/freetalk/runner.py)
- [__init__.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/src/localragagent/freetalk/__init__.py)

### 3.2 Orchestration layer

Назначение:

- загрузить контекст и state;
- выбрать route;
- решить, нужен ли tool call;
- обновить память;
- сформировать финальный ответ.

Текущий файл:

- [agent.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/src/localragagent/freetalk/agent.py)

Это главный центр логики FT.

### 3.3 Routing / contract layer

Назначение:

- описать schema router output;
- валидировать `intent`, `entities`, `missing_slots`, `tool_plan`;
- задавать канон routing-решения.

Текущий файл:

- [clinical_router.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/src/localragagent/freetalk/clinical_router.py)

Фактически это не “поиск по клинике”, а routing-contract слой.

### 3.4 Prompting layer

Назначение:

- собирать router prompt;
- собирать post-tool verifier prompt;
- собирать general / tool-result / summary prompts.

Текущий файл:

- [prompts.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/src/localragagent/freetalk/prompts.py)

### 3.5 Tool planning / heuristics layer

Назначение:

- определять medical query;
- делать heuristic fallback tool plan;
- выделять web-search signal;
- выявлять about-agent запросы.

Текущий файл:

- [tool_registry.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/src/localragagent/freetalk/tool_registry.py)

### 3.6 Tool execution layer

Назначение:

- вызвать нужный tool handler;
- завернуть результат в единый `ToolCallResult`;
- детерминированно определить, есть ли в payload полезные данные.

Текущий файл:

- [tool_dispatcher.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/src/localragagent/freetalk/tool_dispatcher.py)

### 3.7 State / contracts layer

Назначение:

- типизировать `DialogAct`, `DialogState`, `AgentReply`, `SessionContext`.

Текущий файл:

- [contracts.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/src/localragagent/freetalk/contracts.py)

### 3.8 Memory layer

Назначение:

- хранить tail истории, summary и meta;
- хранить `clinical_dialog_state` и session entity memory;
- переживать отсутствие Redis через fallback.

Текущие файлы:

- [memory_redis.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/src/localragagent/freetalk/memory_redis.py)
- [memory_persist.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/src/localragagent/freetalk/memory_persist.py)

### 3.9 External boundaries

Назначение:

- быть boundary между FT и внешними зависимостями.

Текущие файлы:

- [freetalk_services_port.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/src/localragagent/ports/freetalk_services_port.py)
- [freetalk_web_search_port.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/src/localragagent/ports/freetalk_web_search_port.py)
- [freetalk_llm_port.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/src/localragagent/ports/freetalk_llm_port.py)

## 4. Карта модулей

| Модуль | Текущая роль | Размер / сложность | Оценка состояния | Комментарий |
|---|---|---|---|---|
| [runner.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/src/localragagent/freetalk/runner.py) | Entrypoint и wiring | Низкая | Устойчив | Можно оставить почти как есть |
| [agent.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/src/localragagent/freetalk/agent.py) | Почти вся оркестрация FT | Очень высокая, ~3000 строк | Перегружен | Главный кандидат на разрезание |
| [clinical_router.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/src/localragagent/freetalk/clinical_router.py) | Router contract / schema normalization | Средняя | Нормален, но название неточно | Логичнее как `routing_contract` |
| [prompts.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/src/localragagent/freetalk/prompts.py) | Prompt builders | Средняя | Частично устарел | Требует согласования с новым каноном данных |
| [tool_registry.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/src/localragagent/freetalk/tool_registry.py) | Heuristic routing signals | Средняя | Нормален, но доменно смешан | Позже лучше разнести planning и signal detection |
| [tool_dispatcher.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/src/localragagent/freetalk/tool_dispatcher.py) | Dispatcher | Низкая | Устойчив | Хороший узкий модуль |
| [contracts.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/src/localragagent/freetalk/contracts.py) | Dataclasses | Низкая | Устойчив | Вероятно потребует расширения, не rewrite |
| [memory_redis.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/src/localragagent/freetalk/memory_redis.py) | Session storage engine | Средняя | Устойчив | Нужна migration semantics, не rewrite engine |
| [memory_persist.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/src/localragagent/freetalk/memory_persist.py) | Summary persistence | Низкая | Устойчив | Не первоочередной участок |
| [config.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/src/localragagent/freetalk/config.py) | Runtime config | Низкая | Устойчив | Менять только при появлении новых настроек |
| [observability.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/src/localragagent/freetalk/observability.py) | Логирование | Низкая | Устойчив | Дополнить event names при рефакторинге |
| [system_prompt.txt](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/src/localragagent/freetalk/system_prompt.txt) | Общий behavioral prompt | Низкая | Умеренно устарел | Не главный источник хаоса |
| [freetalk_services_port.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/src/localragagent/ports/freetalk_services_port.py) | Gateway к legacy services | Низкая | Устойчив | Сохраняется как boundary |

## 5. Что именно перегружает `agent.py`

Сейчас в одном файле одновременно живут:

1. orchestration chat-loop;
2. context guard;
3. dialog state lifecycle;
4. merge старого и нового routing decision;
5. follow-up detection;
6. session memory reuse;
7. entity grounding;
8. partial request translation;
9. post-tool verification;
10. fallback rendering;
11. source tagging;
12. summary compaction orchestration.

Это делает control flow извилистым и затрудняет:

- локальную правку без регрессии;
- изоляцию багов;
- замену legacy alias logic на явный adapter layer;
- чтение и ревью кода.

## 6. Текущие архитектурные проблемы

### 6.1 Нет явного adapter layer

Следствие:

- FT normalized entities не отделены от backend payload;
- legacy-ключи протекают внутрь оркестратора;
- request/response normalization размазана по runtime-коду.

### 6.2 Routing contract и orchestration слишком тесно связаны

Следствие:

- `clinical_router.py` формально валидирует schema;
- но реальные merge policy и stale-state logic живут в `agent.py`;
- semantic contract не выражен одним слоем.

### 6.3 Memory policy смешана с domain logic

Следствие:

- правила “когда помнить врача”;
- “когда подмешивать услугу”;
- “когда сбрасывать старый state”

сейчас не выделены в отдельный policy-модуль.

### 6.4 Rendering смешан с orchestration

Следствие:

- fallback render и source tagging живут там же, где routing и state machine;
- сложнее менять ответный слой независимо от tool orchestration.

### 6.5 Имена некоторых модулей уже неточны

Примеры:

- `clinical_router.py` фактически описывает routing contract, а не собственно “поиск по клинике”;
- `tool_registry.py` уже выполняет не только “реестр tools”, а heuristic planning/signals;
- `agent.py` фактически является orchestrator runtime.

## 7. Что можно считать устойчивым уже сейчас

Следующие решения выглядят правильной основой:

1. отдельный `runner.py` как integration entrypoint;
2. отдельный `ToolDispatcher`;
3. отдельный Redis-backed memory store;
4. отдельные ports к external systems;
5. выделенная prompt-сборка отдельным модулем;
6. dataclass-контракты для dialog state и replies.

То есть проблема не в том, что FT “вообще без структуры”, а в том, что слишком много критической доменной логики не вынесено из `agent.py`.

## 8. Минимальная целевая идея

Целевое направление архитектуры:

- `runner` только собирает зависимости;
- `orchestrator` только управляет стадиями диалога;
- `adapter` только переводит FT data model <-> backend data model;
- `routing contract` только валидирует router output и canonical names;
- `memory policy` только решает reuse/reset behavior;
- `rendering` только отвечает за final answer shaping.

Это и есть путь к более straight-line implementation.

Подробный план перехода описан в:

- [free_talk_refactor_plan.md](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/docs/free_talk_refactor_plan.md)
