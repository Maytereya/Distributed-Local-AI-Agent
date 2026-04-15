# Архитектура runtime Free Talk

Статус: v0.2 (синхронизировано с кодом)  
Дата: 2026-04-15

## 1. Назначение

Этот документ фиксирует текущую runtime-архитектуру FT:

- какие модули реально участвуют в обработке диалога;
- где находятся основные зоны ответственности;
- какие части уже выглядят как устойчивые слои;
- какие части перегружены и требуют разукрупнения.

Документ фиксирует уже сложившуюся структуру runtime после основного рефакторинга FT.

Связанные документы:

- [free_talk_data_registry.md](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/docs/free_talk_data_registry.md)
- [free_talk_capability_matrix.md](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/docs/free_talk_capability_matrix.md)
- [free_talk_translation_rules.md](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/docs/free_talk_translation_rules.md)
- [free_talk_processing_pipeline.md](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/docs/free_talk_processing_pipeline.md)
- [free_talk_refactor_plan.md](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/docs/free_talk_refactor_plan.md)
- [free_talk_feature_integration_playbook.md](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/docs/free_talk_feature_integration_playbook.md)

## 2. Текущий runtime-контур

В упрощённом виде FT сейчас работает так:

```text
UI / API
  -> runner.py
  -> FreeTalkAgent facade in agent.py
  -> orchestrator.py
  -> routing_contract.py + routing_prompting.py + tool_planning.py
  -> state / follow-up / memory / grounding / medical policies
  -> adapter.py
  -> tool_dispatcher.py
  -> legacy services port / web search port
  -> payload normalization + rendering
  -> Redis / summary persistence
```

Важно:

- отдельный `adapter layer` уже реализован в `adapter.py`;
- `agent.py` больше не является центром всей доменной логики, а работает как публичный фасад;
- source-of-truth для routing/prompting/planning уже живёт в целевых модулях;
- `freetalk_services_port.py` остаётся gateway к legacy backend.

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

Текущие файлы:

- [agent.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/src/localragagent/freetalk/agent.py)
- [orchestrator.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/src/localragagent/freetalk/orchestrator.py)

Роли:

- `agent.py` — публичный фасад и совместимая оболочка `FreeTalkAgent`;
- `orchestrator.py` — основной chat/runtime flow.

### 3.3 Routing / contract layer

Назначение:

- описать schema router output;
- валидировать `intent`, `entities`, `missing_slots`, `tool_plan`;
- задавать канон routing-решения.

Текущий файл:

- [routing_contract.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/src/localragagent/freetalk/routing_contract.py)

Этот модуль является source-of-truth для router schema и канона `missing_slots/entities`.

### 3.4 Prompting layer

Назначение:

- собирать router prompt;
- собирать post-tool verifier prompt;
- собирать general / tool-result / summary prompts.

Текущий файл:

- [routing_prompting.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/src/localragagent/freetalk/routing_prompting.py)

### 3.5 Tool planning / heuristics layer

Назначение:

- определять medical query;
- делать heuristic fallback tool plan;
- выделять web-search signal;
- выявлять about-agent запросы.

Текущий файл:

- [tool_planning.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/src/localragagent/freetalk/tool_planning.py)

### 3.6 Policy and state support layer

Назначение:

- canonical dialog state lifecycle;
- follow-up / topic shift;
- session memory reuse;
- intent / candidate / catalog / grounding policy;
- medical pre-tool и post-tool orchestration support.

Текущие файлы:

- [dialog_state.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/src/localragagent/freetalk/dialog_state.py)
- [followup_policy.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/src/localragagent/freetalk/followup_policy.py)
- [memory_policy.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/src/localragagent/freetalk/memory_policy.py)
- [intent_policy.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/src/localragagent/freetalk/intent_policy.py)
- [candidate_policy.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/src/localragagent/freetalk/candidate_policy.py)
- [catalog_policy.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/src/localragagent/freetalk/catalog_policy.py)
- [routing_policy.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/src/localragagent/freetalk/routing_policy.py)
- [grounding_policy.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/src/localragagent/freetalk/grounding_policy.py)
- [medical_pretool_policy.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/src/localragagent/freetalk/medical_pretool_policy.py)
- [post_tool_policy.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/src/localragagent/freetalk/post_tool_policy.py)
- [medical_toolloop.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/src/localragagent/freetalk/medical_toolloop.py)

### 3.7 Tool execution layer

Назначение:

- вызвать нужный tool handler;
- завернуть результат в единый `ToolCallResult`;
- детерминированно определить, есть ли в payload полезные данные.

Текущий файл:

- [tool_dispatcher.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/src/localragagent/freetalk/tool_dispatcher.py)

### 3.8 State / contracts layer

Назначение:

- типизировать `DialogAct`, `DialogState`, `AgentReply`, `SessionContext`.

Текущий файл:

- [contracts.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/src/localragagent/freetalk/contracts.py)

### 3.9 Memory layer

Назначение:

- хранить tail истории, summary и meta;
- хранить `clinical_dialog_state` и session entity memory;
- переживать отсутствие Redis через fallback.

Текущие файлы:

- [memory_redis.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/src/localragagent/freetalk/memory_redis.py)
- [memory_persist.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/src/localragagent/freetalk/memory_persist.py)

### 3.10 Adapter and external boundaries

Назначение:

- быть boundary между FT и внешними зависимостями.

Текущие файлы:

- [adapter.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/src/localragagent/freetalk/adapter.py)
- [adapter_contracts.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/src/localragagent/freetalk/adapter_contracts.py)
- [freetalk_services_port.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/src/localragagent/ports/freetalk_services_port.py)
- [freetalk_web_search_port.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/src/localragagent/ports/freetalk_web_search_port.py)
- [freetalk_llm_port.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/src/localragagent/ports/freetalk_llm_port.py)

## 4. Карта модулей

| Модуль | Текущая роль | Размер / сложность | Оценка состояния | Комментарий |
|---|---|---|---|---|
| [runner.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/src/localragagent/freetalk/runner.py) | Entrypoint и wiring | Низкая | Устойчив | Тонкий слой интеграции |
| [agent.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/src/localragagent/freetalk/agent.py) | Публичный фасад `FreeTalkAgent` | Средняя | Стабилизирован | Держит совместимый API и thin wrappers |
| [orchestrator.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/src/localragagent/freetalk/orchestrator.py) | Основной runtime flow | Средняя | Ключевой core | Здесь живёт верхний chat-loop |
| [routing_contract.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/src/localragagent/freetalk/routing_contract.py) | Router contract / schema normalization | Средняя | Устойчив | Source-of-truth для routing schema |
| [routing_prompting.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/src/localragagent/freetalk/routing_prompting.py) | Prompt builders | Средняя | Устойчив | Согласован с текущим каноном |
| [tool_planning.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/src/localragagent/freetalk/tool_planning.py) | Heuristic signals и fallback tool planning | Средняя | Устойчив | Разделён с routing contract |
| [adapter.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/src/localragagent/freetalk/adapter.py) | FT -> backend translation и нормализация ответа | Средняя | Ключевой core | Legacy naming остаётся только здесь |
| [tool_dispatcher.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/src/localragagent/freetalk/tool_dispatcher.py) | Dispatcher | Низкая | Устойчив | Хороший узкий модуль |
| [dialog_state.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/src/localragagent/freetalk/dialog_state.py) | Dialog state lifecycle | Средняя | Устойчив | Канон session/dialog state |
| [followup_policy.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/src/localragagent/freetalk/followup_policy.py) | Follow-up / topic shift policy | Средняя | Устойчив | Детерминированная контекстная логика |
| [memory_policy.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/src/localragagent/freetalk/memory_policy.py) | Session entity memory rules | Средняя | Устойчив | Переиспользование сущностей между ходами |
| [contracts.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/src/localragagent/freetalk/contracts.py) | Dataclasses | Низкая | Устойчив | Вероятно потребует расширения, не rewrite |
| [memory_redis.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/src/localragagent/freetalk/memory_redis.py) | Session storage engine | Средняя | Устойчив | Нужна migration semantics, не rewrite engine |
| [memory_persist.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/src/localragagent/freetalk/memory_persist.py) | Summary persistence | Низкая | Устойчив | Не первоочередной участок |
| [config.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/src/localragagent/freetalk/config.py) | Runtime config | Низкая | Устойчив | Менять только при появлении новых настроек |
| [observability.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/src/localragagent/freetalk/observability.py) | Логирование | Низкая | Устойчив | Дополнить event names при рефакторинге |
| [system_prompt.txt](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/src/localragagent/freetalk/system_prompt.txt) | Общий behavioral prompt | Низкая | Умеренно устарел | Не главный источник хаоса |
| [freetalk_services_port.py](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/src/localragagent/ports/freetalk_services_port.py) | Gateway к legacy services | Низкая | Устойчив | Сохраняется как boundary |

## 5. Что ещё остаётся в `agent.py`

После разукрупнения `agent.py` больше не держит всю критическую логику FT, но в нём ещё остаются:

1. публичный класс `FreeTalkAgent`;
2. совместимые wrapper-методы поверх `orchestrator.py` и policy-модулей;
3. wiring зависимостей runtime;
4. точка входа для тестовых monkeypatch и локального debug.

Текущая проблема уже не в том, что `agent.py` архитектурно хаотичен, а в том, что он остаётся наиболее заметной публичной точкой входа и требует дисциплины: туда не должна возвращаться новая доменная логика.

## 6. Текущие архитектурные узкие места

### 6.1 Runtime разбит на много policy-модулей

Это уже лучше, чем монолитный `agent.py`, но теперь важен контроль связности:

- новый функционал нельзя добавлять напрямую в фасад;
- новые policy должны подключаться через понятный orchestration flow;
- документы должны оставаться синхронны с этой картой модулей.

### 6.2 Legacy backend всё ещё задаёт часть naming constraints

Следствие:

- `adapter.py` остаётся обязательным boundary;
- legacy поля вроде `filial`, `number`, `branch` должны быть вытеснены из FT-слоя и жить только в adapter/backend boundary.

### 6.3 Основной риск сместился в качество state/merge поведения

Критичные сценарии теперь проверяются не архитектурой модулей, а качеством:

- topic shift;
- follow-up reuse;
- clarify loops;
- self-error correction;
- unsupported city / partial data cases.

## 7. Что можно считать устойчивым уже сейчас

Следующие решения выглядят правильной основой:

1. отдельный `runner.py` как integration entrypoint;
2. отдельный `orchestrator.py` как runtime core;
3. отдельный `adapter.py` как boundary к legacy backend;
4. отдельные routing/prompting/planning модули;
5. отдельные policy-модули для state, memory, grounding и medical flow;
6. отдельный Redis-backed memory store и ports к external systems.

То есть FT уже имеет рабочую многослойную структуру. Основная задача теперь не разукрупнение как таковое, а поддержание чистых границ между слоями.

## 8. Минимальная целевая идея

Текущее целевое направление архитектуры:

- `runner` только собирает зависимости;
- `agent.py` только предоставляет публичный фасад;
- `orchestrator` управляет стадиями диалога;
- `adapter` переводит FT data model <-> backend data model;
- `routing contract` валидирует router output и canonical names;
- `memory/followup/policy` решают reuse/reset/grounding behavior;
- `rendering` отвечает за final answer shaping.

Это и есть путь к более straight-line implementation.

Подробный план перехода описан в:

- [free_talk_refactor_plan.md](/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/docs/free_talk_refactor_plan.md)
