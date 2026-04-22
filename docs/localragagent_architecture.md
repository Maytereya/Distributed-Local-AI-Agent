# localragagent: package architecture

Дата: 2026-04-08

## Цель

Ввести единый package boundary `localragagent` (`src`-layout), сохранив доступ к существующему коду проекта через adapter/port слой и добавив автоматические архитектурные проверки в CI.

## Слои

- `interfaces/` — публичные точки входа (HTTP/UI adapters).
- `orchestration/` — координация сценариев без прямой работы с host-кодом.
- `freetalk/` — автономный orchestration-модуль свободного общения (tool loop, память, prompting).
- `domain/` — чистые модели/правила.
- `infrastructure/` — инфраструктурные helper-модули.
- `ports/` — единственный слой, где разрешен доступ к legacy host-модулям.

## Правила зависимостей

- `interfaces` -> `orchestration|domain|infrastructure|port`
- `orchestration` -> `domain|infrastructure|port`
- `infrastructure` -> `domain|port`
- `domain` -> (ничего)
- `port` -> (ничего внутри package; bridge к host)

Отдельное правило:
- прямые импорты host-модулей (`agent_logic_*`, `messengers_router`, `agent_api`, и т.д.) разрешены только в `ports/*`.

## Проверка

Локально:

```bash
python3 messengers_router/scripts/check_architecture_imports.py
python3 tools/check_localragagent_architecture.py
```

В GitHub:

- workflow `.github/workflows/messengers_router_arch_guardrails.yml` запускает обе проверки.
