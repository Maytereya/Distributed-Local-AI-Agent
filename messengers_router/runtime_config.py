"""Локальный порт к runtime-конфигу проекта.

Цель: убрать прямые импорты `agent_logic_2.config` из core-модулей
`messengers_router` и централизовать точку привязки к host-конфигу.
"""

from __future__ import annotations

from agent_logic_2 import config as _host_config


class _RuntimeConfigProxy:
    """Прокси поверх host-config.

    Сохраняет текущее поведение чтения/записи атрибутов (включая monkeypatch в тестах),
    но изолирует прямую зависимость от `agent_logic_2` в одном модуле.
    """

    def __getattr__(self, name: str):
        return getattr(_host_config, name)

    def __setattr__(self, name: str, value):
        setattr(_host_config, name, value)


config = _RuntimeConfigProxy()

