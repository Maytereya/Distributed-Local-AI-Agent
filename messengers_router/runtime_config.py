"""Локальный порт к runtime-конфигу проекта.

Цель: убрать прямые импорты `agent_logic_2.config` из core-модулей
`messengers_router` и централизовать точку привязки к host-конфигу.
"""

from __future__ import annotations

import os

from agent_logic_2 import config as _host_config


def _env_bool_flag(name: str, default: bool = False) -> bool:
    """Читает булев флаг из env в формате `1/true/yes/on`.

    :param name: имя переменной окружения
    :param default: значение по умолчанию
    :return: распарсенное булево значение
    """

    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


class _RuntimeConfigProxy:
    """Прокси поверх host-config.

    Сохраняет текущее поведение чтения/записи атрибутов (включая monkeypatch в тестах),
    но изолирует прямую зависимость от `agent_logic_2` в одном модуле.
    """

    def __getattr__(self, name: str):
        if name == "MR_USE_ORCHESTRATOR":
            return getattr(_host_config, name, _env_bool_flag(name, default=False))
        return getattr(_host_config, name)

    def __setattr__(self, name: str, value):
        setattr(_host_config, name, value)

    def __delattr__(self, name: str):
        delattr(_host_config, name)


config = _RuntimeConfigProxy()
