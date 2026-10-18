"""Скрипты, инструменты и стенд импортируют из кода бота только то, что в нём есть (18.10).

Класс: чистка мёртвого кода ядра (T7: снята `_build_multi_price_payload`, 5e78193) молча ломала инструменты вне гейта —
`scripts/audit_cart_misses.py` падал на импорте при запуске, а гейт это не видел: скрипты тестами не импортируются.

Инвариант: каждое `from messengers_router… / agent_logic_2… / evalbench… import X` в `scripts/`, `tools/`, `evalbench/`,
`messengers_router/scripts/` указывает на существующий модуль и имя. Скрипты не исполняются — только разбираются
(`ast`), импортируются модули бота, на которые они ссылаются.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_FOLDERS = ("scripts", "tools", "evalbench", "messengers_router/scripts")
_OURS = ("messengers_router", "agent_logic_2", "evalbench")


def _imports() -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    for folder in _FOLDERS:
        for path in sorted((_ROOT / folder).rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module and node.module.split(".")[0] in _OURS:
                    out.extend((str(path.relative_to(_ROOT)), node.module, alias.name) for alias in node.names if alias.name != "*")
    return out


def test_there_is_something_to_check():
    assert len(_imports()) > 50


@pytest.mark.parametrize(("path", "module", "name"), _imports(), ids=lambda v: str(v))
def test_imported_name_exists(path, module, name):
    mod = importlib.import_module(module)
    if hasattr(mod, name):
        return
    importlib.import_module(f"{module}.{name}")  # подмодуль пакета; нет и его — ImportError с именем скрипта в id
