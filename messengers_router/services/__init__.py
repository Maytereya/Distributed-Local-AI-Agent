"""Фасад пакета сервисов для поэтапной декомпозиции legacy-модуля.

Пока пакет только реэкспортирует публичные символы из `services_legacy`,
чтобы внешние импорты `messengers_router.services` остались совместимыми.
Новые доменные подмодули будут постепенно наполняться и вытеснять legacy-код.
"""

import sys
import types

from .. import services_legacy as _legacy

_EXPORTED_NAMES = [
    name for name in dir(_legacy)
    if name not in {"__builtins__", "__cached__", "__doc__", "__file__", "__loader__", "__name__", "__package__", "__spec__"}
]
_SYNC_EXCLUDE = {"_legacy", "_EXPORTED_NAMES", "_SYNC_EXCLUDE", "_ServicesFacadeModule"}

__all__ = getattr(
    _legacy,
    "__all__",
    [name for name in _EXPORTED_NAMES if not name.startswith("_")],
)

globals().update({name: getattr(_legacy, name) for name in _EXPORTED_NAMES})


class _ServicesFacadeModule(types.ModuleType):
    """Прокси-модуль, синхронизирующий monkeypatch между facade и legacy.

    :param name: полное имя модуля
    :return: экземпляр прокси-класса модуля
    """

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name not in _SYNC_EXCLUDE:
            setattr(_legacy, name, value)

    def __delattr__(self, name):
        super().__delattr__(name)
        if name not in _SYNC_EXCLUDE and hasattr(_legacy, name):
            delattr(_legacy, name)


sys.modules[__name__].__class__ = _ServicesFacadeModule
