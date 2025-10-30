import sys
import sysconfig
import importlib

# Этот пакет имеет то же имя, что и внешняя библиотека `gigachat` (SDK),
# поэтому Python по умолчанию будет импортировать локальный пакет.
# Шина ниже поднимает путь site-packages в начало sys.path и
# реэкспортирует объекты из внешнего SDK под тем же пространством имён.

# 1) Поднимем путь site-packages (purelib) в начало, чтобы следующий импорт взял SDK
_purelib = sysconfig.get_paths().get("purelib")
if _purelib and _purelib in sys.path:
    # Переместим в начало, если он уже присутствует
    sys.path.remove(_purelib)
    sys.path.insert(0, _purelib)
elif _purelib:
    sys.path.insert(0, _purelib)

# 2) Импортируем внешний пакет gigachat и прокинем ключевые символы
_sdk = importlib.import_module("gigachat")

# Реэкспорт основных символов для совместимости: from gigachat import GigaChat
GigaChat = getattr(_sdk, "GigaChat")

# Прокидываем подмодуль models так, чтобы работало: from gigachat.models import Chat, Messages, MessagesRole
models = importlib.import_module("gigachat.models")
sys.modules[__name__ + ".models"] = models

__all__ = [
    "GigaChat",
    "models",
]


