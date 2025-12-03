import logging
from pathlib import Path
from typing import Dict, Union

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Папка data/prompts внутри пакета
BASE_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = BASE_DIR / "data" / "prompts"
PROMPTS_DIR.mkdir(parents=True, exist_ok=True)

# Кэш «загруженных один раз» шаблонов
_PROMPT_CACHE: Dict[str, str] = {}


def load_prompt(name: str, inform: bool = True) -> str | tuple[str, str]:
    """
    Загружает текст промпта из кэша или файла.

    Логика:
    - если промпт уже есть в кэше — вернуть его;
    - иначе прочитать файл `<name>.txt` из каталога PROMPTS_DIR и положить в кэш;
    - если файла нет — вернуть пустую строку (и сообщение, если inform=True).

    :param name: Имя промпта без расширения (имя файла .txt).
    :param inform:
        - True: вернуть (text, message);
        - False: вернуть только text.
    :return:
        - при успехе: текст промпта, либо (текст, сообщение);
        - если файла нет: "" или ("", сообщение).
    """
    if name in _PROMPT_CACHE:
        if inform:
            return _PROMPT_CACHE[name], f"✅ Промпт {name} загружен из кэша"
        return _PROMPT_CACHE[name]

    path = PROMPTS_DIR / f"{name}.txt"

    if not path.exists():
        if inform:
            logger.info("❌ Не найден файл промпта %s", name)
            return "", f"❌ Не найден файл промпта {name}"
        return ""

    text = path.read_text(encoding="utf-8")
    _PROMPT_CACHE[name] = text
    logger.info("✅ Загружен промпт %s", name)
    if inform:
        return text, f"✅ Промпт {name} загружен из файла"
    return text


def write_prompt(name: str, text: str) -> str:
    """
    Сохраняет шаблон промпта в файл и обновляет кэш.

    :param name: Имя промпта (будет сохранён в файл <name>.txt).
    :param text: Текст промпта для сохранения.
    :return: Строка-статус операции (успех / ошибка).
    """
    try:
        path = PROMPTS_DIR / f"{name}.txt"
        path.write_text(text, encoding="utf-8")
        _PROMPT_CACHE[name] = text
        logger.info("✅ Промпт %s сохранен", name)
        return f"✅ Промпт {name} сохранен"
    except Exception as e:
        logger.error("❌ Ошибка при сохранении промпта %s: %s", name, str(e))
        return f"❌ Ошибка при сохранении промпта {name}: {str(e)}"
