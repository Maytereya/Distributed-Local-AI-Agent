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


def load_prompt(name: str, inform: bool = True) -> Union[(str, str) or str]:
    """
    Loads a prompt from a file or the cache. If the specified prompt is found in
    the cache, returns it directly. Otherwise, attempts to read the prompt
    from a designated directory. Optionally, logs information about the process
    and includes any relevant messages in the return value.

    If the prompt file does not exist and logging is enabled, an information log
    entry is generated. The function optionally returns an empty string and an
    appropriate message in such a case.

    :param name: The name of the prompt file (without the extension) to load.
    :type name: str
    :param inform: A flag indicating whether to log messages or include status
                   messages in the returned tuple.
                   Default is True.
    :type inform: bool
    :return: The loaded prompt or a tuple containing the loaded prompt and an
             informational message. If inform is False, only the prompt is
             returned. If the prompt does not exist, an empty string and a
             message are returned when inform is True; otherwise, only an
             empty string is returned.
    :rtype: (str, str) or str
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
            # raise FileNotFoundError(f"Prompt file not found: {path}")
        return ""

    text = path.read_text(encoding="utf-8")
    _PROMPT_CACHE[name] = text
    logger.info("✅ Загружен промпт %s", name)
    if inform:
        return text, f"✅ Промпт {name} загружен из файла"
    return text


def write_prompt(name: str, text: str) -> str:
    """
    Сохраняет отредактированный шаблон в файл и в кэш.
    Кеш - словарь, ключ - имя файла, значение - текст в файле.
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
