import logging
from pathlib import Path
from typing import Dict

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Папка data/prompts внутри пакета
BASE_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = BASE_DIR / "data" / "prompts"
PROMPTS_DIR.mkdir(parents=True, exist_ok=True)

# Кэш «загруженных один раз» шаблонов
_PROMPT_CACHE: Dict[str, str] = {}


def load_prompt(name: str) -> (str, str) :
    """
    Возвращает текст шаблона из файла или из кэша.
    Подставлять параметры и форматировать нужно снаружи - через интерфейс Gradio.
    """
    if name in _PROMPT_CACHE:
        return _PROMPT_CACHE[name], f"✅ Промпт {name} загружен из кэша"

    path = PROMPTS_DIR / f"{name}.txt"

    if not path.exists():
        logger.info("❌ Не найден файл промпта %s", name)
        return "", f"❌ Не найден файл промпта {name}"
        # raise FileNotFoundError(f"Prompt file not found: {path}")

    text = path.read_text(encoding="utf-8")
    _PROMPT_CACHE[name] = text
    logger.info("✅ Загружен промпт %s", name)
    return text, f"✅ Промпт {name} загружен из файла"


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
