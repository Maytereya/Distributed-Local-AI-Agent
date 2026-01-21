import logging
from typing import Dict

from agent_logic_2.persist import PROMPTS_DIR, ensure_dir

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# гарантируем папку (без копирования дефолтов — это делается в main())
PROMPTS_DIR = ensure_dir(PROMPTS_DIR, "prompts")

# Кэш «загруженных один раз» шаблонов
_PROMPT_CACHE: Dict[str, str] = {}


def load_prompt(name: str, inform: bool = True) -> str | tuple[str, str]:
    if name in _PROMPT_CACHE:
        msg = f"✅ Промпт {name} загружен из кэша"
        return (_PROMPT_CACHE[name], msg) if inform else _PROMPT_CACHE[name]

    path = PROMPTS_DIR / f"{name}.txt"
    if not path.exists():
        msg = f"❌ Не найден файл промпта {name}"
        logger.info(msg)
        return ("", msg) if inform else ""

    text = path.read_text(encoding="utf-8")
    _PROMPT_CACHE[name] = text
    msg = f"✅ Промпт {name} загружен из файла"
    logger.info(msg)
    return (text, msg) if inform else text


def write_prompt(name: str, text: str) -> str:
    try:
        path = PROMPTS_DIR / f"{name}.txt"
        path.write_text(text, encoding="utf-8")
        _PROMPT_CACHE[name] = text
        logger.info("✅ Промпт %s сохранен", name)
        return f"✅ Промпт {name} сохранен"
    except Exception as e:
        logger.exception("❌ Ошибка при сохранении промпта %s", name)
        return f"❌ Ошибка при сохранении промпта {name}: {e}"