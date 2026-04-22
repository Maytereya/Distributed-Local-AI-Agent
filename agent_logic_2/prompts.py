import logging
import os
from typing import Dict

from agent_logic_2.persist import PROMPTS_DIR as _DEFAULT_PROMPTS_DIR, ensure_dir

logger = logging.getLogger(__name__)

# Папку гарантируем, но не настраиваем logging.basicConfig тут — это задача entrypoint/app
PROMPTS_DIR = ensure_dir(
    ensure_dir(_DEFAULT_PROMPTS_DIR, "prompts"),
    "prompts",
)

# Кэш «загруженных один раз» шаблонов
_PROMPT_CACHE: Dict[str, str] = {}


def load_prompt(
    name: str,
    inform: bool = True,
    required: bool = False,
    default: str = "",
) -> str | tuple[str, str]:
    """
    required=False: отсутствие файла — нормальная ситуация, вернём default и залогируем warning/debug.
    required=True: отсутствие файла — исключение (чтобы не продолжать на пустом промпте).
    """
    if name in _PROMPT_CACHE:
        msg = f"prompt {name}: cache"
        return (_PROMPT_CACHE[name], msg) if inform else _PROMPT_CACHE[name]

    path = PROMPTS_DIR / f"{name}.txt"
    if not path.exists():
        msg = f"prompt file not found: {path}"
        if required:
            logger.error(msg)
            raise FileNotFoundError(msg)
        # для опциональных промптов — не “ошибка”, максимум warning
        logger.warning(msg)
        return (default, msg) if inform else default

    text = path.read_text(encoding="utf-8")
    _PROMPT_CACHE[name] = text
    msg = f"prompt {name}: loaded"
    logger.info(msg)
    return (text, msg) if inform else text


def write_prompt(name: str, text: str) -> str:
    try:
        path = PROMPTS_DIR / f"{name}.txt"
        path.write_text(text, encoding="utf-8")
        _PROMPT_CACHE[name] = text
        logger.info("prompt %s saved", name)
        return f"✅ Промпт {name} сохранен"
    except Exception as e:
        logger.exception("Error saving prompt %s", name)
        return f"❌ Ошибка при сохранении промпта {name}: {e}"