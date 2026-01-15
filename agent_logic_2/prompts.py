import logging
import shutil
from pathlib import Path
from importlib import resources
from typing import Dict, Union
import config as c
# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Папка data/prompts внутри пакета
DEFAULT_DATA_DIR = Path(c.APP_DATA_DIR).resolve()
PROMPTS_DIR = Path(DEFAULT_DATA_DIR / "prompts").resolve()

PKG = "agent_logic_2"  # имя пакета, откуда идет загрузка дефолтных промптов для агента
DEFAULTS_SUBDIR = ("data", "prompts")  # адрес, где лежат дефолтные .txt внутри пакета

def ensure_default_prompts(dest_dir: Path, overwrite: bool = False) -> int:
    """
    Копирует дефолтные *.txt промпты из пакета в dest_dir.
    По умолчанию НЕ перезаписывает существующие файлы.

    :return: сколько файлов скопировано
    """
    dest_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    src_root = resources.files(PKG).joinpath(*DEFAULTS_SUBDIR)

    # src_root — Traversable (может быть внутри zip/wheel)
    for entry in src_root.iterdir():
        if not entry.is_file():
            continue
        if entry.name.startswith(".") or entry.suffix.lower() != ".txt":
            continue

        target = dest_dir / entry.name
        if target.exists() and not overwrite:
            continue

        # открываем ресурс как бинарный поток (работает и из wheel)
        with entry.open("r", encoding="utf-8") as r:
            target.write_text(r.read(), encoding="utf-8")

        copied += 1

    if copied:
        logger.info("✅ Скопировано дефолтных промптов: %d в %s", copied, dest_dir)
    else:
        logger.info("ℹ️ Дефолтные промпты уже на месте: %s", dest_dir)

    return copied

try:
    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
except (OSError, PermissionError) as e:
    logger.error("❌ Не удалось создать директорию для промптов: %s. Используется временная директория.", e)
    # Резервный вариант в текущей папке, если основной путь недоступен
    PROMPTS_DIR = Path.cwd() / "prompts"
    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)

# Копируем дефолтные промпты в рабочую папку
ensure_default_prompts(PROMPTS_DIR, overwrite=False)

# BASE_DIR = Path(__file__).resolve().parent
# PROMPTS_DIR = BASE_DIR / "data" / "prompts"
# PROMPTS_DIR.mkdir(parents=True, exist_ok=True)


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
