from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict

from ollama import Options

from agent_logic_2 import config as c

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Резервная копия OPTIONS for Ollama, вшитая в код
_OPTIONS = {
    "conservative": {
        "temperature": 0.1,
        "top_k": 30,
        "top_p": 0.9,
        "repeat_penalty": 1.1,
        "stop": ["<|eot_id|>"],
    },
    "expressive": {
        "temperature": 0.15,
        "top_p": 1,
        "top_k": 0,
        "repeat_penalty": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "max_new_tokens": 4096,
        "stop": ["<|eot_id|>", "— Конец списка —"]
    },
}

# Создание директории и файла для сохранения
# 1. Берём директорию, в которой лежит этот скрипт
BASE_DIR: Path = Path(__file__).resolve().parent

# 2. Формируем нужный путь к папке и файлу
settings_dir: Path = BASE_DIR / "data" / "settings"
settings_path: Path = settings_dir / "ollama_request_settings.json"
main_model_path: Path = settings_dir / "main_model.txt"

# Дополнительный узел синхронизации моделей,
# чтобы не вызывать разные случайно.

# OLLAMA_MODEL: str = c.ll_model_small
# _MODEL_CACHE: str = ""

if main_model_path.read_text(encoding="utf-8"):
    _MODEL_CACHE = main_model_path.read_text(encoding="utf-8")
    OLLAMA_MODEL = _MODEL_CACHE
    logger.info("✅ Загружено имя базовой LLM: %s из кэша", _MODEL_CACHE)
    print("OLLAMA_MODEL changed to: ", OLLAMA_MODEL)
else:
    OLLAMA_MODEL: str = c.ll_model_small
    logger.info("✅ Загружено имя базовой LLM: %s из ini", OLLAMA_MODEL)
    print("data was taken from c.ll_model_small")


def load_settings(explain: bool = True, raw_mode: bool = False) -> (str, str) | str:
    """
    Загружает данные из сохраненного файла, если он существует.
    Returns:
        Dict: Данные для базовой настройки Ollama.
    """
    # На случай если настройки из файла получить не удалось:
    raw_options: Dict = _OPTIONS["expressive"]
    default_options: str = json.dumps(raw_options, ensure_ascii=False,
                                      indent=4)

    if settings_path.exists():
        try:
            with open(settings_path, 'r', encoding='utf-8') as f:
                raw: dict = json.load(f)
                data = json.dumps(raw, ensure_ascii=False, indent=4)
                logger.info("✅ Загружены данные о настройках Ollama")

                if raw_mode:
                    return raw
                if explain:
                    return data, f"✅ Загружены данные о настройках Ollama для {OLLAMA_MODEL}"
                return data

        except Exception as e:
            logger.error(
                f"\n❌ Ошибка при чтении файла с настройками Ollama: {e}, \nзагружены базовые настройки для {OLLAMA_MODEL}.")
            if raw_mode:
                return raw_options
            if explain:
                return default_options, f"⚠️ Установлены дефолтные настройки Ollama для {OLLAMA_MODEL} в связи с ошибкой: {e}"
            return default_options  # Если пойдет не так, возвращаем базовый дефолт
    else:
        logger.error(
            f"\n❌ Ошибка при чтении файла с настройками Ollama: нет пути к файлу настроек, \nзагружены базовые настройки для {OLLAMA_MODEL}.")
        if raw_mode:
            return raw_options
        if explain:
            return default_options, f"⚠️ Установлены дефолтные настройки Ollama для {OLLAMA_MODEL} в связи с отсутствием пути к файлу настроек"
        return default_options  # Если все окончательно пойдет не так, возвращаем базовый дефолт


# Кеширование для быстрой передачи, минуя загрузку из файла
_cached_opts: Dict = load_settings(explain=False, raw_mode=True)


def options_set() -> Options:
    """
      :return: Options set for Ollama.
      """
    return Options(**_cached_opts)  # важно, чтобы передался Dict, а не str.


def write_settings(data: Dict) -> str:
    try:
        settings_dir.mkdir(parents=True, exist_ok=True)
        with open(settings_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            global _cached_opts
            _cached_opts = data  # обновляем кэш
            logger.info(f"✅ Данные о настройках Ollama сохранены")
            return f"✅ Данные о настройках Ollama для {OLLAMA_MODEL} сохранены"
    except Exception as e:
        logger.info(f"\n❌ Ошибка при сохранении настроек Ollama: {e}")
        return f"❌ Ошибка при сохранении настроек Ollama: {e} для модели {OLLAMA_MODEL}"


def load_main_model_name(inform: bool = True) -> (str, str) or str:
    name: str = ""
    try:
        global _MODEL_CACHE
        name = main_model_path.read_text(encoding="utf-8")
        if name and inform:
            _MODEL_CACHE = name
            logger.info("✅ Выбрано имя модели %s", name)
            return name, f"✅ Имя модели {name} загружено из файла"
        elif name and not inform:
            logger.info("✅ Выбрано имя модели %s", name)
            return name
        elif not name and inform:
            return OLLAMA_MODEL, f"Загружено имя модели по умолчанию: {OLLAMA_MODEL}"
        else:
            return OLLAMA_MODEL
    except Exception as e:
        logger.error("❌ Ошибка при загрузке имени модели %s: %s", name, str(e))
        if inform:
            return OLLAMA_MODEL, (f"❌ Ошибка при загрузке имени {str(e)}, "
                                  f"загружено имя модели по умолчанию: {OLLAMA_MODEL}")
        return OLLAMA_MODEL


def write_main_model_name(name: str, ) -> str:
    """
    Сохраняет имя выбранной LLM в файл и в кэш.
    Кеш - строка с именем.
    """
    try:
        main_model_path.write_text(name, encoding="utf-8")
        global _MODEL_CACHE
        _MODEL_CACHE = name
        logger.info("✅ Имя модели %s сохранено", name)
        return f"✅ Выбрана модель {name} в качестве основной"
    except Exception as e:
        logger.error("❌ Ошибка при сохранении имени модели %s: %s", name, str(e))
        return f"❌ Ошибка при сохранении имени модели {name}: {str(e)}"


if __name__ == "__main__":
    # write_settings()
    # print(load_settings())
    # print(settings_path)
    print("load_main_model_name------------------")
    print(load_main_model_name(inform=True))
    print("_MODEL_CACHE: ", _MODEL_CACHE)
