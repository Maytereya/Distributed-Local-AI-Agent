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

# Дополнительный узел синхронизации моделей,
# чтобы не вызывать разные случайно.
OLLAMA_MODEL: str = c.ll_model_small

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


def options_set() -> Options:
    """
      :return: Option for Ollama.
      """
    return Options(**load_settings())


# Создание директории и файла для сохранения
# 1. Берём директорию, в которой лежит этот скрипт
BASE_DIR = Path(__file__).resolve().parent

# 2. Формируем нужный путь к папке и файлу
settings_dir = BASE_DIR / "data" / "settings"
settings_path = settings_dir / "ollama_request_settings.json"


def load_settings(explain: bool = True) -> (str, str) | str:
    """
    Загружает данные из сохраненного файла, если он существует.
    Returns:
        Dict: Данные для базовой настройки Ollama.
    """
    default_options = json.dumps(_OPTIONS.get(OLLAMA_MODEL, _OPTIONS["expressive"]), ensure_ascii=False,
                                 indent=4)
    if settings_path.exists():
        try:
            with open(settings_path, 'r', encoding='utf-8') as f:
                raw: dict = json.load(f)
                data = json.dumps(raw, ensure_ascii=False, indent=4)

                logger.info("✅ Загружены данные о настройках Ollama")
                if explain:
                    return data, f"✅ Загружены данные о настройках Ollama для {OLLAMA_MODEL}"
                return data

        except Exception as e:
            logger.error(
                f"\n❌ Ошибка при чтении файла с настройками Ollama: {e}, \nзагружены базовые настройки для {OLLAMA_MODEL}.")
            if explain:
                return default_options, f"⚠️ Установлены дефолтные настройки Ollama для {OLLAMA_MODEL} в связи с ошибкой: {e}"
            return default_options  # Если пойдет не так, возвращаем базовый дефолт
    else:
        logger.error(
            f"\n❌ Ошибка при чтении файла с настройками Ollama: нет пути к файлу настроек, \nзагружены базовые настройки для Mistral Small.")
        if explain:
            return default_options, f"⚠️ Установлены дефолтные настройки Ollama для {OLLAMA_MODEL} в связи с отсутствием пути к файлу настроек"
        return default_options  # Если все окончательно пойдет не так, возвращаем базовый дефолт


def write_settings(data: Dict) -> str:
    try:
        settings_dir.mkdir(parents=True, exist_ok=True)
        with open(settings_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"✅ Данные о настройках Ollama сохранены")
            return f"✅ Данные о настройках Ollama для {OLLAMA_MODEL} сохранены"
    except Exception as e:
        logger.info(f"\n❌ Ошибка при сохранении настроек Ollama: {e}")
        return f"❌ Ошибка при сохранении настроек Ollama: {e} для модели {OLLAMA_MODEL}"


if __name__ == "__main__":
    # write_settings()
    print(load_settings())
    print(settings_path)
