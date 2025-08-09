# Нет уверенности, что options меняются должным образом.
# ToDo: Перепроверить options которые передаются способом, аналогичным для имени модели
# Think влияет только на бенчмаркинг.
# ToDo: Перепроверить данные, которые передаются в benchmark_tab
# Не все модели имеют выдачу генерации, названную также как и у Mistral. Поэтому может быть пустое сообщение от модели.
# Скорее всего это касается рассуждающих моделей. Перепроверить.

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Union

from ollama import Options

from agent_logic_2 import config as c

# Настройка логирования
# logging.basicConfig(level=logging.INFO)
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
# Инициализация кешей
OLLAMA_MODEL: str
OLLAMA_MODEL = ""
_cached_opts: Dict
_cached_opts = {}


# --------------------------------------------------
# ------------  OLLAMA OPTIONS SECTION ------------
# --------------------------------------------------

def init_options():
    settings_dir.mkdir(parents=True, exist_ok=True)
    # we are Loading JSON into cache here
    global _cached_opts
    if settings_path.exists():
        _cached_opts = json.loads(settings_path.read_text(encoding='utf-8'))
    else:
        _cached_opts = _OPTIONS["expressive"]
    return _cached_opts


def load_ollama_options(explain: bool = True, ) -> Union[str, (str, str)]:
    """
    Загружает данные из сохраненного файла, если он существует.
    Returns:
        Dict: Данные для базовой настройки Ollama.
    """
    try:
        init_options()
        data: str = json.dumps(_cached_opts, ensure_ascii=False, indent=2)
        logger.info("✅ Загружены данные о настройках Ollama")
        if explain:
            return data, f"✅ Загружены данные о настройках Ollama для {OLLAMA_MODEL}"
        return data

    except Exception as e:
        global _OPTIONS
        logger.error(
            f"\n❌ Ошибка при чтении файла с настройками Ollama: {e}, загружены базовые настройки для {OLLAMA_MODEL}.")
        if explain:
            return _OPTIONS, f"⚠️ Установлены дефолтные настройки Ollama для {OLLAMA_MODEL} в связи с ошибкой: {e}"
        return _OPTIONS  # Если пойдет не так, возвращаем базовый дефолт, вшитый в код


def options_set() -> Options:
    """
      :return: Options set for Ollama.
      """
    return Options(
        **_cached_opts)  # важно, чтобы передался Dict, а не str. Без сообщений в строку Status интерфейса


def write_options(data: Dict) -> str:
    try:
        settings_dir.mkdir(parents=True, exist_ok=True)
        with open(settings_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            global _cached_opts
            _cached_opts = data  # обновляем кэш
            return f"✅ Данные о настройках Ollama для {OLLAMA_MODEL} сохранены"
    except Exception as e:
        return f"❌ Ошибка при сохранении настроек Ollama: {e} для модели {OLLAMA_MODEL}"


# --------------------------------------------------
# ------------  MAIN MODEL NAME SECTION ------------
# --------------------------------------------------

def init_model_name():
    """
    """
    settings_dir.mkdir(parents=True, exist_ok=True)
    global OLLAMA_MODEL
    if main_model_path.exists():
        OLLAMA_MODEL = main_model_path.read_text(encoding="utf-8").strip()
        if OLLAMA_MODEL != "":
            logger.info("✅ Инициализировано имя базовой LLM: %s из кэша ollama_settings", OLLAMA_MODEL)
        else:
            OLLAMA_MODEL = c.ll_model_small
            logger.info("✅ Инициализировано имя базовой LLM: %s из config.ini", OLLAMA_MODEL)
    return OLLAMA_MODEL


def load_main_model_name(inform: bool = True) -> Union[str, (str, str)]:
    init_model_name()
    global OLLAMA_MODEL
    logger.info("✅ Загружено имя модели %s", OLLAMA_MODEL)
    if inform:
        return OLLAMA_MODEL, f"✅ Имя модели {OLLAMA_MODEL} загружено"
    else:
        return OLLAMA_MODEL


def write_main_model_name(name: str, ) -> str:
    """
    Сохраняет имя выбранной LLM в файл и в кэш.
    Кеш - строка с именем.
    """
    global _model_cache
    _model_cache = name
    try:
        main_model_path.write_text(name, encoding="utf-8")
        logger.info("✅ Имя модели %s сохранено", name)
        return f"✅ Выбрана модель {name} в качестве основной"
    except Exception as e:
        logger.error("❌ Ошибка при сохранении имени модели %s: %s", name, str(e))
        return f"❌ Ошибка при сохранении имени модели {name}: {str(e)}"


if __name__ == "__main__":
    print(write_main_model_name("gpt-4"))
    print(load_main_model_name())
    print(OLLAMA_MODEL)
