# ToDo: Перепроверить данные, которые передаются в benchmark_tab
# Не все модели имеют выдачу генерации, названную также как и у Mistral. Поэтому может быть пустое сообщение от модели.
# Скорее всего это касается рассуждающих моделей. Перепроверить.

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple, Union

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
think_path: Path = settings_dir / "think.txt"

# Дополнительный узел синхронизации моделей,
# Инициализация кешей
OLLAMA_MODEL: str
OLLAMA_MODEL = ""
_cached_opts: Dict
_cached_opts = {}
_think: bool
_think = False


# --------------------------------------------------
# -----  OLLAMA THINK (REASONING) секция  ----------
# --------------------------------------------------

# ---- THINK helpers -------------------------------
def get_think() -> bool:
    """
    Возвращает текущее значение флага Think.

    Если значение ещё не инициализировано, выполняет init_thinking()
    и берёт состояние из файла / config.ini.

    :return: Текущее значение параметра Think (True/False).
    """
    global _think
    if "_think" not in globals() or _think is None:
        init_thinking()
    return _think


def set_think(status: bool) -> None:
    """
    Устанавливает новое значение флага Think и сохраняет его в файл.

    Тонкая обёртка над write_think_status: обновляет и кэш, и файл.

    :param status: Новое состояние параметра Think (True/False).
    """
    write_think_status(bool(status))


def resolve_think(override: bool | None) -> bool | None:
    """
    Возвращает финальное значение флага Think с учётом override.

    :param override:
        - None — вернуть текущее значение Think из настроек (файл / config.ini);
        - True/False — вернуть переданное значение, не трогая файл.
    :return: Итоговое значение параметра Think или None (если явно так задано).
    """
    return get_think() if override is None else override


# -----------------------------------------------------
def init_thinking() -> bool:
    """
    Инициализирует флаг Think при старте приложения.

    Логика:
    - если существует think.txt — читаем значение из файла;
    - иначе берём дефолт из config.ini (c.think) и считаем его текущим.

    :return: Инициализированное значение параметра Think.
    """
    settings_dir.mkdir(parents=True, exist_ok=True)
    global _think
    if think_path.exists():
        _think = bool(think_path.read_text(encoding="utf-8").strip())
        if _think:
            logger.info("✅ Инициализировано состояние параметра Think: %s из кэша", _think)
    else:
        _think = bool(c.think)
        logger.info("✅ Инициализировано состояние параметра Think: %s из config.ini", _think)
    return _think


def write_think_status(status: bool, ) -> str:
    """
    Сохраняет статус параметра Think в файл и обновляет кэш.

    При True в файле хранится строка "True".
    При False файл очищается, но состояние всё равно фиксируется в кэше.

    :param status: Новое состояние параметра Think.
    :return: Строка-статус операции (успех / ошибка).
    """
    global _think
    _think = bool(status)
    if status:
        try:
            think_path.write_text(str(status), encoding="utf-8")
            logger.info("✅ Статус параметра Think %s сохранен", status)
            return f"✅ Статус параметра Think {status} сохранен"
        except Exception as e:
            logger.error("❌ Ошибка при сохранении статуса параметра Think %s: %s", status, str(e))
            return f"❌ Ошибка при сохранении статуса параметра Think {status}: {str(e)}"
    else:
        try:
            think_path.write_text("", encoding="utf-8")
            logger.info("✅ Статус параметра Think %s сохранен", status)
            return f"✅ Статус параметра Think {status} сохранен"
        except Exception as e:
            logger.error("❌ Ошибка при сохранении статуса параметра Think %s: %s", status, str(e))
            return f"❌ Ошибка при сохранении статуса параметра Think {status}: {str(e)}"


def read_think_status(inform: bool = True) -> Union[Tuple[bool, str], bool]:
    """
    Читает и инициализирует состояние Think, возвращая его (с текстом или без).

    Вызов всегда приводит к init_thinking() и обновлению глобального _think.

    :param inform:
        - True: вернуть (status, message);
        - False: вернуть только status.
    :return: bool или (bool, str) в зависимости от флага inform.
    """
    status = init_thinking()

    global _think
    _think = status

    logger.info("✅ Загружено состояние параметра Think: %s", str(status))
    if inform:
        return status, f"Загружено состояние параметра Think: {str(status)}"
    else:
        return status


# --------------------------------------------------
# ------------  OLLAMA OPTIONS СЕКЦИЯ --------------
# --------------------------------------------------

def init_options() -> Dict[str, Any]:
    """
    Инициализирует кэш настроек Ollama из файла или дефолтных значений.

    Если settings_path существует, пытается прочитать JSON и проверить, что это dict.
    При любой ошибке чтения/разбора используется пресет _OPTIONS["expressive"].

    :return: Текущий словарь настроек Ollama (_cached_opts).
    """
    settings_dir.mkdir(parents=True, exist_ok=True)
    global _cached_opts

    if settings_path.exists():
        try:
            loaded = json.loads(settings_path.read_text(encoding='utf-8'))
            if not isinstance(loaded, dict):
                raise ValueError("Настройки Ollama должны быть словарём (dict)")
            _cached_opts = loaded
        except Exception as e:
            logger.error(
                "❌ Ошибка при чтении/разборе настроек Ollama (%s), "
                "используются дефолтные значения 'expressive'", e
            )
            _cached_opts = _OPTIONS["expressive"]
    else:
        _cached_opts = _OPTIONS["expressive"]

    return _cached_opts


def load_ollama_options(explain: bool = True) -> str | tuple[str, str]:
    """
    Загружает текущие настройки Ollama и возвращает их в виде JSON-строки.

    Источник:
    - если файл настроек существует и корректен — берёт данные из него;
    - при ошибке чтения/разбора — использует дефолтные настройки 'expressive'.

    :param explain:
        - True: вернуть (json_str, message);
        - False: вернуть только json_str.
    :return: JSON-строка с настройками (и опционально текстовое сообщение).
    """
    try:
        opts = init_options()
        data: str = json.dumps(opts, ensure_ascii=False, indent=2)
        logger.info("✅ Загружены данные о настройках Ollama")
        if explain:
            return data, f"✅ Загружены данные о настройках Ollama для {OLLAMA_MODEL}"
        return data

    except Exception as e:
        # сюда мы, по идее, попадать не должны, но на всякий случай
        logger.error(
            "❌ Критическая ошибка при загрузке настроек Ollama: %s, "
            "пытаемся использовать дефолтные значения 'expressive'", e
        )
        fallback = _OPTIONS["expressive"]
        data = json.dumps(fallback, ensure_ascii=False, indent=2)
        if explain:
            return data, (
                f"⚠️ Установлены дефолтные настройки Ollama для {OLLAMA_MODEL} "
                f"в связи с ошибкой: {e}"
            )
        return data


def options_set() -> Options:
    """
    Преобразует текущие кэшированные настройки Ollama в объект Options.

    Перед вызовом ожидается, что init_options()/load_ollama_options()
    уже были вызваны и _cached_opts содержит валидный словарь.

    :return: Объект ollama.Options, готовый к передаче в клиент Ollama.
    """
    return Options(**_cached_opts)


def write_options(data: Dict) -> str:
    """
    Сохраняет настройки Ollama в JSON-файл и обновляет кэш.

    :param data: Словарь с настройками (параметры для Ollama).
    :return: Строка-статус операции (успех / ошибка).
    """
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
# ------------  MAIN MODEL NAME СЕКЦИЯ ---0---------
# --------------------------------------------------

def init_model_name() -> str:
    """
    Инициализирует имя основной модели Ollama (OLLAMA_MODEL).

    Логика:
    - если main_model_path существует и содержит непустое имя —
      читаем его и используем как базовую LLM;
    - иначе берём значение из config.ini (c.ll_model_small).

    :return: Текущее имя базовой модели (строка).
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


def read_main_model_name(inform: bool = True) -> str | tuple[str, str]:
    """
    Читает имя основной модели Ollama (из файла или config.ini) и возвращает его.

    Всегда вызывает init_model_name(), тем самым гарантируя актуальное
    значение в глобальной переменной OLLAMA_MODEL.

    :param inform:
        - True: вернуть (name, message);
        - False: вернуть только name.
    :return: Строка с именем модели или кортеж (name, message).
    """
    init_model_name()
    global OLLAMA_MODEL
    logger.info("✅ Загружено имя модели %s", OLLAMA_MODEL)
    if inform:
        return OLLAMA_MODEL, f"✅ Имя модели {OLLAMA_MODEL} загружено"
    else:
        return OLLAMA_MODEL


def write_main_model_name(name: str) -> str:
    """
    Сохраняет имя выбранной LLM в файл и обновляет кэш (OLLAMA_MODEL).

    :param name: Имя модели Ollama (например, "llama3.1:8b").
    :return: Строка-статус операции (успех / ошибка).
    """
    global OLLAMA_MODEL
    OLLAMA_MODEL = name
    try:
        main_model_path.write_text(name, encoding="utf-8")
        logger.info("✅ Имя модели %s сохранено", name)
        return f"✅ Выбрана модель {name} в качестве основной"
    except Exception as e:
        logger.error("❌ Ошибка при сохранении имени модели %s: %s", name, str(e))
        return f"❌ Ошибка при сохранении имени модели {name}: {str(e)}"


if __name__ == "__main__":
    print(write_think_status(False))
    print(read_think_status())
    print(_think)
