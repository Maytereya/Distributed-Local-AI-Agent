from typing import Literal

from ollama import Options

from agent_logic_2 import config as c

# Дополнительный узел синхронизации моделей,
# чтобы не вызывать разные случайно.
OLLAMA_MODEL = c.ll_model_small

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


def options_set(level: Literal["conservative", "expressive"] = "expressive") -> Options:
    """
      Выбор из двух вариантов настроек генерации. Для удобства подбора.
      :param level: Conservative - изначальный вариант, expressive - вариант с чуть большей свободой.
      :return: Option for Ollama.
      """
    params = _OPTIONS.get(level, _OPTIONS["expressive"])
    return Options(**params)
