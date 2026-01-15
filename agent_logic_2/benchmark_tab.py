import asyncio
import json
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from ollama import AsyncClient, GenerateResponse, Options

from agent_logic_2 import ollama_settings
from agent_logic_2.ollama_settings import LLMName #

from agent_logic_2.config import ollama_url
from agent_logic_2.llama_func_call import with_retries

# Настройки клиента Ollama
ollama_client = AsyncClient(ollama_url)

FAST_TASKS = [
    "Назовите три языка программирования",
    "Сколько дней в високосном году?",
]

SLOW_TASKS = [
    "Напишите развёрнутый конспект по принципам REST API",
    "Сгенерируйте код ORM-слоя для сложной ER-модели из 10 таблиц",
]


@with_retries(tries=2)
async def ollama_test_call(prompt: str, llm: str, think: bool = None, ) -> tuple[
    tuple[GenerateResponse], dict[str, bool | None | str | Options]]:
    if not llm:
        raise ValueError("Языковая модель не определена для функции 'ollama_call'")
    # think = ollama_settings.resolve_think(think)
    # Словарь для вывода актуальных настроек Ollama в момент генерации ответа
    llm_meta = {"think_from_ollama_settings": think,
                "llm_as_argument": llm,
                "llm_from_ollama_settings": ollama_settings.OLLAMA_MODEL,
                "options": ollama_settings.options_set(), }

    res = await ollama_client.generate(
        model=llm,
        prompt=prompt,
        options=ollama_settings.options_set(),
        # keep_alive=-1,
        think=think,
    ),

    return res, llm_meta


async def measure(task: str, current_model: str, arg_think: bool) -> tuple[
    float, float | int, float | int, str, str, str, str, dict[Any, Any], Any]:
    t0 = time.time()
    res0, llm_options = await ollama_test_call(task,
                                               current_model,
                                               think=arg_think,
                                               )
    wall = time.time() - t0
    # res0 — это GenerateResponse ИЛИ dict
    if isinstance(res0, tuple):
        answer = res0[0].response
        eval_s = res0[0].eval_duration / 1e9
        tokens = res0[0].eval_count or len(answer.split())
    else:
        answer = getattr(res0, "response", "")
        eval_s = getattr(res0, "eval_duration", 0) / 1e9
        tokens = getattr(res0, "eval_count", len(answer.split()))

    tps = tokens / eval_s if eval_s > 0 else -1
    opt_think = llm_options.get("think_from_ollama_settings", False)
    if opt_think:
        opt_think = "opt_think_on"
    else:
        opt_think = "opt_think_off"
    if arg_think:
        arg_think = "arg_think_on"
    else:
        arg_think = "arg_think_off"
    engaged_llm = llm_options.get("llm_as_argument", "default")
    defaulted_llm = llm_options.get("llm_from_ollama_settings", "default")
    opt = llm_options.get("options", {})

    return wall, eval_s, tps, opt_think, arg_think, engaged_llm, defaulted_llm, opt, answer


# ------------------------------------------
# ГЛАВНАЯ ФУНКЦИЯ: тестирует список моделей
# ------------------------------------------

async def gradio_benchmark(models: list[str], laps: int = 2, think: bool = False):
    log_lines = []
    results_table = []
    all_results = []

    # Создание директории и файла для лога
    log_dir = Path("data")
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"benchmark_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    for model in models:
        log_lines.append(f"\n▶️ Модель: {model}")
        for task_type, task_list in {"fast": FAST_TASKS, "slow": SLOW_TASKS}.items():
            stats = []
            for task in task_list:
                for i in range(laps):
                    wall, eval_s, tps, opt_think, arg_think, engaged_llm, defaulted_llm, opt, answer = await measure(task,
                                                                                                             model,
                                                                                                             think)
                    stats.append((wall, eval_s, tps))
                    temperature = opt.get("temperature", 10.0)
                    log_lines.append(
                        f"[{model}], {opt_think} = {arg_think}, {engaged_llm} = {defaulted_llm}, {temperature}, "
                        f"{task_type} '{task}': wall={wall:.2f}s, eval={eval_s:.2f}s, tps={tps:.1f}, \nОтвет: {answer}")

            walls, evals, tpss = zip(*stats)
            summary = {
                "model": model,
                "type": task_type,
                "wall_avg": round(statistics.mean(walls), 3),
                "wall_std": round(statistics.stdev(walls), 3) if len(walls) > 1 else 0,
                "eval_avg": round(statistics.mean(evals), 3),
                "eval_std": round(statistics.stdev(evals), 3) if len(evals) > 1 else 0,
                "tps_avg": round(statistics.mean(tpss), 1),
                "tps_std": round(statistics.stdev(tpss), 1) if len(tpss) > 1 else 0,
            }

            results_table.append([
                summary["model"], summary["type"],
                summary["wall_avg"], summary["wall_std"],
                summary["eval_avg"], summary["eval_std"],
                summary["tps_avg"], summary["tps_std"]
            ])

            all_results.append(summary)

    try:
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log_lines.append(f"\n❌ Ошибка при сохранении лога: {e}")

    return "\n".join(log_lines), results_table, all_results, str(log_path)


# ------------------------------------------------------
# Загрузка предыдущего отчёта для отображения в таблице
# ------------------------------------------------------

def load_previous_log(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        table = [
            [item["model"], item["type"], item["wall_avg"], item["wall_std"],
             item["eval_avg"], item["eval_std"], item["tps_avg"], item["tps_std"]]
            for item in data
        ]
        return table
    except Exception as e:
        return [["Ошибка при загрузке файла:", str(e)]]


if __name__ == "__main__":
    rez, _ = asyncio.run(
        ollama_test_call(prompt="Print out synonyms of the words 'burglar and thief' pls",
                         llm=LLMName.get(), ))


    print(type(rez[0].response))
    print(rez[0].response)
