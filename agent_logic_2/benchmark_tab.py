import asyncio
import statistics
import time
from typing import Tuple

from ollama import AsyncClient, Options

from agent_logic_2 import llama_func_call1 as func_mod
from agent_logic_2.config import ollama_url

# Настройки клиента Ollama
ollama_client = AsyncClient(ollama_url)
orig_options = Options(temperature=0)


FAST_TASKS = [
    "Назовите три языка программирования",
    "Сколько дней в високосном году?",
]

SLOW_TASKS = [
    "Напишите развёрнутый конспект по принципам REST API",
    "Сгенерируйте код ORM-слоя для сложной ER-модели из 10 таблиц",
]


# ------------------------------------------
# ГЛАВНАЯ ФУНКЦИЯ: тестирует список моделей
# ------------------------------------------

async def gradio_benchmark(models: list[str], laps: int = 3):
    log_lines = []
    results_table = []

    for model in models:

        log_lines.append(f"\n▶️ Модель: {model}")


        async def measure(task: str) -> Tuple[float, float, float]:
            t0 = time.time()
            res = await func_mod.investigate(task, model=model)
            wall = time.time() - t0
            eval_s = res.get("eval_duration", 0) / 1e9 if isinstance(res, dict) else 0
            tokens = res.get("eval_count", len(str(res).split())) if isinstance(res, dict) else len(str(res).split())
            tps = tokens / eval_s if eval_s > 0 else 0
            return wall, eval_s, tps

        for task_type, task_list in {"fast": FAST_TASKS, "slow": SLOW_TASKS}.items():
            stats = []
            for task in task_list:
                for i in range(laps):
                    wall, eval_s, tps = await measure(task)
                    stats.append((wall, eval_s, tps))
                    log_lines.append(
                        f"[{model}] {task_type} '{task}': wall={wall:.2f}s, eval={eval_s:.2f}s, tps={tps:.1f}")

            walls, evals, tpss = zip(*stats)
            row = [
                model,
                task_type,
                round(statistics.mean(walls), 3), round(statistics.stdev(walls), 3) if len(walls) > 1 else 0,
                round(statistics.mean(evals), 3), round(statistics.stdev(evals), 3) if len(evals) > 1 else 0,
                round(statistics.mean(tpss), 1), round(statistics.stdev(tpss), 1) if len(tpss) > 1 else 0,
            ]
            results_table.append(row)

    return "\n".join(log_lines), results_table


if __name__ == "__main__":
    asyncio.run(gradio_benchmark(["mistral-small3.1:24b-instruct-2503-q8_0"]))
