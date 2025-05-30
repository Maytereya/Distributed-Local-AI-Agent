import json
import statistics
import time
from datetime import datetime
from typing import Tuple

from ollama import AsyncClient, Options

from agent_logic_2 import llama_func_call2 as func_mod
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
    from pathlib import Path

    log_lines = []
    results_table = []
    all_results = []

    # Создание директории и файла для лога
    log_dir = Path("data")
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"benchmark_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    for model in models:
        log_lines.append(f"\n▶️ Модель: {model}")

        async def measure(task: str) -> Tuple[float, float, float]:
            t0 = time.time()
            res = await func_mod.investigate(task)
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
