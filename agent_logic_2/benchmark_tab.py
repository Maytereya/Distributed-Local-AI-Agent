import asyncio
import csv
import json
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Callable

from ollama import AsyncClient, GenerateResponse, Options

from agent_logic_2 import ollama_settings
from agent_logic_2.config import ollama_url
from agent_logic_2.llama_func_call import with_retries
from agent_logic_2.ollama_settings import LLMName

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

TASK_GROUPS: dict[str, list[str]] = {"fast": FAST_TASKS, "slow": SLOW_TASKS}


@with_retries(tries=2)
async def ollama_test_call(prompt: str, llm: str, think: bool = None) -> tuple[
    GenerateResponse, dict[str, bool | None | str | Options]
]:
    if not llm:
        raise ValueError("Языковая модель не определена для функции 'ollama_call'")

    llm_meta = {
        "think_from_ollama_settings": think,
        "llm_as_argument": llm,
        "llm_from_ollama_settings": LLMName.get(),
        "options": ollama_settings.options_set(),
    }

    res = await ollama_client.generate(
        model=llm,
        prompt=prompt,
        options=ollama_settings.options_set(),
        think=think,
    )
    return res, llm_meta


async def measure(task: str, current_model: str, arg_think: bool) -> tuple[
    float, float | int, float | int, str, str, str, str, dict[Any, Any], Any
]:
    t0 = time.time()
    res0, llm_options = await ollama_test_call(task, current_model, think=arg_think)
    wall = time.time() - t0
    answer = getattr(res0, "response", "")
    eval_s = getattr(res0, "eval_duration", 0) / 1e9
    tokens = getattr(res0, "eval_count", len(answer.split()))

    tps = tokens / eval_s if eval_s > 0 else -1
    opt_think = "opt_think_on" if llm_options.get("think_from_ollama_settings", False) else "opt_think_off"
    arg_think_meta = "arg_think_on" if arg_think else "arg_think_off"
    engaged_llm = llm_options.get("llm_as_argument", "default")
    defaulted_llm = llm_options.get("llm_from_ollama_settings", "default")
    opt = llm_options.get("options", {})

    return wall, eval_s, tps, opt_think, arg_think_meta, engaged_llm, defaulted_llm, opt, answer


def _build_summary(model: str, task_type: str, stats: list[tuple[float, float, float]]) -> dict[str, Any]:
    if not stats:
        return {
            "model": model,
            "type": task_type,
            "wall_avg": 0.0,
            "wall_std": 0.0,
            "eval_avg": 0.0,
            "eval_std": 0.0,
            "tps_avg": 0.0,
            "tps_std": 0.0,
        }

    walls, evals, tpss = zip(*stats)
    return {
        "model": model,
        "type": task_type,
        "wall_avg": round(statistics.mean(walls), 3),
        "wall_std": round(statistics.stdev(walls), 3) if len(walls) > 1 else 0.0,
        "eval_avg": round(statistics.mean(evals), 3),
        "eval_std": round(statistics.stdev(evals), 3) if len(evals) > 1 else 0.0,
        "tps_avg": round(statistics.mean(tpss), 1),
        "tps_std": round(statistics.stdev(tpss), 1) if len(tpss) > 1 else 0.0,
    }


def _summary_to_row(summary: dict[str, Any]) -> list[Any]:
    return [
        summary["model"],
        summary["type"],
        summary["wall_avg"],
        summary["wall_std"],
        summary["eval_avg"],
        summary["eval_std"],
        summary["tps_avg"],
        summary["tps_std"],
    ]


def save_benchmark_reports(all_results: list[dict[str, Any]], suffix: str = "") -> tuple[str | None, str | None, str | None]:
    if not all_results:
        return None, None, None

    log_dir = Path("data")
    log_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = suffix.strip()
    base_name = f"benchmark_log_{stamp}{suffix}"

    json_path = log_dir / f"{base_name}.json"
    csv_path = log_dir / f"{base_name}.csv"

    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
    except Exception as e:
        return None, None, f"Ошибка сохранения JSON-отчёта: {e}"

    try:
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["model", "type", "wall_avg", "wall_std", "eval_avg", "eval_std", "tps_avg", "tps_std"],
            )
            writer.writeheader()
            for row in all_results:
                writer.writerow(row)
    except Exception as e:
        return str(json_path), None, f"Ошибка сохранения CSV-отчёта: {e}"

    return str(json_path), str(csv_path), None


async def gradio_benchmark_stream(
    models: list[str],
    laps: int = 2,
    think: bool = False,
    stop_checker: Callable[[], bool] | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    log_lines: list[str] = []
    results_table: list[list[Any]] = []
    all_results: list[dict[str, Any]] = []

    if not models:
        yield {
            "log": "🟡 Сначала загрузите и выберите модели для тестирования.",
            "table": [],
            "results": [],
            "done": True,
            "stopped": False,
            "json_path": None,
            "csv_path": None,
        }
        return

    total_runs = len(models) * sum(len(tasks) for tasks in TASK_GROUPS.values()) * max(1, int(laps))
    current_run = 0
    log_lines.append(
        f"Старт тестирования: моделей={len(models)}, прогонов={laps}, think={bool(think)}, измерений={total_runs}"
    )
    yield {
        "log": "\n".join(log_lines),
        "table": results_table,
        "results": all_results,
        "done": False,
        "stopped": False,
        "json_path": None,
        "csv_path": None,
    }

    for model in models:
        log_lines.append(f"\n▶️ Модель: {model}")
        yield {
            "log": "\n".join(log_lines),
            "table": results_table,
            "results": all_results,
            "done": False,
            "stopped": False,
            "json_path": None,
            "csv_path": None,
        }

        for task_type, task_list in TASK_GROUPS.items():
            stats: list[tuple[float, float, float]] = []

            for task in task_list:
                for _ in range(max(1, int(laps))):
                    if stop_checker and stop_checker():
                        log_lines.append("\n🛑 Остановка тестирования запрошена пользователем.")
                        json_path, csv_path, save_err = save_benchmark_reports(all_results, suffix="_partial")
                        if save_err:
                            log_lines.append(f"❌ {save_err}")
                        elif json_path and csv_path:
                            log_lines.append(f"📁 Частичный отчёт сохранён: {json_path}, {csv_path}")

                        yield {
                            "log": "\n".join(log_lines),
                            "table": results_table,
                            "results": all_results,
                            "done": True,
                            "stopped": True,
                            "json_path": json_path,
                            "csv_path": csv_path,
                        }
                        return

                    try:
                        wall, eval_s, tps, opt_think, arg_think, engaged_llm, defaulted_llm, opt, answer = await measure(
                            task,
                            model,
                            think,
                        )
                    except Exception as e:
                        current_run += 1
                        log_lines.append(f"[{current_run}/{total_runs}] ❌ Ошибка прогона: {model} / {task_type}: {e}")
                        yield {
                            "log": "\n".join(log_lines),
                            "table": results_table,
                            "results": all_results,
                            "done": False,
                            "stopped": False,
                            "json_path": None,
                            "csv_path": None,
                        }
                        continue

                    stats.append((wall, eval_s, tps))
                    current_run += 1
                    temperature = opt.get("temperature", 10.0)
                    log_lines.append(
                        f"[{current_run}/{total_runs}] [{model}] {opt_think}={arg_think}, {engaged_llm}={defaulted_llm}, "
                        f"temp={temperature}, {task_type} '{task}': wall={wall:.2f}s, eval={eval_s:.2f}s, "
                        f"tps={tps:.1f}\nОтвет: {answer}"
                    )
                    yield {
                        "log": "\n".join(log_lines),
                        "table": results_table,
                        "results": all_results,
                        "done": False,
                        "stopped": False,
                        "json_path": None,
                        "csv_path": None,
                    }

            summary = _build_summary(model, task_type, stats)
            results_table.append(_summary_to_row(summary))
            all_results.append(summary)
            log_lines.append(
                f"✅ Сводка {model}/{task_type}: wall_avg={summary['wall_avg']}, eval_avg={summary['eval_avg']}, "
                f"tps_avg={summary['tps_avg']}"
            )
            yield {
                "log": "\n".join(log_lines),
                "table": results_table,
                "results": all_results,
                "done": False,
                "stopped": False,
                "json_path": None,
                "csv_path": None,
            }

    json_path, csv_path, save_err = save_benchmark_reports(all_results)
    if save_err:
        log_lines.append(f"\n❌ {save_err}")
    elif json_path and csv_path:
        log_lines.append(f"\n📁 Отчёты сохранены: {json_path}, {csv_path}")

    yield {
        "log": "\n".join(log_lines),
        "table": results_table,
        "results": all_results,
        "done": True,
        "stopped": False,
        "json_path": json_path,
        "csv_path": csv_path,
    }


# Обратная совместимость со старым API: вернуть финальные данные одним ответом.
async def gradio_benchmark(models: list[str], laps: int = 2, think: bool = False):
    final_snapshot: dict[str, Any] = {
        "log": "",
        "table": [],
        "results": [],
        "json_path": None,
    }
    async for snapshot in gradio_benchmark_stream(models=models, laps=laps, think=think):
        final_snapshot = snapshot
    return (
        final_snapshot.get("log", ""),
        final_snapshot.get("table", []),
        final_snapshot.get("results", []),
        final_snapshot.get("json_path"),
    )


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
        ollama_test_call(
            prompt="Print out synonyms of the words 'burglar and thief' pls",
            llm=LLMName.get(),
        )
    )
    print(type(rez.response))
    print(rez.response)
