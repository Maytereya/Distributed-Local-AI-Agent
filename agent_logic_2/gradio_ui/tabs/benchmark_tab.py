import csv
import json
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Callable

import gradio as gr
from ollama import AsyncClient, GenerateResponse, Options

from agent_logic_2 import config as c
from agent_logic_2 import ollama_settings
from agent_logic_2.gradio_ui.handlers.benchmark_runtime import (
    benchmark_finish,
    benchmark_is_running,
    benchmark_request_stop,
    benchmark_stop_requested,
    benchmark_try_start,
)
from agent_logic_2.llama_func_call import with_retries
from agent_logic_2.ollama_settings import LLMName

# Настройки клиента Ollama
ollama_client = AsyncClient(c.ollama_url)

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

def build_benchmark_tab(
    *,
    blocks,
    fn_load_main_model_fn: Callable[[], list[str]],
    think_checkbox,
    extract_model_names_fn: Callable[[Any], list[str]],
    tab_visible: bool = False,
) -> dict[str, Any]:
    with gr.Tab("🚀️ Тестирование производительности", visible=tab_visible) as benchmark_tab:
        gr.Markdown("""<h3>⚙️ Тестирование производительности генеративных моделей и сервера Ollama</h3>""")
        with gr.Row():
            initial_benchmark_models = fn_load_main_model_fn()
            model_selector = gr.Dropdown(
                choices=initial_benchmark_models,
                value=initial_benchmark_models,
                multiselect=True,
                label="Выберите модели для тестирования",
                interactive=True,
                max_choices=2,
            )

            laps_slider = gr.Slider(
                value=2,
                minimum=1,
                maximum=10,
                step=1,
                label="Количество прогонов",
                interactive=True,
            )

            with gr.Column():
                refresh_models_btn = gr.Button("🔄 Загрузить/обновить список моделей", scale=20, size="md")
                start_benchmark_btn = gr.Button(
                    "⏳ Проверка backend...",
                    scale=20,
                    size="md",
                    interactive=False,
                )
                stop_benchmark_btn = gr.Button(
                    "🛑 Остановить тест",
                    scale=20,
                    size="md",
                    variant="stop",
                    interactive=False,
                )

        with gr.Column():
            log_output = gr.Textbox(
                label="Ход выполнения",
                autoscroll=False,
                lines=28,
            )

        gr.Markdown("""### ℹ️ Пояснение к метрикам
            - **Wall Avg(s)** – среднее полное время ответа (что видит пользователь).
            - **Eval Avg(s)** – среднее время генерации без задержек.
            - **TPS Avg** – скорость генерации текста (Tokens Per Second).
            - **σ** — стандартное отклонение (разброс значений).
            """)

        benchmark_kpi_md = gr.Markdown(
            "**Статус:** Ожидание запуска  \n"
            "**Лучший Wall Avg:** —  \n"
            "**Лучший TPS:** —"
        )
        benchmark_backend_status_md = gr.Markdown(
            "**Backend Ollama:** ⏳ Проверка доступности..."
        )

        result_table = gr.DataFrame(
            headers=["Модель", "Тип", "Wall Avg (s)", "Wall σ", "Eval Avg (s)", "Eval σ", "TPS Avg", "TPS σ"],
            row_count=(1, "dynamic"),
            interactive=False,
            label="Результаты замеров",
            show_copy_button=True,
            show_search="filter",
            pinned_columns=1,
        )

        with gr.Row():
            json_view = gr.JSON(label="📄 JSON‑отчёт", visible=True, scale=3)
            with gr.Column(scale=1):
                json_download_btn = gr.DownloadButton(
                    "⬇️ Скачать JSON",
                    value=None,
                    interactive=False,
                    size="sm",
                    variant="secondary",
                )
                csv_download_btn = gr.DownloadButton(
                    "⬇️ Скачать CSV",
                    value=None,
                    interactive=False,
                    size="sm",
                    variant="secondary",
                )

        def normalize_selected_models(current_models: Any) -> list[str]:
            if isinstance(current_models, list):
                return [m for m in current_models if m]
            if current_models:
                return [str(current_models)]
            return []

        def choose_selected_models(current_models: Any, models: list[str]) -> list[str]:
            selected = [m for m in normalize_selected_models(current_models) if m in models][:2]
            if selected:
                return selected

            current_main = LLMName.get()
            if current_main in models:
                return [current_main]
            if models:
                return [models[-1]]
            return []

        def build_backend_status(available: bool, models_count: int = 0, error_text: str | None = None) -> str:
            if available:
                return (
                    f"**Backend Ollama:** ✅ Доступен (`{c.ollama_url}`)  \n"
                    f"**Моделей обнаружено:** `{models_count}`"
                )
            details = (error_text or "не удалось подключиться").strip()
            return (
                f"**Backend Ollama:** 🚫 Недоступен (`{c.ollama_url}`)  \n"
                f"**Окружение:** `{c.environment}`  \n"
                f"**Причина:** `{details}`"
            )

        async def probe_backend(current_models):
            try:
                models_response = await ollama_client.list()
                models = extract_model_names_fn(models_response)
                selected = choose_selected_models(current_models, models)
                return True, models, selected, build_backend_status(True, models_count=len(models)), None
            except Exception as e:
                error_text = str(e).strip() or e.__class__.__name__
                return False, [], [], build_backend_status(False, error_text=error_text), error_text

        async def update_dropdown(current_models):
            backend_ok, models, selected, backend_status, error_text = await probe_backend(current_models)
            if backend_ok:
                start_enabled = bool(selected)
                if not models:
                    log_text = "⚠️ Backend Ollama доступен, но список моделей пуст."
                    start_caption = "🚫 Нет моделей в Ollama"
                elif not start_enabled:
                    log_text = "🟡 Выберите модель для запуска тестирования."
                    start_caption = "🚀 Выберите модель для запуска"
                else:
                    log_text = f"✅ Backend Ollama доступен. Найдено моделей: {len(models)}."
                    start_caption = "🚀 Запустить тестирование"

                return (
                    gr.update(choices=models, value=selected),
                    backend_status,
                    gr.update(interactive=start_enabled, value=start_caption),
                    gr.update(interactive=False, value="🛑 Остановить тест"),
                    log_text,
                )

            local_hint = (
                "🚫 Backend Ollama недоступен. "
                "Для локального запуска переключите `[APP] environment = LOCAL` "
                "или поднимите Ollama по URL из текущего окружения."
            )
            if error_text:
                local_hint = f"{local_hint}\nТехническая причина: {error_text}"

            return (
                gr.update(choices=[], value=[]),
                backend_status,
                gr.update(interactive=False, value="🚫 Backend недоступен"),
                gr.update(interactive=False, value="🛑 Остановить тест"),
                local_hint,
            )

        def table_update_for_benchmark(rows: list[list[Any]]):
            rows = rows or []
            visible_rows = max(1, min(len(rows), 8))
            return gr.update(value=rows, row_count=(visible_rows, "dynamic"))

        def build_benchmark_kpi(results: list[dict[str, Any]], status: str) -> str:
            if not results:
                return (
                    f"**Статус:** {status}  \n"
                    "**Лучший Wall Avg:** —  \n"
                    "**Лучший TPS:** —"
                )

            best_wall = min(results, key=lambda x: float(x.get("wall_avg", float("inf"))))
            best_tps = max(results, key=lambda x: float(x.get("tps_avg", float("-inf"))))
            return (
                f"**Статус:** {status}  \n"
                f"**Лучший Wall Avg:** `{best_wall.get('model')} ({best_wall.get('type')})` = "
                f"`{best_wall.get('wall_avg')}s`  \n"
                f"**Лучший TPS:** `{best_tps.get('model')} ({best_tps.get('type')})` = "
                f"`{best_tps.get('tps_avg')}`"
            )

        def request_stop_benchmark(current_log: str):
            if not benchmark_is_running():
                txt = (current_log or "").rstrip()
                msg = "🟡 Тест не запущен."
                return (
                    f"{txt}\n{msg}" if txt else msg,
                    gr.update(interactive=False, value="🛑 Остановить тест"),
                )

            benchmark_request_stop()
            txt = (current_log or "").rstrip()
            msg = "🛑 Получен запрос на остановку. Завершаю текущий прогон..."
            return (
                f"{txt}\n{msg}" if txt else msg,
                gr.update(interactive=False, value="⏳ Остановка..."),
            )

        async def wrapped_benchmark_stream(models, laps, think: bool):
            backend_status = build_backend_status(False, error_text="проверка не выполнялась")
            if benchmark_is_running():
                yield (
                    "🟡 Тест уже выполняется. Дождитесь завершения или нажмите «Остановить тест».",
                    table_update_for_benchmark([]),
                    [],
                    build_benchmark_kpi([], "Выполняется"),
                    gr.update(),
                    gr.update(value=None, interactive=False),
                    gr.update(value=None, interactive=False),
                    gr.update(interactive=False, value="⏳ Тест выполняется..."),
                    gr.update(interactive=True, value="🛑 Остановить тест"),
                )
                return

            backend_ok, available_models, _, backend_status, backend_error = await probe_backend(models)
            if not backend_ok:
                log_text = (
                    "🚫 Тестирование недоступно: backend Ollama не отвечает. "
                    "Используйте `LOCAL` окружение для локального запуска или проверьте URL сервера."
                )
                if backend_error:
                    log_text = f"{log_text}\nТехническая причина: {backend_error}"
                yield (
                    log_text,
                    table_update_for_benchmark([]),
                    [],
                    build_benchmark_kpi([], "Backend недоступен"),
                    backend_status,
                    gr.update(value=None, interactive=False),
                    gr.update(value=None, interactive=False),
                    gr.update(interactive=False, value="🚫 Backend недоступен"),
                    gr.update(interactive=False, value="🛑 Остановить тест"),
                )
                return

            selected_models = [m for m in normalize_selected_models(models) if m in available_models][:2]
            if not selected_models:
                yield (
                    "🟡 Сначала загрузите и выберите модели для тестирования.",
                    table_update_for_benchmark([]),
                    [],
                    build_benchmark_kpi([], "Ожидание запуска"),
                    backend_status,
                    gr.update(value=None, interactive=False),
                    gr.update(value=None, interactive=False),
                    gr.update(interactive=False, value="🚀 Выберите модель для запуска"),
                    gr.update(interactive=False, value="🛑 Остановить тест"),
                )
                return

            if not benchmark_try_start():
                yield (
                    "🟡 Тест уже выполняется. Дождитесь завершения или нажмите «Остановить тест».",
                    table_update_for_benchmark([]),
                    [],
                    build_benchmark_kpi([], "Выполняется"),
                    gr.update(),
                    gr.update(value=None, interactive=False),
                    gr.update(value=None, interactive=False),
                    gr.update(interactive=False, value="⏳ Тест выполняется..."),
                    gr.update(interactive=True, value="🛑 Остановить тест"),
                )
                return

            try:
                yield (
                    "🚀 Тестирование запущено...",
                    table_update_for_benchmark([]),
                    [],
                    build_benchmark_kpi([], "Выполняется"),
                    backend_status,
                    gr.update(value=None, interactive=False),
                    gr.update(value=None, interactive=False),
                    gr.update(interactive=False, value="⏳ Тест выполняется..."),
                    gr.update(interactive=True, value="🛑 Остановить тест"),
                )

                async for snapshot in gradio_benchmark_stream(
                    selected_models,
                    laps,
                    think,
                    stop_checker=benchmark_stop_requested,
                ):
                    log_text = snapshot.get("log", "")
                    table_rows = snapshot.get("table", []) or []
                    json_data = snapshot.get("results", []) or []
                    done = bool(snapshot.get("done", False))
                    stopped = bool(snapshot.get("stopped", False))
                    stop_requested = benchmark_stop_requested()

                    if done:
                        status = "Остановлено" if stopped else "Завершено"
                    else:
                        status = "Остановка..." if stop_requested else "Выполняется"

                    if done:
                        json_download_update = gr.update(
                            value=snapshot.get("json_path"),
                            interactive=bool(snapshot.get("json_path")),
                        )
                        csv_download_update = gr.update(
                            value=snapshot.get("csv_path"),
                            interactive=bool(snapshot.get("csv_path")),
                        )
                        start_btn_update = gr.update(interactive=True, value="🚀 Запустить тестирование")
                        stop_btn_update = gr.update(interactive=False, value="🛑 Остановить тест")
                    else:
                        json_download_update = gr.update(value=None, interactive=False)
                        csv_download_update = gr.update(value=None, interactive=False)
                        start_btn_update = gr.update(interactive=False, value="⏳ Тест выполняется...")
                        stop_btn_update = gr.update(
                            interactive=not stop_requested,
                            value="⏳ Остановка..." if stop_requested else "🛑 Остановить тест",
                        )

                    yield (
                        log_text,
                        table_update_for_benchmark(table_rows),
                        json_data,
                        build_benchmark_kpi(json_data, status),
                        backend_status,
                        json_download_update,
                        csv_download_update,
                        start_btn_update,
                        stop_btn_update,
                    )
            except Exception as e:
                yield (
                    f"❌ Ошибка тестирования: {e}",
                    table_update_for_benchmark([]),
                    [],
                    build_benchmark_kpi([], "Ошибка"),
                    backend_status,
                    gr.update(value=None, interactive=False),
                    gr.update(value=None, interactive=False),
                    gr.update(interactive=True, value="🚀 Запустить тестирование"),
                    gr.update(interactive=False, value="🛑 Остановить тест"),
                )
            finally:
                benchmark_finish()

        refresh_models_btn.click(
            fn=update_dropdown,
            inputs=[model_selector],
            outputs=[
                model_selector,
                benchmark_backend_status_md,
                start_benchmark_btn,
                stop_benchmark_btn,
                log_output,
            ],
        )

        blocks.load(
            fn=update_dropdown,
            inputs=[model_selector],
            outputs=[
                model_selector,
                benchmark_backend_status_md,
                start_benchmark_btn,
                stop_benchmark_btn,
                log_output,
            ],
        )

        stop_benchmark_btn.click(
            fn=request_stop_benchmark,
            inputs=[log_output],
            outputs=[log_output, stop_benchmark_btn],
            queue=False,
        )

        start_benchmark_btn.click(
            fn=wrapped_benchmark_stream,
            inputs=[model_selector, laps_slider, think_checkbox],
            outputs=[
                log_output,
                result_table,
                json_view,
                benchmark_kpi_md,
                benchmark_backend_status_md,
                json_download_btn,
                csv_download_btn,
                start_benchmark_btn,
                stop_benchmark_btn,
            ],
        )

    return {
        "tab": benchmark_tab,
    }
