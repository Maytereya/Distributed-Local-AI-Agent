from __future__ import annotations

from typing import Any

import gradio as gr
import pandas as pd

from container_managenment import system_data

MONITORED = [
    "bookworm-agent",
    "ollama",
    "whisper-gpu",
    "meili_server",
    "chroma_container",
    "vosk-ru",
    "nginx_proxy",
]

THRESHOLDS = {
    "cpu_warning": 70,
    "cpu_critical": 90,
    "ram_warning_mb": 96_000,
    "ram_critical_mb": 196_000,
    "vram_warning_free_mb": 1500,
    "vram_critical_free_mb": 700,
}


def _update_ollama_container_diag():
    diag = system_data.get_container_diagnostics(
        container_name="ollama",
        log_tail=400,
        important_limit=25,
    )

    def fmt(value: Any, suffix: str = "") -> str:
        if value is None or value == "":
            return "—"
        return f"{value}{suffix}" if suffix else str(value)

    if not diag.get("available"):
        error_text = str(diag.get("error") or "контейнер недоступен")
        hint_text = str(diag.get("hint") or "")
        md = (
            "**Контейнер Ollama:** 🚫 Недоступен  \n"
            f"**Причина:** `{error_text}`"
        )
        if hint_text:
            md += f"  \n**Подсказка:** {hint_text}"

        rows = [
            ["Контейнер", diag.get("container_name", "ollama")],
            ["Статус", "Недоступен"],
            ["CPU, %", "—"],
            ["RAM used, MB", "—"],
            ["RAM limit, MB", "—"],
            ["RAM, %", "—"],
            ["PIDs", "—"],
            ["OOMKilled", "—"],
            ["RestartCount", "—"],
            ["Model total memory (log)", "—"],
            ["System free (log)", "—"],
            ["GPU min free (log)", "—"],
        ]
        messages = f"Ошибка: {error_text}"
        if hint_text:
            messages = f"{messages}\nПодсказка: {hint_text}"
        return md, gr.update(value=rows, row_count=(len(rows), "fixed")), messages

    status = str(diag.get("status") or "unknown")
    if status == "running":
        icon = "✅"
    elif status in {"restarting", "paused", "created"}:
        icon = "🟡"
    else:
        icon = "🚫"

    metrics = diag.get("metrics", {}) or {}
    runtime_memory = diag.get("runtime_memory", {}) or {}

    system_free_log = "—"
    if runtime_memory.get("system_free") and runtime_memory.get("system_total"):
        system_free_log = f"{runtime_memory.get('system_free')} / {runtime_memory.get('system_total')}"
    elif runtime_memory.get("system_free"):
        system_free_log = str(runtime_memory.get("system_free"))

    rows = [
        ["Контейнер", diag.get("container_name", "ollama")],
        ["Container ID", fmt(diag.get("container_id"))],
        ["Статус", status],
        ["Health", fmt(diag.get("health"))],
        ["CPU, %", fmt(metrics.get("cpu_%"))],
        ["RAM used, MB", fmt(metrics.get("ram_used_mb"))],
        ["RAM limit, MB", fmt(metrics.get("ram_limit_mb"))],
        ["RAM, %", fmt(metrics.get("ram_%"))],
        ["PIDs", fmt(metrics.get("pids"))],
        ["OOMKilled", str(bool(diag.get("oom_killed", False)))],
        ["RestartCount", fmt(diag.get("restart_count"))],
        ["Model total memory (log)", fmt(runtime_memory.get("model_total_memory"))],
        ["System free (log)", system_free_log],
        ["GPU min free (log)", fmt(runtime_memory.get("gpu_min_free"))],
    ]

    important_logs = diag.get("important_logs") or []
    important_count = len(important_logs)
    md = (
        f"**Контейнер Ollama:** {icon} `{status}`  \n"
        f"**Важных сообщений (tail):** `{important_count}`"
    )

    if important_logs:
        messages = "\n".join(important_logs[-25:])
    else:
        last_log = str(diag.get("last_log") or "").strip()
        messages = "Критичных сообщений в последних логах не найдено."
        if last_log:
            messages = f"{messages}\nПоследняя строка лога:\n{last_log}"

    diag_error = str(diag.get("error") or "").strip()
    if diag_error:
        messages = f"⚠️ Ошибка метрик: {diag_error}\n\n{messages}"

    log_error = str(diag.get("log_error") or "").strip()
    if log_error:
        messages = f"{messages}\n\n⚠️ Ошибка чтения логов: {log_error}"

    return md, gr.update(value=rows, row_count=(len(rows), "fixed")), messages


def build_monitoring_tab(*, blocks, tab_visible: bool = False) -> dict[str, Any]:
    with gr.Tab("📈 Мониторинг нагрузки", visible=tab_visible) as monitor_tab:
        gr.Markdown("<h3>Показатели использования RAM, SSD, GPU, CPU</h3>")

        history_state = gr.State([])

        summary_md = gr.Markdown()
        warnings_md = gr.Markdown()
        gpu_note = gr.Markdown()

        with gr.Accordion("🩺 Диагностика контейнера Ollama", open=False):
            with gr.Row():
                mon_refresh_ollama_diag_btn = gr.Button(
                    "🩺 Обновить диагностику контейнера Ollama",
                    size="sm",
                    variant="secondary",
                )
                mon_ollama_diag_tick_slider = gr.Slider(
                    label="Автообновление диагностики, сек",
                    minimum=2,
                    maximum=60,
                    step=1,
                    value=10,
                )
            mon_ollama_diag_timer = gr.Timer(10.0)
            mon_ollama_container_diag_md = gr.Markdown(
                "**Контейнер Ollama:** ⏳ Проверка доступности..."
            )
            mon_ollama_container_metrics_df = gr.DataFrame(
                headers=["Параметр", "Значение"],
                value=[["Статус", "Ожидание диагностики"]],
                row_count=(12, "fixed"),
                interactive=False,
                label="Ключевые параметры контейнера Ollama",
            )
            mon_ollama_container_logs_tb = gr.Textbox(
                label="Важные сообщения из логов контейнера (error/warn/oom)",
                lines=8,
                max_lines=12,
                interactive=False,
                autoscroll=False,
            )

        with gr.Row():
            gpu_table = gr.Dataframe(
                label="Видеокарты (Общее использование / Загрузка памяти)",
                interactive=False,
                wrap=False,
            )
        with gr.Row():
            top_ram = gr.Dataframe(
                label="Топ контейнеров по загрузке оперативной памяти",
                interactive=False,
                wrap=True,
            )
            top_cpu = gr.Dataframe(
                label="Топ контейнеров по загрузке процессора",
                interactive=False,
                wrap=True,
            )

        cpu_plot = gr.LinePlot(
            x="tick",
            y="cpu_host_%",
            title="Загрузка центрального процессора (CPU), %",
            height=260,
        )
        ram_plot = gr.LinePlot(
            x="tick",
            y="ram_mb",
            title="Оперативка, RAM (занятая контейнерами), MB",
            height=260,
        )
        vram_plot = gr.LinePlot(
            x="tick",
            y="vram_free_mb_min",
            title="Свободная видеопамять у самой загруженной видеокарты, MB",
            height=260,
        )

        with gr.Accordion("Детальный отчет JSON", open=False):
            details_json = gr.JSON(
                label="Нагрузка на сервер (по всем docker - контейнерам)",
                min_height=400,
                max_height=900,
            )

        t = gr.Timer(1.0)

        with gr.Row():
            btn = gr.Button("Обновить данные", size="sm", variant="secondary")
            tick_slider = gr.Slider(
                label="Частота обновления с шагом 0.5 сек",
                minimum=0.5,
                maximum=2.0,
                step=0.5,
                value=1.0,
            )

        def render(payload, history):
            s = payload.get("summary", {})
            host = payload.get("host", {})
            summary = (
                f"### Сводка\n"
                f"- **CPU (ядра):** {s.get('cpu_cores_used', 0)}\n"
                f"- **CPU (% от сервера):** {s.get('cpu_host_%', 0)}% (логических CPU: {host.get('cpu_count', '?')})\n"
                f"- **RAM (сумма контейнеров):** {s.get('total_ram_used_mb', 0)} MB\n"
                f"- **Docker-сеть IN/OUT:** {s.get('total_net_in_mb', 0)} / {s.get('total_net_out_mb', 0)} MB\n"
                f"- **PIDs (суммарно):** {s.get('total_pids', 0)}\n"
            )

            warns = system_data.compute_alerts(payload, THRESHOLDS)
            warnings_text = (
                "### ⚠️ Предупреждения\n" + "\n".join([f"- {w}" for w in warns])
                if warns
                else "### ✅ Предупреждения\n- Нет превышений порогов."
            )

            top = payload.get("top_consumers", {})
            by_ram = top.get("by_ram", [])
            by_cpu = top.get("by_cpu", [])

            top_ram_df = pd.DataFrame(by_ram) if by_ram else pd.DataFrame(
                columns=["container", "ram_used_mb", "cpu_%"]
            )
            top_cpu_df = pd.DataFrame(by_cpu) if by_cpu else pd.DataFrame(
                columns=["container", "cpu_%", "ram_used_mb"]
            )

            gpu = payload.get("gpu", {})
            if isinstance(gpu, dict) and "gpus" in gpu:
                gpu_df = pd.DataFrame(gpu["gpus"])
                gpu_note_ = ""
            else:
                gpu_df = pd.DataFrame(
                    columns=["gpu", "name", "util_gpu_%", "vram_used_mb", "vram_free_mb", "vram_total_mb"]
                )
                gpu_note_ = gpu.get("note", "GPU данные недоступны.")

            history = system_data.update_history(history, payload, max_points=180)
            df = system_data.history_to_df(history)
            return summary, warnings_text, top_ram_df, top_cpu_df, gpu_df, gpu_note_, df, df, df, payload, history

        def tick(history):
            payload = system_data.make_human_monitor_payload(MONITORED)
            return render(payload, history)

        btn.click(
            fn=tick,
            inputs=[history_state],
            outputs=[
                summary_md,
                warnings_md,
                top_ram,
                top_cpu,
                gpu_table,
                gpu_note,
                cpu_plot,
                ram_plot,
                vram_plot,
                details_json,
                history_state,
            ],
        )

        t.tick(
            fn=tick,
            inputs=[history_state],
            outputs=[
                summary_md,
                warnings_md,
                top_ram,
                top_cpu,
                gpu_table,
                gpu_note,
                cpu_plot,
                ram_plot,
                vram_plot,
                details_json,
                history_state,
            ],
        )

        mon_refresh_ollama_diag_btn.click(
            fn=_update_ollama_container_diag,
            inputs=None,
            outputs=[
                mon_ollama_container_diag_md,
                mon_ollama_container_metrics_df,
                mon_ollama_container_logs_tb,
            ],
            queue=False,
        )

        mon_ollama_diag_tick_slider.change(
            fn=lambda value: float(value),
            inputs=[mon_ollama_diag_tick_slider],
            outputs=[mon_ollama_diag_timer],
            queue=False,
        )
        mon_ollama_diag_timer.tick(
            fn=_update_ollama_container_diag,
            inputs=None,
            outputs=[
                mon_ollama_container_diag_md,
                mon_ollama_container_metrics_df,
                mon_ollama_container_logs_tb,
            ],
            queue=False,
        )
        blocks.load(
            fn=_update_ollama_container_diag,
            inputs=None,
            outputs=[
                mon_ollama_container_diag_md,
                mon_ollama_container_metrics_df,
                mon_ollama_container_logs_tb,
            ],
        )

        first_change = gr.State(True)

        def tick_slider_change(value: float, is_first: bool):
            if is_first:
                return value, False
            gr.Info(f"Частота обновления данных: {value}", duration=3.0, title="Системный монитор")
            return value, False

        tick_slider.change(
            fn=tick_slider_change,
            inputs=[tick_slider, first_change],
            outputs=[t, first_change],
        )

    return {
        "tab": monitor_tab,
    }
