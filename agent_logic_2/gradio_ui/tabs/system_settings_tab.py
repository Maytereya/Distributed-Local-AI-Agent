from __future__ import annotations

from typing import Any, Callable

import gradio as gr


def build_system_settings_tab(
    *,
    blocks,
    fn_load_main_model_fn: Callable[[], list[str]],
    reassert_main_model_dropdown_fn,
    fn_assert_main_model_fn: Callable[[str], None],
    write_think_status_fn: Callable[[bool], Any],
    fn_load_options_fn: Callable[[], str | None],
    fn_save_options_fn: Callable[[str], None],
    restart_fn: Callable[[], None],
    fn_load_prompt_with_status_fn: Callable[[str], str],
    fn_save_prompt_fn: Callable[[str, str], None],
    fn_load_prompt_fn: Callable[[str, bool], str],
    fn_load_prompt_with_fallback_silent_fn: Callable[[str, str], str],
    fn_load_options_silent_fn: Callable[[], str],
    read_think_status_fn: Callable[[bool], bool],
    whisper_module,
    prompt_code_mr_rich,
    prompt_code_mr_critic,
    tab_visible: bool = False,
) -> dict[str, Any]:
    with gr.Tab("⚙️ Общие настройки системы", visible=tab_visible) as settings_tab:
        gr.Markdown("""<h3>⚙️ Настройки нейросетей и серверов Ollama/Uvicorn</h3>""")

        with gr.Column():
            with gr.Row():
                with gr.Column():
                    main_model_selector = gr.Dropdown(
                        choices=fn_load_main_model_fn(),
                        multiselect=False,
                        label="Выбор основной LLM",
                        interactive=True,
                        container=True,
                    )
                    think_checkbox = gr.Checkbox(
                        label="Активировать рассуждение",
                        value=False,
                        container=True,
                    )

                    reload_main_model_btn = gr.Button("⬇️ Загрузить доступные модели", size="sm")
                    save_model_btn = gr.Button("💾 Утвердить выбранную модель", size="sm")

                reload_main_model_btn.click(
                    fn=reassert_main_model_dropdown_fn,
                    inputs=[],
                    outputs=main_model_selector,
                )

                save_model_btn.click(
                    fn=fn_assert_main_model_fn,
                    inputs=main_model_selector,
                    outputs=None,
                )

                think_checkbox.change(write_think_status_fn, inputs=[think_checkbox], outputs=None)

                json_ollama_options = gr.Code(
                    label="📄Ollama Options",
                    value="",
                    language="json",
                    visible=True,
                    interactive=True,
                    scale=4,
                )
                with gr.Column():
                    load_options_btn = gr.Button("⬇️ Загрузить текущие опции", scale=20, size="sm")
                    save_options_btn = gr.Button("💾 Сохранить новые опции", scale=20, size="sm")
                    restart_ollama_btn = gr.Button(
                        "🔃 Перезагрузить Ollama",
                        scale=20,
                        size="sm",
                        variant="stop",
                    )

        load_options_btn.click(
            fn=fn_load_options_fn,
            inputs=[],
            outputs=[json_ollama_options],
        )
        save_options_btn.click(fn=fn_save_options_fn, inputs=json_ollama_options, outputs=None)
        restart_ollama_btn.click(fn=restart_fn, outputs=None)

        with gr.Row():
            with gr.Accordion(label="Split Prompt", open=False):
                prompt_code_2 = gr.Code(
                    value="",
                    language=None,
                    label="Split Prompt section",
                    interactive=True,
                    lines=20,
                    scale=4,
                )
                with gr.Row():
                    btn_load_2 = gr.Button("⬇️ Загрузить", size="sm", variant="secondary")
                    btn_save_2 = gr.Button("💾 Сохранить", size="sm", variant="primary")

            btn_load_2.click(lambda: fn_load_prompt_with_status_fn("split_prompt"), [], [prompt_code_2])
            btn_save_2.click(lambda txt: fn_save_prompt_fn("split_prompt", txt), [prompt_code_2])

        with gr.Row():
            with gr.Accordion(label="Classificator Prompt", open=False):
                prompt_code_3 = gr.Code(
                    value="",
                    language=None,
                    label="Classificator Prompt section",
                    interactive=True,
                    lines=10,
                    scale=4,
                )

                with gr.Row():
                    btn_load_3 = gr.Button("⬇️ Загрузить", size="sm", variant="secondary")
                    btn_save_3 = gr.Button("💾 Сохранить", size="sm", variant="primary")

        btn_load_3.click(lambda: fn_load_prompt_with_status_fn("classificator_prompt"), [], [prompt_code_3])
        btn_save_3.click(lambda txt: fn_save_prompt_fn("classificator_prompt", txt), prompt_code_3)

        with gr.Row():
            with gr.Accordion(label="Final Answer Prompt", open=False):
                prompt_code_1 = gr.Code(
                    value="",
                    language=None,
                    label="Final Answering Prompt section",
                    interactive=True,
                    lines=20,
                    scale=4,
                )
                with gr.Row():
                    btn_load_1 = gr.Button("⬇️ Загрузить", size="sm", variant="secondary")
                    btn_save_1 = gr.Button("💾 Сохранить", size="sm", variant="primary")

        btn_load_1.click(lambda: fn_load_prompt_with_status_fn("final_answer"), [], [prompt_code_1])
        btn_save_1.click(lambda txt: fn_save_prompt_fn("final_answer", txt), prompt_code_1)

        with gr.Row():
            with gr.Accordion(label="Examples for Classification", open=False):
                prompt_code_c1 = gr.Code(
                    value="",
                    language=None,
                    label="EXAMPLES section: примеры для классификации",
                    interactive=True,
                    lines=20,
                    scale=4,
                )

                with gr.Row():
                    btn_load_c1 = gr.Button("⬇️ Загрузить", size="sm", variant="secondary")
                    btn_save_c1 = gr.Button("💾 Сохранить", size="sm", variant="primary")

                btn_load_c1.click(lambda: fn_load_prompt_with_status_fn("EXAMPLES"), [], [prompt_code_c1])
                btn_save_c1.click(lambda txt: fn_save_prompt_fn("EXAMPLES", txt), prompt_code_c1)

        with gr.Row():
            with gr.Accordion(label="Markers for Text Chunks", open=False):
                prompt_code_c2 = gr.Code(
                    value="",
                    language=None,
                    label="LABEL_DOC section: образцы маркировки распознанных текстовых сегментов",
                    interactive=True,
                    lines=10,
                    scale=4,
                )

                with gr.Row():
                    btn_load_c2 = gr.Button("⬇️ Загрузить", size="sm", variant="secondary")
                    btn_save_c2 = gr.Button("💾 Сохранить", size="sm", variant="primary")

            btn_load_c2.click(lambda: fn_load_prompt_with_status_fn("LABEL_DOC"), [], [prompt_code_c2])
            btn_save_c2.click(lambda txt: fn_save_prompt_fn("LABEL_DOC", txt), prompt_code_c2)

        with gr.Row():
            with gr.Accordion(label="Text Chunks Layout Priority", open=False):
                prompt_code_c3 = gr.Code(
                    value="",
                    language=None,
                    label="LABEL_PRIORITY section: приоритет расположения текстовых сегментов после распознавания",
                    interactive=True,
                    lines=5,
                    scale=4,
                )

                with gr.Row():
                    btn_load_c3 = gr.Button("⬇️ Загрузить", size="sm", variant="secondary")
                    btn_save_c3 = gr.Button("💾 Сохранить", size="sm", variant="primary")

            btn_load_c3.click(lambda: fn_load_prompt_with_status_fn("LABEL_PRIORITY"), [], [prompt_code_c3])
            btn_save_c3.click(lambda txt: fn_save_prompt_fn("LABEL_PRIORITY", txt), prompt_code_c3)

        with gr.Row():
            with gr.Accordion(label="MODULES: The names of agents's functions", open=False):
                prompt_code_c4 = gr.Code(
                    value="",
                    language=None,
                    label="MODULES section: имена агентских функций, ассоциированных с маркерами, только чтение",
                    interactive=False,
                    lines=5,
                    scale=4,
                )

                with gr.Row():
                    btn_load_c4 = gr.Button("⬇️ Загрузить", size="sm", variant="secondary")
                    btn_save_c4 = gr.Button(
                        "💾 Сохранить",
                        size="sm",
                        variant="primary",
                        interactive=False,
                    )

            btn_load_c4.click(lambda: fn_load_prompt_with_status_fn("MODULES"), [], [prompt_code_c4])
            btn_save_c4.click(lambda txt: fn_save_prompt_fn("MODULES", txt), prompt_code_c4)

        def load_all_prompts():
            return (
                fn_load_prompt_fn("final_answer", False),
                fn_load_prompt_fn("split_prompt", False),
                fn_load_prompt_fn("classificator_prompt", False),
                fn_load_prompt_with_fallback_silent_fn(
                    "mr_renderer_patient_rich",
                    "renderer_patient_rich",
                ),
                fn_load_prompt_with_fallback_silent_fn(
                    "mr_renderer_critic_patient_alignment",
                    "renderer_critic_patient_alignment",
                ),
                fn_load_prompt_fn("EXAMPLES", False),
                fn_load_prompt_fn("LABEL_DOC", False),
                fn_load_prompt_fn("LABEL_PRIORITY", False),
                fn_load_prompt_fn("MODULES", False),
                fn_load_options_silent_fn(),
                read_think_status_fn(False),
            )

        blocks.load(
            fn=load_all_prompts,
            inputs=None,
            outputs=[
                prompt_code_1,
                prompt_code_2,
                prompt_code_3,
                prompt_code_mr_rich,
                prompt_code_mr_critic,
                prompt_code_c1,
                prompt_code_c2,
                prompt_code_c3,
                prompt_code_c4,
                json_ollama_options,
                think_checkbox,
            ],
        )

        with gr.Row():
            with gr.Accordion(
                label="📖 Словарь фамилий, терминов и профессий для системы перевода речи в текст",
                open=False,
            ):
                prompt_code_whisper = gr.Code(
                    value=whisper_module.load_prompt(),
                    language=None,
                    label="Whisper",
                    interactive=True,
                    lines=50,
                    scale=4,
                )

                with gr.Row():
                    load_btn = gr.Button("📥 Загрузить", size="sm", variant="primary")
                    save_btn = gr.Button("📤 Сохранить", size="sm", variant="secondary")
                load_btn.click(fn=whisper_module.load_prompt, inputs=None, outputs=prompt_code_whisper)
                save_btn.click(fn=whisper_module.save_prompt, inputs=prompt_code_whisper)

    return {
        "tab": settings_tab,
        "think_checkbox": think_checkbox,
    }

