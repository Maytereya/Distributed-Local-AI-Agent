from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, TypedDict

import gradio as gr


class AssistantTabRefs(TypedDict):
    tab: Any
    radio_type_of_search: Any
    value_n_results_slider: Any
    thresholdvalue_slider: Any
    value_k_slider: Any
    meili_search_indexes_dropdown: Any
    chroma_search_collection_dropdown: Any


_EXPORT_DIR = Path(__file__).resolve().parents[3] / "app_data" / "chat_exports"


def _build_chat_export_payload(
    *,
    chat_history: list[dict[str, Any]] | None,
    mode: str,
    session_id: Any,
) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    transcript_lines: list[str] = []
    for item in chat_history or []:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip() or "assistant"
        content = item.get("content")
        if isinstance(content, (str, int, float, bool)) or content is None:
            normalized_content: Any = "" if content is None else str(content)
        else:
            normalized_content = content
        messages.append(
            {
                "role": role,
                "content": normalized_content,
            }
        )
        content_text = normalized_content if isinstance(normalized_content, str) else json.dumps(
            normalized_content,
            ensure_ascii=False,
            default=str,
        )
        transcript_lines.append(f"{role}: {content_text}")

    return {
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "mode": str(mode or "").strip(),
        "session_id": str(session_id or "").strip(),
        "messages": messages,
        "transcript": "\n\n".join(transcript_lines),
    }


def export_chat_dialog(
    chat_history: list[dict[str, Any]] | None,
    mode: str,
    session_id: Any,
) -> str:
    payload = _build_chat_export_payload(
        chat_history=chat_history,
        mode=mode,
        session_id=session_id,
    )
    _EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_slug = str(mode or "dialog").strip().lower().replace(" ", "_")
    file_path = _EXPORT_DIR / f"{mode_slug}_dialog_{timestamp}.json"
    file_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return str(file_path)


def build_assistant_tab(
    universal_echo_fn: Callable[..., Any],
    ws_transcribe_fn: Callable[[Any], Awaitable[str]],
    gr_existed_indexes_fn: Callable[[], list[str]],
    gr_existed_collections_fn: Callable[[], list[str]],
    index_label: str,
    collection_label: str,
) -> AssistantTabRefs:
    with gr.Tab("📖 AI - ассистент") as assistant_tab:
        chatbot = gr.Chatbot(
            type="messages",
            autoscroll=False,
            placeholder="<strong>🧠 ИИ - помощник</strong><br>Знает всю информацию о врачах и услугах клиники Наука",
            height=620,
            max_height=1000,
            label="Моя Наука",
            elem_id="assistant-chatbot",
        )

        textbox = gr.Textbox(
            lines=2,
            max_lines=12,
            placeholder="Напишите свой вопрос",
            submit_btn=True,
            stop_btn=True,
            container=True,
            autoscroll=False,
            autofocus=False,
            min_width=0,
            scale=70,
            html_attributes=gr.InputHTMLAttributes(autocorrect="off", spellcheck=True),
        )
        messenger_session_state = gr.State(value=None)

        radio_type_of_search = gr.Radio(
            ["Free-talk-Ai", "Call-Center-Ai", "Messengers-Ai", "gigachat", "meilisearch", "vectorstore", "db"],
            label="Способы поиска в базе знаний, выбор нейросети или канала связи (для администраторов или клиентов)",
            value="Call-Center-Ai",
            container=True,
            render=False,
            info="Выберите алгоритм/канал",
        )

        meili_search_indexes_dropdown = gr.Dropdown(
            choices=gr_existed_indexes_fn(),
            label=index_label,
            info="Выберите Индекс для поиска информации",
            interactive=False,
            render=False,
        )
        chroma_search_collection_dropdown = gr.Dropdown(
            choices=gr_existed_collections_fn(),
            label=collection_label,
            info="Выберите Коллекцию для поиска информации",
            interactive=False,
            allow_custom_value=False,
            render=False,
        )

        value_n_results_slider = gr.Slider(
            value=5,
            minimum=1,
            maximum=20,
            step=1,
            label="Количество документов, включенных в выдачу",
            info="Только в режиме vectorstore",
            interactive=False,
            render=False,
        )

        thresholdvalue_slider = gr.Slider(
            value=0.005,
            minimum=0.0025,
            maximum=0.02,
            step=0.0025,
            label="Порог косинусной фильтрации",
            info="Только в режиме vectorstore."
            "Чем выше значение, тем больше текстовых фрагментов с меньшей "
            "релевантностью появится в выдаче",
            interactive=False,
            render=False,
        )

        value_k_slider = gr.Slider(
            value=2,
            minimum=1,
            maximum=20,
            step=1,
            label="Количество документов, включенных в выдачу",
            info="Только в режимах db и meilisearch",
            interactive=False,
            render=False,
        )
        settings_accordion = gr.Accordion("⚙️ Выбор режима диалога", open=False, visible=True, render=False)

        gr.ChatInterface(
            fn=universal_echo_fn,
            type="messages",
            chatbot=chatbot,
            textbox=textbox,
            additional_inputs_accordion=settings_accordion,
            additional_inputs=[
                radio_type_of_search,
                thresholdvalue_slider,
                value_n_results_slider,
                value_k_slider,
                chroma_search_collection_dropdown,
                meili_search_indexes_dropdown,
                messenger_session_state,
            ],
            additional_outputs=[messenger_session_state],
            show_progress="full",
        )

        with gr.Row():
            export_button = gr.Button("Экспорт диалога", variant="secondary")
            export_file = gr.File(
                label="Файл экспорта",
                visible=False,
                interactive=False,
            )

        def export_dialog_for_download(chat_history, mode, current_session):
            path = export_chat_dialog(chat_history, mode, current_session)
            return gr.update(value=path, visible=True)

        export_button.click(
            fn=export_dialog_for_download,
            inputs=[chatbot, radio_type_of_search, messenger_session_state],
            outputs=[export_file],
            queue=False,
        )

        # ====== ЗАХВАТ АУДИО И РАСШИФРОВКА ======

        with gr.Row():
            mic = gr.Audio(
                sources=["microphone"],
                type="numpy",
                streaming=False,
                label="Микрофон",
                interactive=True,
                format="wav",
                min_width=150,
                show_download_button=False,
                show_share_button=False,
                editable=False,
                show_label=True,
                visible=True,
            )
        gr.Markdown(
            "`🎙️ Голосовой ввод: запишите вопрос, дождитесь распознавания и при необходимости отредактируйте текст`")

        async def ws_transcribe_to(audio):
            """
            audio -> текст от Whisper
            :return: текст в textbox
            """
            if audio is None:
                return gr.update(), gr.update()
            try:
                text = await ws_transcribe_fn(audio)  # функция уровнем ниже
            except Exception as e:
                gr.Error(title="Ошибка распознавания речи", message=str(e))
                return gr.update(), gr.update(value=None)
            text = str(text or "").strip()
            if not text:
                gr.Warning("Не удалось распознать речь. Попробуйте записать ещё раз.", title="Предупреждение")
                return gr.update(), gr.update(value=None)
            return gr.update(value=text), gr.update(value=None)

        # Автотранскрипция по окончании записи и очистка по клику на крестик
        mic.change(
            fn=ws_transcribe_to,
            inputs=mic,
            outputs=[textbox, mic],
        )

        def reset_session_on_mode_change(mode: str, current_session: Any):
            if mode in {"Messengers-Ai", "Free-talk-Ai"}:
                return current_session
            return None

        radio_type_of_search.change(
            fn=reset_session_on_mode_change,
            inputs=[radio_type_of_search, messenger_session_state],
            outputs=[messenger_session_state],
            queue=False,
        )

    return {
        "tab": assistant_tab,
        "radio_type_of_search": radio_type_of_search,
        "value_n_results_slider": value_n_results_slider,
        "thresholdvalue_slider": thresholdvalue_slider,
        "value_k_slider": value_k_slider,
        "meili_search_indexes_dropdown": meili_search_indexes_dropdown,
        "chroma_search_collection_dropdown": chroma_search_collection_dropdown,
    }
