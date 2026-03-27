from __future__ import annotations

import html
import json
import os
from typing import Any

import gradio as gr

import agent_logic_2.ollama_settings as ollama_settings
from agent_logic_1 import meilisearch_client as meilisearch
from agent_logic_2.direct_upload_meili_tab import build_blocks
from agent_logic_2.gradio_ui.handlers.chat_modes import universal_echo
from agent_logic_2.gradio_ui.handlers.kb_ops import (
    existed_docs_in_selected_collection,
    existed_docs_in_selected_index,
    gr_add_to_collection,
    gr_add_to_index_universal,
    gr_create_collection,
    gr_create_index,
    gr_existed_collections,
    gr_existed_indexes,
    gr_remove_collection,
    gr_remove_index,
    gr_rm_doc_from_index,
    radio_search_engine_change,
    radio_sliders_change,
    radio_type_of_upl_file_change,
    update_docs_in_chroma_collection,
    update_docs_in_meili_index,
    validate_id_live,
)
from agent_logic_2.gradio_ui.shared.auth import check_auth, get_reserved_logins, get_user_role
from agent_logic_2.gradio_ui.shared import user_store
from agent_logic_2.gradio_ui.shared.bootstrap import bootstrap_files
from agent_logic_2.gradio_ui.shared.constants import (
    COLLECTIONS_IN_CHROMA,
    INDEXES_IN_MEILI,
    OG_HEAD,
    STATIC_DIR,
    custom_css,
)
from agent_logic_2.gradio_ui.tabs.assistant_tab import build_assistant_tab
from agent_logic_2.gradio_ui.tabs.benchmark_tab import build_benchmark_tab
from agent_logic_2.gradio_ui.tabs.documents_tab import (
    build_documents_management_section,
    build_documents_tab_scaffold,
    wire_documents_tab_logic,
)
from agent_logic_2.gradio_ui.tabs.messenger_settings_tab import build_messenger_settings_tab
from agent_logic_2.gradio_ui.tabs.monitoring_tab import build_monitoring_tab
from agent_logic_2.gradio_ui.tabs.system_settings_tab import build_system_settings_tab
from agent_logic_2.gradio_ui.tabs.users_tab import build_users_tab
from agent_logic_2.id_validation import is_valid_id, sanitize_id
from agent_logic_2.ollama_settings import LLMName
from agent_logic_2.prompts import load_prompt, write_prompt
from container_managenment import restart_container
from messengers_router import topic_registry as mr_topic_registry
from messengers_router.prompt_registry import load_prompt_text
from whisper import whisper_dict as w
from whisper.wisper_ws_client import ws_transcribe

# -------------------
# БЕЗОПАСНОСТЬ
# -------------------
# Пока не срабатывает.
os.environ["USER_AGENT"] = "NEIRY.Agent/1.0"


def main():
    # Инициализация при загрузке приложения options и model name
    bootstrap_files()  # Прогружаем файлы промптов, опций, названия модели, think - mode.
    ollama_settings.init_options()
    ollama_settings.init_thinking()

    # Allow serving local /static files via /gradio_api/file=...
    gr.set_static_paths(paths=[STATIC_DIR])

    custom_footer = """
    <div style="
        width: 100%;
        text-align: center;
        padding: 12px 0;
        font-size: 14px;
        color: #888;
        /*border-top: 1px solid #3333;*/
        /*margin-top: 10px;*/
    ">
        © 2025 <b>neiry.ai llc.</b>
    </div>
    """
    with (gr.Blocks(css=custom_css, title="Neiry.ai", head=OG_HEAD) as blocks):
        with gr.Row(elem_id="logo-row"):
            with gr.Column(scale=0, min_width=180):
                gr.HTML(
                    "<div id='logo-bar'>"
                    "<img id='brand-logo' src='/gradio_api/file=static/logo.png' alt='Логотип'>"
                    "</div>"
                )
            with gr.Column(scale=1, min_width=0, elem_id="top-user-col"):
                top_user_identity = gr.HTML(
                    "<div id='top-user-box'>"
                    "<span class='top-user-icon'>👤</span>"
                    "<span class='top-user-name'>—</span>"
                    "<a class='top-user-logout' href='logout'>выйти</a>"
                    "</div>"
                )

        with gr.Tabs(elem_id="main-tabs"):
            # --------------------------------------------------
            # Вкладка 1 — основной интерфейс
            # --------------------------------------------------

            assistant_tab_refs = build_assistant_tab(
                universal_echo_fn=universal_echo,
                ws_transcribe_fn=ws_transcribe,
                gr_existed_indexes_fn=gr_existed_indexes,
                gr_existed_collections_fn=gr_existed_collections,
                index_label=INDEXES_IN_MEILI,
                collection_label=COLLECTIONS_IN_CHROMA,
            )
            assistant_tab = assistant_tab_refs["tab"]

            # --------------------------------------------------
            # Вкладка 2 - Upload PDF to MEILI or CHROMA DB
            # --------------------------------------------------

            with gr.Tab("\U0001F4E4 Документы") as documents_tab:
                docs_refs = build_documents_tab_scaffold(
                    gr_existed_collections_fn=gr_existed_collections,
                    gr_existed_indexes_fn=gr_existed_indexes,
                    existed_docs_in_selected_index_fn=existed_docs_in_selected_index,
                    collections_label=COLLECTIONS_IN_CHROMA,
                    indexes_label=INDEXES_IN_MEILI,
                )
                docs_management_refs = build_documents_management_section(
                    gr_existed_indexes_fn=gr_existed_indexes,
                    gr_existed_collections_fn=gr_existed_collections,
                    existed_docs_in_selected_index_fn=existed_docs_in_selected_index,
                    existed_docs_in_selected_collection_fn=existed_docs_in_selected_collection,
                    indexes_label=INDEXES_IN_MEILI,
                    collections_label=COLLECTIONS_IN_CHROMA,
                )
                wire_documents_tab_logic(
                    docs_refs=docs_refs,
                    management_refs=docs_management_refs,
                    assistant_tab_refs=assistant_tab_refs,
                    build_blocks_fn=build_blocks,
                    meilisearch_module=meilisearch,
                    sanitize_id_fn=sanitize_id,
                    is_valid_id_fn=is_valid_id,
                    validate_id_live_fn=validate_id_live,
                    update_docs_in_meili_index_fn=update_docs_in_meili_index,
                    update_docs_in_chroma_collection_fn=update_docs_in_chroma_collection,
                    radio_sliders_change_fn=radio_sliders_change,
                    radio_search_engine_change_fn=radio_search_engine_change,
                    radio_type_of_upl_file_change_fn=radio_type_of_upl_file_change,
                    gr_create_collection_fn=gr_create_collection,
                    gr_remove_collection_fn=gr_remove_collection,
                    gr_add_to_collection_fn=gr_add_to_collection,
                    gr_remove_index_fn=gr_remove_index,
                    gr_create_index_fn=gr_create_index,
                    gr_add_to_index_universal_fn=gr_add_to_index_universal,
                    gr_rm_doc_from_index_fn=gr_rm_doc_from_index,
                )

            # -------- FUNCTIONS SECTION ------------
            # ---- Функции настроек системы ---------
            # ---------------------------------------

            def fn_load_options() -> str | None:
                """
                Загружает и сериализует настройки Ollama в JSON с отступами.
                :return: JSON-форматированная строка
                    ASCII отключено, отступы есть.
                :rtype: Str
                """
                try:
                    data, info = ollama_settings.load_ollama_options(True)
                    gr.Success(title="Успешно", message=info, duration=3)
                    return data
                except Exception as e:
                    gr.Error(title="Ошибка загрузки options", message=str(e))
                    return None

            def fn_load_options_silent() -> str:

                """
                Загружает и сериализует настройки Ollama в JSON с отступами.
                :return: JSON-форматированная строка
                    ASCII отключено, отступы есть.
                :rtype: Str
                """
                try:
                    res = ollama_settings.load_ollama_options(False)
                    return res[0] if isinstance(res, tuple) else res
                except Exception as e:
                    gr.Error(title="Ошибка загрузки options", message=str(e))
                    return ""

            def fn_save_options(text: str) -> None:
                """
                Сохраняет измененные настройки Ollama в файл JSON.
                """
                try:
                    data = json.loads(text)
                    msg = ollama_settings.write_options(data)
                    gr.Success(title="Успешно",
                               duration=3,
                               message=msg,
                               )

                    return None
                except json.JSONDecodeError as e:
                    gr.Error(f"❌ Ошибка сохранения OPTIONS: {e}")
                    return None

            def fn_load_prompt(name: str, inform: bool = False) -> str:
                return load_prompt(name, inform)

            def fn_load_prompt_with_status(name: str) -> str:

                text, msg = load_prompt(name, inform=True)
                gr.Info(title="Загружен успешно",
                        duration=3,
                        message=msg,
                        )
                return text

            def fn_load_prompt_with_fallback(name: str, fallback_key: str) -> str:
                text, msg = load_prompt(name, inform=True)
                if not isinstance(text, str):
                    text = ""
                if text.strip():
                    gr.Info(title="Загружен успешно", duration=3, message=msg)
                    return text
                try:
                    fallback_text = load_prompt_text(fallback_key)
                    gr.Info(
                        title="Загружен fallback",
                        duration=3,
                        message=f"{msg}; использован встроенный prompt {fallback_key}",
                    )
                    return fallback_text
                except Exception as e:
                    gr.Error(title="Ошибка загрузки prompt", message=str(e))
                    return text

            def fn_load_prompt_with_fallback_silent(name: str, fallback_key: str) -> str:
                text = load_prompt(name, inform=False)
                if isinstance(text, str) and text.strip():
                    return text
                try:
                    return load_prompt_text(fallback_key)
                except Exception:
                    return text if isinstance(text, str) else ""

            def fn_save_prompt(name: str, text: str) -> None:

                try:
                    info = write_prompt(name, text)
                    gr.Success(title="Сохранен успешно",
                               duration=3,
                               message=info,
                               )
                    return None
                except Exception as e:
                    gr.Error(f"❌ Ошибка сохранения на уровне интерфейса: {e}")
                    return None

            # ------------ ASSERT(ACCEPT/VALIDATE/CHOOSE) MAIN LLM ---------------

            def _extract_ollama_model_names(models_response: Any) -> list[str]:
                """
                Унификация формата ответа ollama.list():
                поддержка dict, pydantic-объектов и смешанных элементов.
                """
                if isinstance(models_response, dict):
                    raw_models = models_response.get("models", []) or []
                else:
                    raw_models = getattr(models_response, "models", []) or []

                names: list[str] = []
                for item in raw_models:
                    name = None
                    if isinstance(item, str):
                        name = item
                    elif isinstance(item, dict):
                        name = item.get("model") or item.get("name")
                    else:
                        name = getattr(item, "model", None) or getattr(item, "name", None)
                        if not name:
                            try:
                                name = item["model"]  # type: ignore[index]
                            except Exception:
                                try:
                                    name = item["name"]  # type: ignore[index]
                                except Exception:
                                    name = None

                    if name:
                        norm = str(name).strip()
                        if norm:
                            names.append(norm)

                return sorted(set(names))

            def fn_load_main_model() -> list[str]:
                return [LLMName.get()]

            def fn_assert_main_model(name: str, ) -> None:

                try:
                    gr.Success(title="Выбор сохранен",
                               duration=3,
                               message=f"LLM {name} установлена",
                               )
                    LLMName.set(name)
                    return None

                except Exception as e:
                    gr.Error(f"❌ Ошибка сохранения на уровне интерфейса: {e}")
                    return None

            async def reassert_main_model_dropdown(only_list: bool = False) -> Any:
                models_response = await LLMName.list_all_models()
                models = _extract_ollama_model_names(models_response)
                if only_list:
                    return models
                return gr.update(choices=models, value=models[-1] if models else [])

            # ------------Секция управления контейнерами------------

            def restart() -> None:
                info = restart_container.restart_ollama_container()
                gr.Info(title="Ollama Server",
                        duration=15,
                        message=info,
                        )
                return None

            # ------------------------------------------------------

            messenger_tab_refs = build_messenger_settings_tab(
                blocks=blocks,
                fn_load_prompt_with_fallback_fn=fn_load_prompt_with_fallback,
                fn_save_prompt_fn=fn_save_prompt,
                mr_topic_registry_module=mr_topic_registry,
            )
            messenger_tab = messenger_tab_refs["tab"]
            prompt_code_mr_rich = messenger_tab_refs["prompt_code_mr_rich"]
            prompt_code_mr_critic = messenger_tab_refs["prompt_code_mr_critic"]

            system_settings_refs = build_system_settings_tab(
                blocks=blocks,
                fn_load_main_model_fn=fn_load_main_model,
                reassert_main_model_dropdown_fn=reassert_main_model_dropdown,
                fn_assert_main_model_fn=fn_assert_main_model,
                write_think_status_fn=ollama_settings.write_think_status,
                fn_load_options_fn=fn_load_options,
                fn_save_options_fn=fn_save_options,
                restart_fn=restart,
                fn_load_prompt_with_status_fn=fn_load_prompt_with_status,
                fn_save_prompt_fn=fn_save_prompt,
                fn_load_prompt_fn=fn_load_prompt,
                fn_load_prompt_with_fallback_silent_fn=fn_load_prompt_with_fallback_silent,
                fn_load_options_silent_fn=fn_load_options_silent,
                read_think_status_fn=ollama_settings.read_think_status,
                whisper_module=w,
                prompt_code_mr_rich=prompt_code_mr_rich,
                prompt_code_mr_critic=prompt_code_mr_critic,
                tab_visible=False,
            )
            settings_tab = system_settings_refs["tab"]
            think_checkbox = system_settings_refs["think_checkbox"]

            benchmark_tab_refs = build_benchmark_tab(
                blocks=blocks,
                fn_load_main_model_fn=fn_load_main_model,
                think_checkbox=think_checkbox,
                extract_model_names_fn=_extract_ollama_model_names,
                tab_visible=False,
            )
            benchmark_tab = benchmark_tab_refs["tab"]

            monitoring_tab_refs = build_monitoring_tab(
                blocks=blocks,
                tab_visible=False,
            )
            monitor_tab = monitoring_tab_refs["tab"]

            def _users_rows() -> list[list[str]]:
                users = user_store.list_users()
                return [
                    [
                        str(user.get("name") or ""),
                        str(user.get("login") or ""),
                        "Да" if bool(user.get("active", True)) else "Нет",
                        str(user.get("created_at") or ""),
                    ]
                    for user in users
                ]

            def _is_admin_request(request: gr.Request | None) -> bool:
                username = getattr(request, "username", None) if request else None
                return get_user_role(username) == "admin"

            def fn_users_list(request: gr.Request | None = None) -> tuple[list[list[str]], str]:
                if not _is_admin_request(request):
                    return [], "Доступ к списку пользователей только для администратора"
                try:
                    rows = _users_rows()
                    return rows, f"Пользователей: {len(rows)}"
                except Exception as e:
                    msg = f"Ошибка чтения users_db: {e}"
                    gr.Error(title="Пользователи", message=msg)
                    return [], msg

            def fn_users_create(
                display_name: str,
                login: str,
                password: str,
                request: gr.Request | None = None,
            ) -> tuple[list[list[str]], str, str, str, str]:
                if not _is_admin_request(request):
                    msg = "Создавать пользователей может только администратор"
                    gr.Error(title="Недостаточно прав", message=msg)
                    return [], msg, display_name, login, ""
                try:
                    created = user_store.create_basic_user(
                        name=display_name,
                        login=login,
                        password=password,
                        deny_logins=get_reserved_logins(),
                    )
                    gr.Success(
                        title="Пользователь создан",
                        message=f"Создан пользователь: {created['login']}",
                        duration=3,
                    )
                    rows = _users_rows()
                    return rows, f"Пользователей: {len(rows)}", "", "", ""
                except Exception as e:
                    msg = str(e)
                    gr.Error(title="Ошибка создания пользователя", message=msg)
                    rows = _users_rows()
                    status = f"Пользователей: {len(rows)}"
                    return rows, f"{status}. Ошибка: {msg}", display_name, login, ""

            users_tab_refs = build_users_tab(
                blocks=blocks,
                list_users_fn=fn_users_list,
                create_user_fn=fn_users_create,
                tab_visible=False,
            )
            users_tab = users_tab_refs["tab"]

            def render_top_user_identity(request: gr.Request | None = None) -> str:
                username = str(getattr(request, "username", None) or "").strip()
                safe_username = html.escape(username if username else "user")
                return (
                    "<div id='top-user-box'>"
                    "<span class='top-user-icon'>👤</span>"
                    f"<span class='top-user-name'>{safe_username}</span>"
                    "<a class='top-user-logout' href='logout'>выйти</a>"
                    "</div>"
                )

            def apply_tab_visibility_by_role(request: gr.Request | None = None):
                username = getattr(request, "username", None) if request else None
                is_admin = get_user_role(username) == "admin"
                return (
                    gr.update(visible=True),  # assistant_tab
                    gr.update(visible=True),  # documents_tab
                    gr.update(visible=is_admin),  # users_tab
                    gr.update(visible=is_admin),  # messenger_tab
                    gr.update(visible=is_admin),  # settings_tab
                    gr.update(visible=is_admin),  # benchmark_tab
                    gr.update(visible=is_admin),  # monitor_tab
                )

            blocks.load(
                fn=apply_tab_visibility_by_role,
                inputs=None,
                outputs=[
                    assistant_tab,
                    documents_tab,
                    users_tab,
                    messenger_tab,
                    settings_tab,
                    benchmark_tab,
                    monitor_tab,
                ],
                queue=False,
            )
            blocks.load(
                fn=render_top_user_identity,
                inputs=None,
                outputs=[top_user_identity],
                queue=False,
            )

        # -------------------------
        # Footer html реализация
        # -------------------------
        gr.HTML(custom_footer)
    blocks.queue(
        default_concurrency_limit=8,
        max_size=64)

    blocks.launch(
        server_name="0.0.0.0",
        server_port=7860,
        auth=check_auth,
        show_api=False,
        allowed_paths=[str(STATIC_DIR)],
        favicon_path=str(STATIC_DIR / "logo.png")
    )


if __name__ == "__main__":
    main()
