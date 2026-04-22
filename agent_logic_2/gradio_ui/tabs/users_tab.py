from __future__ import annotations

from typing import Any, Callable

import gradio as gr


def build_users_tab(
    *,
    blocks,
    list_users_fn: Callable[..., tuple[list[list[str]], str]],
    create_user_fn: Callable[..., tuple[list[list[str]], str, str, str, str]],
    tab_visible: bool = False,
) -> dict[str, Any]:
    with gr.Tab("👥 Пользователи", visible=tab_visible) as users_tab:
        gr.Markdown("<h3>Управление базовыми пользователями (доступ только к вкладкам 1 и 2)</h3>")

        with gr.Row():
            user_name_tb = gr.Textbox(
                label="Имя пользователя",
                placeholder="Например: Иван Петров",
            )
            user_login_tb = gr.Textbox(
                label="Логин",
                placeholder="Например: ivan.petrov",
            )
            user_password_tb = gr.Textbox(
                label="Пароль",
                type="password",
                placeholder="Минимум 4 символа",
            )

        with gr.Row():
            users_create_btn = gr.Button("➕ Создать пользователя", variant="primary")
            users_refresh_btn = gr.Button("🔄 Обновить список", variant="secondary")
            gr.Button(
                "🚪 Выйти из аккаунта",
                variant="secondary",
                size="sm",
                link="logout",
            )

        users_df = gr.Dataframe(
            headers=["Имя", "Логин", "Активен", "Создан (UTC)"],
            value=[],
            interactive=False,
            wrap=True,
            label="Зарегистрированные пользователи",
        )
        users_status_md = gr.Markdown("Пользователи не загружены")

        users_create_btn.click(
            fn=create_user_fn,
            inputs=[user_name_tb, user_login_tb, user_password_tb],
            outputs=[users_df, users_status_md, user_name_tb, user_login_tb, user_password_tb],
            queue=False,
        )
        users_refresh_btn.click(
            fn=list_users_fn,
            inputs=None,
            outputs=[users_df, users_status_md],
            queue=False,
        )
        blocks.load(
            fn=list_users_fn,
            inputs=None,
            outputs=[users_df, users_status_md],
            queue=False,
        )

    return {
        "tab": users_tab,
        "users_df": users_df,
        "users_status_md": users_status_md,
    }
