from __future__ import annotations

import json
from functools import partial
from typing import Any, Callable

import gradio as gr
import yaml


def build_messenger_settings_tab(
    *,
    blocks,
    fn_load_prompt_with_fallback_fn: Callable[[str, str], str],
    fn_save_prompt_fn: Callable[[str, str], None],
    mr_topic_registry_module,
) -> dict[str, Any]:
    def _mr_label_choices() -> list[str]:
        try:
            labels = mr_topic_registry_module.list_valid_labels()
            normalized = [str(x).strip().upper() for x in labels if str(x).strip()]
            if normalized:
                return normalized
        except Exception:
            pass
        return [
            "APPOINTMENT",
            "TEST_ASSIST",
            "TEST_RESULT",
            "DOCTOR_INFO",
            "DOCTOR_SCHEDULE",
            "PRICE",
            "ADDRESS",
            "PREPARE",
            "NEWS",
            "COMPLAINT",
            "URGENT",
            "MEDICAL_ADVICE",
            "OTHER",
        ]

    def _mr_registry_summary(data: dict[str, Any]) -> str:
        topics = data.get("topics")
        if not isinstance(topics, list):
            return "Всего тем: 0 | Активных: 0\nПо типам: -"
        total = 0
        enabled = 0
        labels: dict[str, int] = {}
        for item in topics:
            if not isinstance(item, dict):
                continue
            total += 1
            if bool(item.get("enabled", True)):
                enabled += 1
            lbl = str(item.get("label") or "OTHER").strip().upper()
            labels[lbl] = labels.get(lbl, 0) + 1
        chunks = [f"{k}={v}" for k, v in sorted(labels.items())]
        labels_text = ", ".join(chunks) if chunks else "-"
        return f"Всего тем: {total} | Активных: {enabled}\nПо типам: {labels_text}"

    def _mr_runtime_scope_report(data: dict[str, Any]) -> str:
        topics = data.get("topics")
        if not isinstance(topics, list):
            topics = []

        total = 0
        disabled = 0
        no_match_rules: list[str] = []
        no_sources: list[str] = []
        unsupported_kinds: list[str] = []
        unknown_meili_indexes: list[str] = []
        other_without_meili: list[str] = []

        for topic in topics:
            if not isinstance(topic, dict):
                continue
            total += 1
            tid = str(topic.get("topic_id") or f"topic_{total}").strip() or f"topic_{total}"
            if not bool(topic.get("enabled", True)):
                disabled += 1

            match = topic.get("match") if isinstance(topic.get("match"), dict) else {}
            any_keywords = match.get("any_keywords") if isinstance(match.get("any_keywords"), list) else []
            all_keywords = match.get("all_keywords") if isinstance(match.get("all_keywords"), list) else []
            regex_rules = match.get("regex") if isinstance(match.get("regex"), list) else []
            if not any_keywords and not all_keywords and not regex_rules:
                no_match_rules.append(tid)

            route = topic.get("route") if isinstance(topic.get("route"), dict) else {}
            sources = route.get("sources") if isinstance(route.get("sources"), list) else []
            if not sources:
                no_sources.append(tid)

            has_meili_source = False
            for source in sources:
                if not isinstance(source, dict):
                    unsupported_kinds.append(f"{tid}:<invalid>")
                    continue
                kind = str(source.get("kind") or "").strip().lower()
                index = str(source.get("index") or "").strip().lower()
                if kind != "meili":
                    unsupported_kinds.append(f"{tid}:{kind or '?'}")
                    continue
                has_meili_source = True
                if index and index not in {"main_index", "news"}:
                    unknown_meili_indexes.append(f"{tid}:{index}")

            label = str(topic.get("label") or "OTHER").strip().upper()
            if label == "OTHER" and not has_meili_source:
                other_without_meili.append(tid)

        lines = [
            "Runtime (текущий код):",
            "Используются: topic_id, enabled, priority, label, match.any_keywords/all_keywords/regex/exclude_keywords.",
            "Частично: route.sources (kind=meili; index='news' -> news_info, иначе -> main_index_info).",
            "Не используются роутером сейчас: route.strategy, context.*, fallback.*, marks.*, defaults.*.",
            "",
            (
                f"Проверка реестра: тем={total}, отключено={disabled}, "
                f"без match-правил={len(no_match_rules)}, без route.sources={len(no_sources)}."
            ),
        ]
        if unsupported_kinds:
            sample = ", ".join(unsupported_kinds[:4])
            suffix = " ..." if len(unsupported_kinds) > 4 else ""
            lines.append(f"Игнорируемые route.sources.kind: {len(unsupported_kinds)} ({sample}{suffix})")
        if unknown_meili_indexes:
            sample = ", ".join(unknown_meili_indexes[:4])
            suffix = " ..." if len(unknown_meili_indexes) > 4 else ""
            lines.append(
                f"Неизвестные meili index: {len(unknown_meili_indexes)} "
                f"(сейчас уйдут в main_index_info) ({sample}{suffix})"
            )
        if other_without_meili:
            sample = ", ".join(other_without_meili[:4])
            suffix = " ..." if len(other_without_meili) > 4 else ""
            lines.append(
                f"Темы OTHER без meili-source: {len(other_without_meili)} "
                f"(планировщик не построит tool-шаги) ({sample}{suffix})"
            )
        return "\n".join(lines)

    def _mr_snapshot(selected_topic_id: str | None = None) -> tuple[str, Any, str, str, str]:
        data = mr_topic_registry_module.load_registry(force_reload=True)
        topics = mr_topic_registry_module.list_topics(enabled_only=False)
        topic_ids = [str(t.get("topic_id") or "").strip() for t in topics if str(t.get("topic_id") or "").strip()]
        if selected_topic_id and selected_topic_id in topic_ids:
            selected_id = selected_topic_id
        else:
            selected_id = topic_ids[0] if topic_ids else None

        selected_topic = mr_topic_registry_module.get_topic(selected_id) if selected_id else None
        topic_json = json.dumps(selected_topic or {}, ensure_ascii=False, indent=2)
        yaml_text = mr_topic_registry_module.load_registry_text()
        summary = _mr_registry_summary(data)
        runtime_scope = _mr_runtime_scope_report(data)
        return (
            yaml_text,
            gr.update(choices=topic_ids, value=selected_id),
            topic_json,
            summary,
            runtime_scope,
        )

    def fn_mr_refresh(selected_topic_id: str | None = None):
        try:
            return _mr_snapshot(selected_topic_id)
        except Exception as e:
            gr.Error(f"❌ Ошибка загрузки topic registry: {e}")
            return (
                "",
                gr.update(),
                "{}",
                "Всего тем: 0 | Активных: 0\nПо типам: -",
                _mr_runtime_scope_report({}),
            )

    def fn_mr_select_topic(topic_id: str | None):
        try:
            return _mr_snapshot(topic_id)
        except Exception as e:
            gr.Error(f"❌ Ошибка выбора topic: {e}")
            return (
                "",
                gr.update(),
                "{}",
                "Всего тем: 0 | Активных: 0\nПо типам: -",
                _mr_runtime_scope_report({}),
            )

    def fn_mr_save_yaml(yaml_text: str, selected_topic_id: str | None):
        try:
            yaml.safe_load(yaml_text) if str(yaml_text or "").strip() else {}
            mr_topic_registry_module.save_registry_text(yaml_text)
            gr.Success(title="Успешно", message="topic_registry.yaml сохранен", duration=3)
            return _mr_snapshot(selected_topic_id)
        except Exception as e:
            gr.Error(f"❌ Ошибка сохранения YAML: {e}")
            return fn_mr_refresh(selected_topic_id)

    def fn_mr_save_topic(selected_topic_id: str | None, topic_json_text: str):
        try:
            payload = json.loads(topic_json_text or "{}")
            if not isinstance(payload, dict):
                raise ValueError("Topic JSON должен быть объектом")
            tid_from_payload = str(payload.get("topic_id") or "").strip()
            tid = tid_from_payload or str(selected_topic_id or "").strip()
            if not tid:
                raise ValueError("Не указан topic_id")
            payload["topic_id"] = tid
            saved = mr_topic_registry_module.upsert_topic(payload)
            gr.Success(title="Успешно", message=f"Topic {saved.get('topic_id')} сохранен", duration=3)
            return _mr_snapshot(str(saved.get("topic_id") or ""))
        except Exception as e:
            gr.Error(f"❌ Ошибка сохранения topic: {e}")
            return fn_mr_refresh(selected_topic_id)

    def fn_mr_create_topic(topic_id: str, label: str, priority: float):
        try:
            tid = str(topic_id or "").strip()
            if not tid:
                raise ValueError("Введите topic_id")
            saved = mr_topic_registry_module.create_topic(
                topic_id=tid,
                label=str(label or "OTHER"),
                priority=int(priority),
            )
            gr.Success(title="Успешно", message=f"Topic {saved.get('topic_id')} создан", duration=3)
            return _mr_snapshot(str(saved.get("topic_id") or ""))
        except Exception as e:
            gr.Error(f"❌ Ошибка создания topic: {e}")
            return fn_mr_refresh(None)

    def fn_mr_delete_topic(selected_topic_id: str | None):
        try:
            tid = str(selected_topic_id or "").strip()
            if not tid:
                raise ValueError("Выберите topic для удаления")
            ok = mr_topic_registry_module.delete_topic(tid)
            if not ok:
                raise ValueError(f"Topic {tid} не найден")
            gr.Success(title="Успешно", message=f"Topic {tid} удален", duration=3)
            return _mr_snapshot(None)
        except Exception as e:
            gr.Error(f"❌ Ошибка удаления topic: {e}")
            return fn_mr_refresh(selected_topic_id)

    def fn_mr_toggle_topic(selected_topic_id: str | None, enabled: bool):
        try:
            tid = str(selected_topic_id or "").strip()
            if not tid:
                raise ValueError("Выберите topic")
            saved = mr_topic_registry_module.set_topic_enabled(tid, enabled=bool(enabled))
            if saved is None:
                raise ValueError(f"Topic {tid} не найден")
            state_text = "включен" if bool(saved.get("enabled", True)) else "выключен"
            gr.Success(title="Успешно", message=f"Topic {tid} {state_text}", duration=3)
            return _mr_snapshot(tid)
        except Exception as e:
            gr.Error(f"❌ Ошибка изменения статуса topic: {e}")
            return fn_mr_refresh(selected_topic_id)

    with gr.Tab("🧭 Настройки мессенджера", visible=False) as messenger_tab:
        gr.Markdown("""<h3>🧭 Настройки роутера входящих сообщений</h3>""")
        gr.Markdown(
            "Здесь управляется пакет `messengers_router`: промпты рендера и реестр тем маршрутизации."
        )

        with gr.Accordion("🧠 Промпты пакета messengers_router", open=False):
            gr.Markdown(
                "Изменяйте только если понимаете влияние на стиль ответа и self-check. "
                "Промпты сохраняются в `app_data/prompts`."
            )

            with gr.Row():
                with gr.Accordion(label="MR Rich Generator Prompt", open=False):
                    prompt_code_mr_rich = gr.Code(
                        value="",
                        language=None,
                        label="messengers_router: mr_renderer_patient_rich",
                        interactive=True,
                        lines=18,
                        scale=4,
                    )
                    with gr.Row():
                        btn_load_mr_rich = gr.Button("⬇️ Загрузить", size="sm", variant="secondary")
                        btn_save_mr_rich = gr.Button("💾 Сохранить", size="sm", variant="primary")

            btn_load_mr_rich.click(
                lambda: fn_load_prompt_with_fallback_fn(
                    "mr_renderer_patient_rich",
                    "renderer_patient_rich",
                ),
                [],
                [prompt_code_mr_rich],
            )
            btn_save_mr_rich.click(
                lambda txt: fn_save_prompt_fn("mr_renderer_patient_rich", txt),
                prompt_code_mr_rich,
            )

            with gr.Row():
                with gr.Accordion(label="MR Critic Prompt (JSON)", open=False):
                    prompt_code_mr_critic = gr.Code(
                        value="",
                        language=None,
                        label="messengers_router: mr_renderer_critic_patient_alignment",
                        interactive=True,
                        lines=18,
                        scale=4,
                    )
                    with gr.Row():
                        btn_load_mr_critic = gr.Button("⬇️ Загрузить", size="sm", variant="secondary")
                        btn_save_mr_critic = gr.Button("💾 Сохранить", size="sm", variant="primary")

            btn_load_mr_critic.click(
                lambda: fn_load_prompt_with_fallback_fn(
                    "mr_renderer_critic_patient_alignment",
                    "renderer_critic_patient_alignment",
                ),
                [],
                [prompt_code_mr_critic],
            )
            btn_save_mr_critic.click(
                lambda txt: fn_save_prompt_fn("mr_renderer_critic_patient_alignment", txt),
                prompt_code_mr_critic,
            )

            with gr.Row():
                with gr.Accordion(label="MR Messenger Final Answer Prompt", open=False):
                    prompt_code_mr_final_answer = gr.Code(
                        value="",
                        language=None,
                        label="messengers_router: mr_messenger_final_answer",
                        interactive=True,
                        lines=18,
                        scale=4,
                    )
                    with gr.Row():
                        btn_load_mr_final_answer = gr.Button("⬇️ Загрузить", size="sm", variant="secondary")
                        btn_save_mr_final_answer = gr.Button("💾 Сохранить", size="sm", variant="primary")

            btn_load_mr_final_answer.click(
                lambda: fn_load_prompt_with_fallback_fn(
                    "mr_messenger_final_answer",
                    "messenger_final_answer",
                ),
                [],
                [prompt_code_mr_final_answer],
            )
            btn_save_mr_final_answer.click(
                lambda txt: fn_save_prompt_fn("mr_messenger_final_answer", txt),
                prompt_code_mr_final_answer,
            )

        gr.Markdown(
            "Рабочий режим: выберите тему, правьте JSON и сохраняйте. "
            "YAML ниже нужен только для массового редактирования."
        )

        with gr.Row():
            mr_refresh_btn = gr.Button("🔄 Обновить данные из файла", size="sm")

        mr_registry_summary_box = gr.Textbox(
            label="Краткая сводка реестра (всего/активно/по label)",
            value="",
            interactive=False,
            lines=3,
        )
        mr_runtime_scope_box = gr.Textbox(
            label="Актуальность настроек для текущего runtime",
            value="",
            interactive=False,
            lines=8,
        )

        with gr.Row():
            mr_topic_selector = gr.Dropdown(
                choices=[],
                value=None,
                label="Тема для редактирования (topic_id)",
                allow_custom_value=False,
                interactive=True,
                scale=60,
            )
            mr_topic_enable_btn = gr.Button("🟢 Включить тему", size="sm", scale=20)
            mr_topic_disable_btn = gr.Button("⚪ Выключить тему", size="sm", scale=20)
            mr_topic_delete_btn = gr.Button("⛔ Удалить", size="sm", variant="stop", scale=20)

        gr.Markdown(
            "🟢 Включить: тема участвует в матчинге. "
            "⚪ Выключить: тема остается в файле, но не используется. "
            "⛔ Удалить: удаляет тему из реестра."
        )

        mr_topic_json_code = gr.Code(
            label="Карточка темы (JSON): редактирование выбранного topic",
            language="json",
            value="{}",
            interactive=True,
            lines=18,
        )

        with gr.Row():
            mr_topic_save_btn = gr.Button("💾 Сохранить изменения темы", size="sm", variant="primary")

        with gr.Row():
            mr_new_topic_id = gr.Textbox(
                label="ID новой темы (topic_id)",
                placeholder="например: prepare_ultrasound",
                value="",
                scale=45,
            )
            mr_new_topic_label = gr.Dropdown(
                choices=_mr_label_choices(),
                value="OTHER",
                label="Целевой label",
                scale=20,
            )
            mr_new_topic_priority = gr.Number(
                label="Приоритет",
                value=100,
                precision=0,
                scale=15,
            )
            mr_create_topic_btn = gr.Button("✅ Создать тему", size="sm", scale=20)

        with gr.Accordion("🛠 Режим эксперта: прямое редактирование topic_registry.yaml", open=False):
            gr.Markdown(
                "Используйте при массовых правках. Перед сохранением YAML проверяется на синтаксис."
            )
            mr_registry_yaml_code = gr.Code(
                label="topic_registry.yaml (экспертный режим)",
                language="yaml",
                value="",
                interactive=True,
                lines=16,
            )
            with gr.Row():
                mr_save_yaml_btn = gr.Button("💾 Сохранить YAML", size="sm", variant="primary")

        mr_refresh_btn.click(
            fn=fn_mr_refresh,
            inputs=[mr_topic_selector],
            outputs=[
                mr_registry_yaml_code,
                mr_topic_selector,
                mr_topic_json_code,
                mr_registry_summary_box,
                mr_runtime_scope_box,
            ],
        )
        mr_save_yaml_btn.click(
            fn=fn_mr_save_yaml,
            inputs=[mr_registry_yaml_code, mr_topic_selector],
            outputs=[
                mr_registry_yaml_code,
                mr_topic_selector,
                mr_topic_json_code,
                mr_registry_summary_box,
                mr_runtime_scope_box,
            ],
        )
        mr_topic_selector.change(
            fn=fn_mr_select_topic,
            inputs=[mr_topic_selector],
            outputs=[
                mr_registry_yaml_code,
                mr_topic_selector,
                mr_topic_json_code,
                mr_registry_summary_box,
                mr_runtime_scope_box,
            ],
        )
        mr_topic_save_btn.click(
            fn=fn_mr_save_topic,
            inputs=[mr_topic_selector, mr_topic_json_code],
            outputs=[
                mr_registry_yaml_code,
                mr_topic_selector,
                mr_topic_json_code,
                mr_registry_summary_box,
                mr_runtime_scope_box,
            ],
        )
        mr_create_topic_btn.click(
            fn=fn_mr_create_topic,
            inputs=[mr_new_topic_id, mr_new_topic_label, mr_new_topic_priority],
            outputs=[
                mr_registry_yaml_code,
                mr_topic_selector,
                mr_topic_json_code,
                mr_registry_summary_box,
                mr_runtime_scope_box,
            ],
        )
        mr_topic_delete_btn.click(
            fn=fn_mr_delete_topic,
            inputs=[mr_topic_selector],
            outputs=[
                mr_registry_yaml_code,
                mr_topic_selector,
                mr_topic_json_code,
                mr_registry_summary_box,
                mr_runtime_scope_box,
            ],
        )
        mr_topic_enable_btn.click(
            fn=partial(fn_mr_toggle_topic, enabled=True),
            inputs=[mr_topic_selector],
            outputs=[
                mr_registry_yaml_code,
                mr_topic_selector,
                mr_topic_json_code,
                mr_registry_summary_box,
                mr_runtime_scope_box,
            ],
        )
        mr_topic_disable_btn.click(
            fn=partial(fn_mr_toggle_topic, enabled=False),
            inputs=[mr_topic_selector],
            outputs=[
                mr_registry_yaml_code,
                mr_topic_selector,
                mr_topic_json_code,
                mr_registry_summary_box,
                mr_runtime_scope_box,
            ],
        )

        blocks.load(
            fn=fn_mr_refresh,
            inputs=None,
            outputs=[
                mr_registry_yaml_code,
                mr_topic_selector,
                mr_topic_json_code,
                mr_registry_summary_box,
                mr_runtime_scope_box,
            ],
        )
        blocks.load(
            fn=lambda: fn_load_prompt_with_fallback_fn(
                "mr_messenger_final_answer",
                "messenger_final_answer",
            ),
            inputs=None,
            outputs=[prompt_code_mr_final_answer],
        )

    return {
        "tab": messenger_tab,
        "prompt_code_mr_rich": prompt_code_mr_rich,
        "prompt_code_mr_critic": prompt_code_mr_critic,
        "prompt_code_mr_final_answer": prompt_code_mr_final_answer,
    }
