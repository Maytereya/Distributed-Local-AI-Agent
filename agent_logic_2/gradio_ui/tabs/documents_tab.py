from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

import gradio as gr
from gradio_pdf import PDF

from agent_logic_2.direct_upload_meili_tab import TABLE_HEADERS


def build_documents_tab_scaffold(
    *,
    gr_existed_collections_fn,
    gr_existed_indexes_fn,
    existed_docs_in_selected_index_fn,
    collections_label: str,
    indexes_label: str,
) -> dict[str, Any]:
    gr.Markdown(
        "**Шаг 1:** выберите базу и индекс/коллекцию. "
        "**Шаг 2:** загрузите PDF/JSON или заполните форму конструктора. "
        "**Шаг 3:** обновите список ниже и при необходимости удалите документ."
    )
    with gr.Accordion(label="Загрузка готовых документов в базы знаний Meilisearch и ChromaDB", open=True):
        with gr.Row():
            radio_type_of_db = gr.Radio(
                ["ChromaDB", "Meilisearch"],
                label="Тип базы данных",
                value="Meilisearch",
                container=True,
            )

            radio_type_of_upl_data = gr.Radio(
                ["PDF", "JSON"],
                label="Тип документа",
                value="PDF",
                container=True,
                visible=True,
            )

            upload_collections_dropdown = gr.Dropdown(
                choices=gr_existed_collections_fn(),
                allow_custom_value=True,
                filterable=True,
                label=collections_label,
                visible=False,
            )

            upload_indices_dropdown = gr.Dropdown(
                choices=gr_existed_indexes_fn(),
                allow_custom_value=True,
                filterable=True,
                label=indexes_label,
                visible=True,
            )

            # Кнопки для работы с коллекциями или индексами
            with gr.Column():
                add_collection_button = gr.Button(
                    "✅ Добавить коллекцию",
                    visible=False,
                    size="sm",
                    variant="primary",
                    scale=10,
                )
                rm_collection_button = gr.Button(
                    "⛔ Удалить коллекцию",
                    visible=False,
                    size="sm",
                    variant="stop",
                    scale=10,
                )

                add_index_button = gr.Button(
                    "✅ Добавить индекс",
                    visible=True,
                    size="sm",
                    variant="primary",
                    scale=10,
                )
                rm_index_button = gr.Button(
                    "⛔ Удалить индекс",
                    visible=True,
                    size="sm",
                    variant="stop",
                    scale=10,
                )

        with gr.Row():
            pdf = PDF(label="Загрузить PDF", interactive=True, scale=80)

            # Загрузка JSON (по умолчанию невидим)
            json_file = gr.File(
                label="Загрузить JSON",
                visible=False,
                scale=80,
                file_types=[".json"],
                type="filepath",
            )

            with gr.Column():
                add_to_collection_button = gr.Button(
                    "Добавить в коллекцию",
                    visible=False,
                    size="sm",
                    variant="primary",
                )
                add_to_index_button = gr.Button(
                    "Добавить в индекс",
                    visible=True,
                    size="sm",
                    variant="primary",
                )

    # --------------------------------------------------------------------------------
    # Секция оформления страницы - конструктора документа
    # --------------------------------------------------------------------------------
    meta_state = gr.State(value=None)  # dict: {"doc_id": str, "index": str, "blocks": list[dict]}
    orig_doc_id_state = gr.State(value=None)  # для работы переименования в редакторе документов

    default_empty_table = [["", ""], ["", ""], ["", ""]]
    table_state = gr.State(value=default_empty_table)

    with gr.Accordion(label="Форма для добавления информации в базу знаний Meilisearch", open=True):
        with gr.Column():
            with gr.Row():
                io_radio = gr.Radio(
                    [("Создать", "create"), ("Редактировать", "change")],
                    value="create",
                    scale=20,
                    label="Выберите действие",
                )

                index_dropdown = gr.Dropdown(
                    choices=gr_existed_indexes_fn(),
                    info="main_index: скрипты, news: акции/новости",
                    label="Выберите индекс Meilisearch",
                    interactive=True,
                    scale=30,
                )

                id_input = gr.Textbox(
                    label="ID документа (латиница/цифры/нижнее подчеркивание, 3–60)",
                    info="Введите уникальный для добавления нового документа",
                    value="",
                    visible=True,
                    placeholder="например: price_list_2025 или izmeneniya_grafika_priema",
                    scale=50,
                )

                id_select = gr.Dropdown(
                    label="ID документа",
                    info="Выберите для редактирования документа",
                    choices=existed_docs_in_selected_index_fn(index_dropdown.value, "ID"),
                    allow_custom_value=False,
                    visible=False,
                    interactive=True,
                    scale=50,
                )

                with gr.Column():
                    normalize_id_btn = gr.Button("🧹 Нормализовать ID", size="sm", variant="secondary", visible=True)
                    generate_id_from_title_btn = gr.Button(
                        "🪄 ID из заголовка",
                        size="sm",
                        variant="primary",
                        visible=True,
                    )
                    vanish_screen_btn = gr.Button("🧹 Очистить ввод", size="sm", variant="stop", visible=True)
                    load_doc_btn = gr.Button("⬇️ Загрузить по ID", size="sm", variant="primary", visible=False)
                    save_doc_btn_direct = gr.Button(
                        "💾 Сохранить (обновить по ID)",
                        size="sm",
                        variant="primary",
                        visible=False,
                    )

            # ---------------------------------------------
            # Компилятор новостей и диапазонов действия новостей
            # ---------------------------------------------
            far_future_dt = datetime(2099, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
            type_for_index = {"main_index": "static", "news": "news"}

            with gr.Row():
                doc_type_radio = gr.Radio(
                    choices=[("Скрипты", "static"), ("Новость / Акция", "news")],
                    value="static",
                    label="Тип документа",
                    scale=30,
                    interactive=False,
                    visible=True,
                )
                title_input = gr.Textbox(label="Заголовок, title *", scale=70)

            with gr.Row(visible=False) as news_dates_row:
                valid_from_dp = gr.DateTime(
                    label="Действует с (UTC)",
                    type="datetime",
                    include_time=False,
                    timezone="UTC",
                )
                valid_to_dp = gr.DateTime(
                    label="Действует по (UTC)",
                    type="datetime",
                    include_time=False,
                    timezone="UTC",
                )
                permanent_cb = gr.Checkbox(label="Бессрочно", value=False, visible=False)

            def on_doc_type_change(t):
                is_news = t == "news"
                return (
                    gr.update(visible=is_news),  # news_dates_row
                    gr.update(visible=is_news),  # permanent_cb
                )

            doc_type_radio.change(on_doc_type_change, inputs=[doc_type_radio], outputs=[news_dates_row, permanent_cb])

            def on_index_change(idx: str):
                t = type_for_index.get(idx, "static")
                is_news = t == "news"
                return (
                    gr.update(value=t),  # doc_type_radio
                    gr.update(visible=is_news),  # news_dates_row
                    gr.update(visible=is_news),  # permanent_cb
                )

            index_dropdown.change(
                on_index_change,
                inputs=[index_dropdown],
                outputs=[doc_type_radio, news_dates_row, permanent_cb],
            )

            def to_utc(dt: datetime | None) -> datetime | None:
                if dt is None:
                    return None
                if dt.tzinfo is None:
                    return dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc)

            def to_ts(dt: datetime) -> int:
                return int(dt.timestamp())

            content_input = gr.Textbox(label="Основной текст, content *", lines=20, max_lines=80)
            keywords_input = gr.Textbox(label="Ключевые слова, keywords (через запятую)")
            show_table_cb = gr.Checkbox(label="Добавить таблицу", value=False)

            table_df = gr.Dataframe(
                label="Таблица (необязательно)",
                visible=False,
                headers=TABLE_HEADERS,
                datatype="str",
                row_count=(3, "dynamic"),
                col_count=(len(TABLE_HEADERS), "fixed"),
                type="array",
                value=[["", ""], ["", ""], ["", ""]],
                interactive=True,
                show_fullscreen_button=True,
            )

            def on_show_table_change(use_table: bool):
                return gr.update(visible=bool(use_table))

            show_table_cb.change(on_show_table_change, inputs=[show_table_cb], outputs=[table_df])

            def _passthrough_table(t):
                # t — это list[list]; чисто прокидываем в State
                return t

            table_df.change(_passthrough_table, inputs=[table_df], outputs=[table_state])
            preview_button = gr.Button("Предпросмотр блоков")
            preview_json = gr.JSON(label="Предпросмотр JSON", visible=False)
            save_button = gr.Button("Сохранить и отправить в индекс")

    return {
        "radio_type_of_db": radio_type_of_db,
        "radio_type_of_upl_data": radio_type_of_upl_data,
        "upload_collections_dropdown": upload_collections_dropdown,
        "upload_indices_dropdown": upload_indices_dropdown,
        "add_collection_button": add_collection_button,
        "rm_collection_button": rm_collection_button,
        "add_index_button": add_index_button,
        "rm_index_button": rm_index_button,
        "pdf": pdf,
        "json_file": json_file,
        "add_to_collection_button": add_to_collection_button,
        "add_to_index_button": add_to_index_button,
        "meta_state": meta_state,
        "orig_doc_id_state": orig_doc_id_state,
        "default_empty_table": default_empty_table,
        "table_state": table_state,
        "io_radio": io_radio,
        "index_dropdown": index_dropdown,
        "id_input": id_input,
        "id_select": id_select,
        "normalize_id_btn": normalize_id_btn,
        "generate_id_from_title_btn": generate_id_from_title_btn,
        "vanish_screen_btn": vanish_screen_btn,
        "load_doc_btn": load_doc_btn,
        "save_doc_btn_direct": save_doc_btn_direct,
        "far_future_dt": far_future_dt,
        "type_for_index": type_for_index,
        "doc_type_radio": doc_type_radio,
        "title_input": title_input,
        "news_dates_row": news_dates_row,
        "valid_from_dp": valid_from_dp,
        "valid_to_dp": valid_to_dp,
        "permanent_cb": permanent_cb,
        "to_utc": to_utc,
        "to_ts": to_ts,
        "content_input": content_input,
        "keywords_input": keywords_input,
        "show_table_cb": show_table_cb,
        "table_df": table_df,
        "preview_button": preview_button,
        "preview_json": preview_json,
        "save_button": save_button,
    }


def build_documents_management_section(
    *,
    gr_existed_indexes_fn,
    gr_existed_collections_fn,
    existed_docs_in_selected_index_fn,
    existed_docs_in_selected_collection_fn,
    indexes_label: str,
    collections_label: str,
) -> dict[str, Any]:
    now = datetime.now()
    formatted_full = now.strftime("%A, %d %B %Y, %H:%M")

    with gr.Accordion(label="База знаний Meilisearch (просмотр и удаление)", open=False):
        with gr.Row():
            meili_ind_for_cont_dropdown = gr.Dropdown(
                choices=gr_existed_indexes_fn(),
                filterable=True,
                interactive=True,
                label=indexes_label,
                visible=True,
                scale=4,
            )

            meili_selected_doc_id_box = gr.Textbox(
                value="",
                label="ID для удаления (клик по строке таблицы)",
                placeholder="Кликните по строке ниже, чтобы выбрать документ",
                interactive=False,
                visible=True,
                scale=4,
            )

            with gr.Column():
                refresh_data_btn = gr.Button(
                    "🔄 Обновить",
                    size="md",
                    visible=True,
                    interactive=True,
                    variant="primary",
                    scale=2,
                )
                rm_doc_from_index_button = gr.Button(
                    "⛔ Удалить из Индекса",
                    size="sm",
                    visible=True,
                    interactive=True,
                    variant="stop",
                    scale=2,
                )

        meili_indices_table = gr.DataFrame(
            value=existed_docs_in_selected_index_fn(meili_ind_for_cont_dropdown.value, "All"),
            label=f"Содержание выбранного Индекса на сегодня: {formatted_full}",
            headers=["ID документа", "Заголовок", "Фрагмент"],
            row_count=(1, "dynamic"),
            col_count=(3, "fixed"),
            datatype="str",
            interactive=False,
            max_height=420,
            show_row_numbers=True,
            show_search="filter",
            max_chars=160,
            pinned_columns=1,
        )

    with gr.Accordion(label="База знаний Chroma DB (просмотр и удаление)", open=False):
        with gr.Row():
            chroma_coll_for_cont_dropdown = gr.Dropdown(
                choices=gr_existed_collections_fn(),
                filterable=True,
                label=collections_label,
                interactive=True,
                visible=True,
                scale=4,
            )
            chroma_collection_content = gr.Dropdown(
                choices=existed_docs_in_selected_collection_fn(chroma_coll_for_cont_dropdown.value),
                allow_custom_value=False,
                label="Выбрать документ для удаления",
                visible=True,
                interactive=True,
                scale=4,
            )

            rm_doc_from_collection_button = gr.Button(
                "Удалить из Коллекции",
                size="sm",
                visible=True,
                interactive=False,
                variant="stop",
                scale=2,
            )

        chroma_collection_table = gr.DataFrame(
            value=existed_docs_in_selected_collection_fn(chroma_coll_for_cont_dropdown.value),
            label="Содержание выбранной Коллекции",
            headers=["Имя файла и страница"],
            row_count=(15, "dynamic"),
            col_count=(1, "fixed"),
            datatype="str",
            interactive=False,
        )

    return {
        "meili_ind_for_cont_dropdown": meili_ind_for_cont_dropdown,
        "meili_selected_doc_id_box": meili_selected_doc_id_box,
        "refresh_data_btn": refresh_data_btn,
        "rm_doc_from_index_button": rm_doc_from_index_button,
        "meili_indices_table": meili_indices_table,
        "chroma_coll_for_cont_dropdown": chroma_coll_for_cont_dropdown,
        "chroma_collection_content": chroma_collection_content,
        "rm_doc_from_collection_button": rm_doc_from_collection_button,
        "chroma_collection_table": chroma_collection_table,
    }


def wire_documents_tab_logic(
    *,
    docs_refs: dict[str, Any],
    management_refs: dict[str, Any],
    assistant_tab_refs: dict[str, Any],
    build_blocks_fn,
    meilisearch_module: Any,
    sanitize_id_fn,
    is_valid_id_fn,
    validate_id_live_fn,
    update_docs_in_meili_index_fn,
    update_docs_in_chroma_collection_fn,
    radio_sliders_change_fn,
    radio_search_engine_change_fn,
    radio_type_of_upl_file_change_fn,
    gr_create_collection_fn,
    gr_remove_collection_fn,
    gr_add_to_collection_fn,
    gr_remove_index_fn,
    gr_create_index_fn,
    gr_add_to_index_universal_fn,
    gr_rm_doc_from_index_fn,
) -> None:
    radio_type_of_db = docs_refs["radio_type_of_db"]
    radio_type_of_upl_data = docs_refs["radio_type_of_upl_data"]
    upload_collections_dropdown = docs_refs["upload_collections_dropdown"]
    upload_indices_dropdown = docs_refs["upload_indices_dropdown"]
    add_collection_button = docs_refs["add_collection_button"]
    rm_collection_button = docs_refs["rm_collection_button"]
    add_index_button = docs_refs["add_index_button"]
    rm_index_button = docs_refs["rm_index_button"]
    pdf = docs_refs["pdf"]
    json_file = docs_refs["json_file"]
    add_to_collection_button = docs_refs["add_to_collection_button"]
    add_to_index_button = docs_refs["add_to_index_button"]
    meta_state = docs_refs["meta_state"]
    orig_doc_id_state = docs_refs["orig_doc_id_state"]
    default_empty_table = docs_refs["default_empty_table"]
    table_state = docs_refs["table_state"]
    io_radio = docs_refs["io_radio"]
    index_dropdown = docs_refs["index_dropdown"]
    id_input = docs_refs["id_input"]
    id_select = docs_refs["id_select"]
    normalize_id_btn = docs_refs["normalize_id_btn"]
    generate_id_from_title_btn = docs_refs["generate_id_from_title_btn"]
    vanish_screen_btn = docs_refs["vanish_screen_btn"]
    load_doc_btn = docs_refs["load_doc_btn"]
    save_doc_btn_direct = docs_refs["save_doc_btn_direct"]
    far_future_dt = docs_refs["far_future_dt"]
    type_for_index = docs_refs["type_for_index"]
    doc_type_radio = docs_refs["doc_type_radio"]
    title_input = docs_refs["title_input"]
    news_dates_row = docs_refs["news_dates_row"]
    valid_from_dp = docs_refs["valid_from_dp"]
    valid_to_dp = docs_refs["valid_to_dp"]
    permanent_cb = docs_refs["permanent_cb"]
    to_utc = docs_refs["to_utc"]
    to_ts = docs_refs["to_ts"]
    content_input = docs_refs["content_input"]
    keywords_input = docs_refs["keywords_input"]
    show_table_cb = docs_refs["show_table_cb"]
    table_df = docs_refs["table_df"]
    preview_button = docs_refs["preview_button"]
    preview_json = docs_refs["preview_json"]
    save_button = docs_refs["save_button"]

    meili_ind_for_cont_dropdown = management_refs["meili_ind_for_cont_dropdown"]
    meili_selected_doc_id_box = management_refs["meili_selected_doc_id_box"]
    refresh_data_btn = management_refs["refresh_data_btn"]
    rm_doc_from_index_button = management_refs["rm_doc_from_index_button"]
    meili_indices_table = management_refs["meili_indices_table"]
    chroma_coll_for_cont_dropdown = management_refs["chroma_coll_for_cont_dropdown"]
    chroma_collection_content = management_refs["chroma_collection_content"]
    chroma_collection_table = management_refs["chroma_collection_table"]

    radio_type_of_search = assistant_tab_refs["radio_type_of_search"]
    value_n_results_slider = assistant_tab_refs["value_n_results_slider"]
    thresholdvalue_slider = assistant_tab_refs["thresholdvalue_slider"]
    value_k_slider = assistant_tab_refs["value_k_slider"]
    meili_search_indexes_dropdown = assistant_tab_refs["meili_search_indexes_dropdown"]
    chroma_search_collection_dropdown = assistant_tab_refs["chroma_search_collection_dropdown"]

    def fn_preview_json(
        current_doc_id,
        title,
        content,
        keywords,
        table,
        selected_index,
        doc_type,
        valid_from,
        valid_to,
        is_permanent,
        split: Literal["on", "off"] = "off",
    ):
        expected_type = type_for_index.get(selected_index, "static")
        if doc_type != expected_type:
            doc_type = expected_type
            gr.Warning("Тип документа приведён к выбранному индексу.", title="Предупреждение")

        if not selected_index:
            gr.Warning("Не выбран индекс", title="Предупреждение")
            return gr.update(visible=False), None

        try:
            doc_id, blocks = build_blocks_fn(current_doc_id, title, content, keywords, table, split=split)
        except ValueError as e:
            gr.Error(f"{e}", title="Ошибка!")
            return gr.update(visible=False), None

        vf_utc = vt_utc = None
        is_perm = False
        if doc_type == "news":
            is_perm = bool(is_permanent)
            if is_perm:
                vf_utc = to_utc(valid_from) or datetime.now(timezone.utc)
                vt_utc = far_future_dt
            else:
                vf_utc = to_utc(valid_from)
                vt_utc = to_utc(valid_to)
                if not vf_utc or not vt_utc:
                    gr.Warning("Укажите обе даты для новости (или отметьте 'Бессрочно')", title="Предупреждение")
                    return gr.update(visible=False), None
                if vf_utc > vt_utc:
                    gr.Warning("Дата 'с' позже даты 'по'", title="Предупреждение")
                    return gr.update(visible=False), None

        if doc_type == "news":
            for b in blocks:
                b.update(
                    {
                        "doc_type": "news",
                        "valid_from": vf_utc.isoformat().replace("+00:00", "Z"),
                        "valid_to": vt_utc.isoformat().replace("+00:00", "Z"),
                        "from_ts": to_ts(vf_utc),
                        "to_ts": to_ts(vt_utc),
                        "is_permanent": is_perm,
                    }
                )
        else:
            for b in blocks:
                b.update({"doc_type": "static"})

        meta = {
            "doc_id": doc_id,
            "index": selected_index,
            "blocks": blocks,
        }

        gr.Info(f"✅ Итого: '{doc_id}', блоков: {len(blocks)} → '{selected_index}'", title="Инфо")
        return (
            gr.update(visible=True, value=blocks),
            meta,
        )

    preview_button.click(
        fn_preview_json,
        inputs=[
            id_input,
            title_input,
            content_input,
            keywords_input,
            table_state,
            index_dropdown,
            doc_type_radio,
            valid_from_dp,
            valid_to_dp,
            permanent_cb,
        ],
        outputs=[preview_json, meta_state],
    )

    def save_and_send_to_meilisearch(meta, meili_view_index: str):
        if not meta:
            gr.Warning("Нет данных (сделайте Предпросмотр)", title="Предупреждение")
            return (
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                meta,
            )

        index_name = meta.get("index")
        blocks = meta.get("blocks") or []
        doc_id = meta.get("doc_id")

        if not index_name:
            gr.Warning("Не выбран индекс", title="Предупреждение")
            return (
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                meta,
            )

        if not blocks:
            gr.Warning("Пустой массив блоков", title="Предупреждение")
            return (
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                meta,
            )

        msg: str = ""
        try:
            msg = meilisearch_module.add_doc_to_meili(blocks, index_name)
        except Exception as e:
            gr.Error(f"{e}, сообщение от сервера Meilisearch: {msg}", title="Ошибка!")
            return (
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                meta,
            )

        gr.Success(
            f"✅ '{doc_id}', добавлено {len(blocks)} блок(ов) в '{index_name}', "
            f"сообщение от сервера Meilisearch: {msg}",
            title="Успешно",
        )

        id_select_update = update_docs_in_meili_index_fn(
            index_name,
            output="id_only",
            current_id=doc_id,
        )

        if meili_view_index == index_name:
            table_update, _ = update_docs_in_meili_index_fn(
                index_name,
                output="full",
            )
        else:
            table_update = gr.update()

        selected_for_delete_update = gr.update(value="")
        id_input_update = gr.update(value="")
        title_update = gr.update(value="")
        content_update = gr.update(value="")
        keywords_update = gr.update(value="")
        permanent_cb_upd = gr.update(value=False)
        preview_update = gr.update(visible=False, value=None)
        meta_out = None

        return (
            id_select_update,
            table_update,
            selected_for_delete_update,
            id_input_update,
            title_update,
            content_update,
            keywords_update,
            preview_update,
            permanent_cb_upd,
            meta_out,
        )

    id_input.change(
        validate_id_live_fn,
        inputs=[id_input, io_radio],
        outputs=[id_input],
    )

    def normalize_id_click(current: str) -> dict[str, Any]:
        cleaned = sanitize_id_fn(current or "")
        if cleaned:
            gr.Info(f"ID корректен: {cleaned}", title="Инфо")
            return gr.update(value=cleaned)

        gr.Warning("ID всё ещё некорректен", title="Предупреждение")
        return gr.update()

    normalize_id_btn.click(normalize_id_click, inputs=id_input, outputs=id_input)

    def gen_id_from_title(title: str) -> dict[str, Any]:
        cleaned = sanitize_id_fn(title or "")
        if cleaned:
            gr.Success(f"ID сгенерирован: {cleaned}", title="Успешно")
            return gr.update(value=cleaned)
        gr.Warning("Не удалось сгенерировать ID из заголовка", title="Предупреждение")
        return gr.update()

    generate_id_from_title_btn.click(gen_id_from_title, inputs=[title_input], outputs=id_input)

    def vanish_all_windows():
        gr.Info("Все окна очищены, готов к загрузке нового документа", title="Инфо")
        return (
            gr.update(value="create"),
            "",
            gr.update(value=""),
            "",
            "",
            "",
            gr.update(value=default_empty_table, visible=False),
            False,
            "static",
            None,
            None,
            False,
            gr.update(visible=False, value=None),
            None,
        )

    vanish_screen_btn.click(
        vanish_all_windows,
        outputs=[
            io_radio,
            id_input,
            id_select,
            title_input,
            content_input,
            keywords_input,
            table_df,
            show_table_cb,
            doc_type_radio,
            valid_from_dp,
            valid_to_dp,
            permanent_cb,
            preview_json,
            meta_state,
        ],
    )

    def _parse_iso(x: str | None):
        if not x:
            return None
        try:
            return datetime.fromisoformat(x.replace("Z", "+00:00")).astimezone(timezone.utc)
        except Exception:
            return None

    def load_doc_into_form_by_id(index_name: str, doc_id: str):
        if not index_name:
            gr.Warning("Укажите индекс", title="Предупреждение")
            return (gr.update(),) * 13

        if not doc_id:
            gr.Warning("Укажите ID", title="Предупреждение")
            return (gr.update(),) * 13

        doc = meilisearch_module.get_document_by_id(index_name, doc_id)
        if not doc:
            gr.Warning("❌ Документ не найден", title="Предупреждение")
            return (gr.update(),) * 13

        title = doc.get("title") or ""
        content = doc.get("content") or ""
        idx_default_type = type_for_index.get(index_name, "static")
        doc_type = doc.get("doc_type") or doc.get("type") or idx_default_type
        if doc_type == "text":
            doc_type = "news" if idx_default_type == "news" else "static"
        if doc_type not in ("news", "static"):
            doc_type = idx_default_type
        is_news = doc_type == "news"

        vf = _parse_iso(doc.get("valid_from"))
        vt = _parse_iso(doc.get("valid_to"))
        is_perm = bool(doc.get("is_permanent", False))
        if is_news and not (vf or vt or is_perm):
            gr.Warning(
                "Это документ из индекса 'news', но даты не заданы. "
                "Укажите период или отметьте 'Бессрочно'.",
                title="Требуются даты",
            )

        raw_keywords = doc.get("keywords", "")
        if isinstance(raw_keywords, list):
            keywords = ", ".join(str(k) for k in raw_keywords if k)
        else:
            keywords = str(raw_keywords or "")

        table_val = doc.get("table") or default_empty_table
        table_has_data = any(str(cell).strip() for row in table_val if isinstance(row, list) for cell in row)

        gr.Success(message="✅ Документ загружен", title="Успешно")
        return (
            gr.update(value=doc_id),
            gr.update(value=title),
            gr.update(value=content),
            gr.update(value=vf),
            gr.update(value=vt),
            gr.update(value=is_perm, visible=is_news),
            gr.update(value=keywords),
            gr.update(value=table_val, visible=table_has_data),
            gr.update(value=table_has_data),
            gr.update(value=doc_type),
            gr.update(visible=is_news),
            gr.update(value=doc_id),
            gr.update(value=doc_id),
        )

    load_doc_btn.click(
        load_doc_into_form_by_id,
        inputs=[index_dropdown, id_select],
        outputs=[
            id_input,
            title_input,
            content_input,
            valid_from_dp,
            valid_to_dp,
            permanent_cb,
            keywords_input,
            table_df,
            show_table_cb,
            doc_type_radio,
            news_dates_row,
            id_select,
            orig_doc_id_state,
        ],
    )

    def save_doc_by_id(
        doc_type: Literal["static", "news", "text"],
        index_name: str,
        current_doc_id: str,
        orig_doc_id: str | None,
        title: str,
        content: str,
        valid_from,
        valid_to,
        permanent,
        keywords: str,
        table,
    ):
        doc_id = (current_doc_id or "").strip()
        orig_id = (orig_doc_id or "").strip() if orig_doc_id is not None else None

        if not index_name:
            gr.Warning("Укажите индекс", title="Предупреждение")
            return gr.update()
        if not doc_id or not is_valid_id_fn(doc_id):
            gr.Warning("Некорректный ID (разрешено [a-z0-9_], длина 3–60).", title="Предупреждение")
            return gr.update()
        if not title:
            gr.Warning("Создайте заголовок", title="Предупреждение")
            return gr.update()
        if not content:
            gr.Warning("Создайте контент", title="Предупреждение")
            return gr.update()

        expected_type = type_for_index.get(index_name, "static")
        if doc_type == "text":
            doc_type = expected_type
        if doc_type != expected_type:
            doc_type = expected_type
            gr.Warning("Тип документа приведён к выбранному индексу.", title="Предупреждение")

        base_doc: dict[str, Any] = {
            "id": doc_id,
            "title": title or "",
            "content": content or "",
            "keywords": keywords or "",
            "table": table or [],
        }

        if doc_type == "news":
            def _to_utc(dt):
                if dt is None:
                    return None
                return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

            if permanent:
                if not valid_from:
                    gr.Warning("Уточните дату начала", title="Предупреждение")
                    return gr.update()
                vf = _to_utc(valid_from)
                vt = datetime(2099, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
            else:
                if not valid_from or not valid_to:
                    gr.Warning("Укажите обе даты или отметьте 'Бессрочно'", title="Предупреждение")
                    return gr.update()
                vf, vt = _to_utc(valid_from), _to_utc(valid_to)
                if vf > vt:
                    gr.Warning("Дата 'с' позже даты 'по'", title="Предупреждение")
                    return gr.update()

            base_doc.update(
                {
                    "doc_type": "news",
                    "valid_from": vf.isoformat().replace("+00:00", "Z"),
                    "valid_to": vt.isoformat().replace("+00:00", "Z"),
                    "from_ts": int(vf.timestamp()),
                    "to_ts": int(vt.timestamp()),
                    "is_permanent": bool(permanent),
                }
            )
        else:
            base_doc.update({"doc_type": "static"})

        is_rename = bool(orig_id) and (orig_id != doc_id)
        if is_rename:
            gr.Info(f"Переименование документа: старый ID = {orig_id}, новый ID = {doc_id}", title="Инфо")
            try:
                print("DEBUG RENAME:", index_name, orig_id, "->", doc_id)
                meilisearch_module.delete_meili_document(index_name, orig_id)
            except Exception as e:
                gr.Warning(
                    f"Не удалось удалить старый документ с ID {orig_id}: {e}",
                    title="Предупреждение",
                )

        try:
            msg = meilisearch_module.upsert_document(index_name, base_doc)
        except Exception as e:
            gr.Error(f"Ошибка при сохранении документа: {e}", title="Ошибка!")
            return gr.update()

        gr.Success(f"✅ Сохранено (ответ сервера: {msg})", title="Успешно")
        return update_docs_in_meili_index_fn(index_name, output="id_only", current_id=doc_id)

    save_doc_btn_direct.click(
        save_doc_by_id,
        inputs=[
            doc_type_radio,
            index_dropdown,
            id_select,
            orig_doc_id_state,
            title_input,
            content_input,
            valid_from_dp,
            valid_to_dp,
            permanent_cb,
            keywords_input,
            table_state,
        ],
        outputs=[id_select],
    )

    def create_or_change_fn(choose: Literal["create", "change"] = "create"):
        if choose == "create":
            return (
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=True),
            )
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        )

    io_radio.change(
        create_or_change_fn,
        inputs=[io_radio],
        outputs=[
            id_input,
            id_select,
            normalize_id_btn,
            generate_id_from_title_btn,
            vanish_screen_btn,
            load_doc_btn,
            save_doc_btn_direct,
            preview_json,
            save_button,
            preview_button,
        ],
    )

    index_dropdown.change(
        fn=lambda idx: update_docs_in_meili_index_fn(idx, output="id_only"),
        inputs=index_dropdown,
        outputs=id_select,
    )

    def on_meili_table_select(evt: gr.SelectData):
        row = evt.row_value or []
        if not row:
            return gr.update(value="")
        doc_id = str(row[0] if row[0] is not None else "").strip()
        return gr.update(value=doc_id)

    meili_indices_table.select(
        on_meili_table_select,
        inputs=None,
        outputs=[meili_selected_doc_id_box],
    )

    radio_type_of_search.change(
        fn=radio_sliders_change_fn,
        inputs=radio_type_of_search,
        outputs=[
            value_n_results_slider,
            thresholdvalue_slider,
            value_k_slider,
            meili_search_indexes_dropdown,
            chroma_search_collection_dropdown,
        ],
    )

    radio_type_of_db.change(
        fn=radio_search_engine_change_fn,
        inputs=radio_type_of_db,
        outputs=[
            upload_collections_dropdown,
            upload_indices_dropdown,
            add_collection_button,
            rm_collection_button,
            add_index_button,
            rm_index_button,
            add_to_collection_button,
            add_to_index_button,
            radio_type_of_upl_data,
        ],
    )

    radio_type_of_upl_data.change(
        fn=radio_type_of_upl_file_change_fn,
        inputs=radio_type_of_upl_data,
        outputs=[pdf, json_file],
    )

    add_collection_button.click(
        gr_create_collection_fn,
        inputs=upload_collections_dropdown,
        outputs=[upload_collections_dropdown, chroma_search_collection_dropdown],
    )

    rm_collection_button.click(
        gr_remove_collection_fn,
        inputs=upload_collections_dropdown,
        outputs=[upload_collections_dropdown, chroma_search_collection_dropdown],
    )

    add_to_collection_button.click(
        gr_add_to_collection_fn,
        inputs=[upload_collections_dropdown, pdf],
        outputs=[pdf],
    )

    save_button.click(
        save_and_send_to_meilisearch,
        inputs=[meta_state, meili_ind_for_cont_dropdown],
        outputs=[
            id_select,
            meili_indices_table,
            meili_selected_doc_id_box,
            id_input,
            title_input,
            content_input,
            keywords_input,
            preview_json,
            permanent_cb,
            meta_state,
        ],
    )

    rm_index_button.click(
        gr_remove_index_fn,
        inputs=upload_indices_dropdown,
        outputs=[
            upload_indices_dropdown,
            meili_search_indexes_dropdown,
            meili_ind_for_cont_dropdown,
            index_dropdown,
        ],
    )

    add_index_button.click(
        gr_create_index_fn,
        inputs=upload_indices_dropdown,
        outputs=[
            upload_indices_dropdown,
            meili_search_indexes_dropdown,
            meili_ind_for_cont_dropdown,
            index_dropdown,
        ],
    )

    add_to_index_button.click(
        gr_add_to_index_universal_fn,
        inputs=[upload_indices_dropdown, pdf, json_file, radio_type_of_upl_data],
        outputs=[pdf, json_file, upload_indices_dropdown, meili_search_indexes_dropdown],
    )

    def rm_doc_and_refresh_all(
        index_for_delete: str,
        doc_id_to_delete: str,
        constructor_index: str,
        constructor_current_id: str | None,
    ):
        gr_rm_doc_from_index_fn(index_for_delete, doc_id_to_delete)
        if not index_for_delete:
            return gr.update(), gr.update(), gr.update(value="")

        table_update, _ = update_docs_in_meili_index_fn(
            index_for_delete,
            output="full",
        )

        if constructor_index == index_for_delete:
            constructor_dd_update = update_docs_in_meili_index_fn(
                constructor_index,
                output="id_only",
                current_id=constructor_current_id,
            )
        else:
            constructor_dd_update = gr.update()

        return (
            table_update,
            constructor_dd_update,
            gr.update(value=""),
        )

    rm_doc_from_index_button.click(
        rm_doc_and_refresh_all,
        inputs=[
            meili_ind_for_cont_dropdown,
            meili_selected_doc_id_box,
            index_dropdown,
            id_select,
        ],
        outputs=[
            meili_indices_table,
            id_select,
            meili_selected_doc_id_box,
        ],
    )

    chroma_coll_for_cont_dropdown.change(
        fn=update_docs_in_chroma_collection_fn,
        inputs=chroma_coll_for_cont_dropdown,
        outputs=[chroma_collection_content, chroma_collection_table],
    )

    def refresh_meili_views(
        index_for_view: str,
        constructor_index: str,
        constructor_current_id: str | None,
    ):
        if not index_for_view:
            return gr.update(), gr.update(), gr.update(value="")

        if index_for_view == "news":
            table_update, _ = update_docs_in_meili_index_fn(
                index_for_view,
                output="full_news",
            )
        else:
            table_update, _ = update_docs_in_meili_index_fn(
                index_for_view,
                output="full",
            )

        if constructor_index == index_for_view:
            constructor_dd_update = update_docs_in_meili_index_fn(
                constructor_index,
                output="id_only",
                current_id=constructor_current_id,
            )
        else:
            constructor_dd_update = gr.update()

        return table_update, constructor_dd_update, gr.update(value="")

    refresh_data_btn.click(
        refresh_meili_views,
        inputs=[meili_ind_for_cont_dropdown, index_dropdown, id_select],
        outputs=[meili_indices_table, id_select, meili_selected_doc_id_box],
    )

    meili_ind_for_cont_dropdown.change(
        fn=refresh_meili_views,
        inputs=[meili_ind_for_cont_dropdown, index_dropdown, id_select],
        outputs=[meili_indices_table, id_select, meili_selected_doc_id_box],
    )
