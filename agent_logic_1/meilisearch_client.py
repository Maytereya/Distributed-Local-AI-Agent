"""
v0.3

This module provides two ways to work with Meilisearch:
1) Using the official Python client for adding documents and searching them.
2) Using raw HTTP requests for listing indexes, listing documents,
   and removing them (as well as deleting entire indexes).

"""
from __future__ import annotations

import json
# Retry section
# import time
import logging
import os
import time
from collections.abc import Sequence, Mapping
from datetime import datetime, timezone
from typing import List, Literal, Any, Optional, Dict
from typing import Union

import meilisearch
import requests
# from httpx import AsyncClient, ConnectError
from tenacity import retry, stop_after_attempt, wait_fixed  # Для автоматических ретраев

#
from agent_logic_2 import config as c

# from gradio_interface import waiter

# --------------------------------------
# Секция загрузки и ретраев для отладки
# --------------------------------------

#  Initialize logging for connection tries
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Функция для ретраев при подключении
@retry(stop=stop_after_attempt(10), wait=wait_fixed(6))  # 10 попыток, ожидание 6 секунд
def connect_to_meilisearch():
    logger.info("🔄 Подключение к MeiliSearch...")
    return meilisearch.Client(c.MEILI_URL, c.MASTER_KEY)


try:
    client = connect_to_meilisearch()
    logger.info("✅ Успешное подключение к MeiliSearch!")
    ver = client.get_version()  # dict
    # прим.: {'pkgVersion': '1.11.0', 'commitSha': '...', 'buildDate': '...'}
    pkg_version = ver.get("pkgVersion")
    logger.info(f"Meilisearch server ver.: {pkg_version}")
except Exception as e:
    logger.error(f"❌ Ошибка подключения к MeiliSearch: {e}")

# -----------------------------------------------------------
# Настройки для индексов двух типов: скриптовый и новостной.
# -----------------------------------------------------------


MAIN_SETTINGS = {
    "searchableAttributes": ["title", "content", "keywords", "html", "csv"],
    "displayedAttributes": ["*"],
    "filterableAttributes": ["doc_id", "doc_type", "page", "block_id", "keywords"],
    "sortableAttributes": ["page", "block_id", "created_at"],
}

NEWS_SETTINGS = {
    "searchableAttributes": ["title",
                             "content",
                             "keywords"],
    "displayedAttributes": ["*"],
    "filterableAttributes": [
        "from_ts",
        "to_ts",
        "is_permanent",
        "doc_type",  # Разобраться с этим атрибутом. Где-то он type...
        "tags",
        "keywords"
    ],
    "sortableAttributes": ["from_ts", "to_ts"],
}

# Какие списковые ключи мержим по-множественному
_LIST_KEYS = {
    "searchableAttributes",
    "displayedAttributes",
    "filterableAttributes",
    "sortableAttributes",
    "stopWords",
    "synonyms",  # тут мапа: отдельно обрабатываем
}


def _as_set(v):
    if v is None:
        return set()
    if isinstance(v, (list, tuple, set)):
        return set(v)
    return set([v])


def ensure_index(client, index_name: str, primary_key: str = "id",
                 settings_patch: Mapping | None = None,
                 wait_fn=None):
    """
    Создаёт индекс при отсутствии и НЕразрушительно дополняет настройки (union).
    settings_patch — словарь с фрагментом настроек, который нужно гарантировать.
    wait_fn(task) —  обвязка ожидания задач Meili (taskUid/uid).
    """
    # 1) get or create
    try:
        index = client.get_index(index_name)
    except Exception:
        task = client.create_index(index_name, {"primaryKey": primary_key})
        if wait_fn:
            wait_fn(task)

        index = client.index(index_name)

    if not settings_patch:
        return index

    # 2) забираем текущие настройки
    current = index.get_settings()  # dict

    # 3) готовим патч без разрушений: union на списках, merge на synonyms
    merged = {}

    for k, want in settings_patch.items():
        if k == "synonyms" and isinstance(want, Mapping):
            cur_syn = current.get("synonyms") or {}
            cur_syn = dict(cur_syn)
            cur_syn.update(want)
            merged[k] = cur_syn
            continue

        if k in _LIST_KEYS:
            have_list = current.get(k) or []
            want_list = list(want or [])

            # ── спец-логика для searchableAttributes ─────────────────────────
            if k == "searchableAttributes":
                if "*" in want_list:
                    # Явно хотим wildcard → так и ставим
                    merged[k] = ["*"]
                elif "*" in have_list:
                    # На индексе сейчас wildcard, а мы хотим конкретный список → ПЕРЕЗАПИСЫВАЕМ
                    merged[k] = want_list
                else:
                    merged[k] = sorted(set(have_list) | set(want_list))
                continue

            # ── спец-логика для displayedAttributes (оставить wildcard) ──────
            if k == "displayedAttributes" and ("*" in have_list or "*" in want_list):
                merged[k] = ["*"]
                continue

            # ── дефолт: объединение множеств ─────────────────────────────────
            merged[k] = sorted(set(have_list) | set(want_list))
        else:
            merged[k] = want

    # 4) применяем, ждём task
    task = index.update_settings(merged)
    if wait_fn:
        wait_fn(task)

    return index


def _extract_task_uid(task_info: object) -> int:
    if isinstance(task_info, int):
        return task_info
    if isinstance(task_info, str) and task_info.isdigit():
        return int(task_info)
    # TaskInfo из 0.33.1
    if hasattr(task_info, "task_uid"):
        return int(getattr(task_info, "task_uid"))
    # на всякий, если кто-то передал dict
    if isinstance(task_info, dict):
        v = task_info.get("task_uid") or task_info.get("uid")
        if v is not None:
            return int(v)
    raise ValueError(f"Не удалось извлечь task_uid: {task_info!r}")


def _task_to_dict(task: object) -> dict:
    if isinstance(task, dict):
        return task
    # ожидаем объект Task из 0.33.1
    try:
        d = {
            "uid": getattr(task, "uid", None),
            "index_uid": getattr(task, "index_uid", None),
            "status": getattr(task, "status", None),
            "type": getattr(task, "type", None),
            "enqueued_at": getattr(task, "enqueued_at", None),
            "started_at": getattr(task, "started_at", None),
            "finished_at": getattr(task, "finished_at", None),
            "duration": getattr(task, "duration", None),
            "error": getattr(task, "error", None),
        }
    except Exception:
        # крайний случай: просто распаковать dataclass/obj
        try:
            d = dict(vars(task))
        except Exception:
            d = {}
    # убрать None-ключи, чтобы не засорять логи
    return {k: v for k, v in d.items() if v is not None}


def wait_for_task_completion(client, task_info: Any, *, timeout: float = 60.0,
                             poll_interval: float = 0.25, raise_on_fail: bool = False) -> tuple[str, dict]:
    """
    Ожидание задачи Meilisearch (SDK 0.33.1+).
    task_info: объект TaskInfo ИЛИ int uid.
    Возвращает: (status, task_dict)
    """
    # TaskInfo.task_uid или сразу int
    task_uid = task_info.task_uid if hasattr(task_info, "task_uid") else int(task_info)

    start = time.time()
    while True:
        t = client.get_task(task_uid)  # -> Task
        status = t.status  # 'enqueued' | 'processing' | 'succeeded' | 'failed' | 'canceled'

        if status in {"succeeded", "failed", "canceled"}:
            if raise_on_fail and status != "succeeded":
                # у Task обычно есть .error (dict | None)
                raise RuntimeError(f"Meili task {task_uid} -> {status}: {getattr(t, 'error', None)}")
            # нормализуем в dict
            td = vars(t) if not isinstance(t, dict) else t
            return status, td

        if time.time() - start > timeout:
            raise TimeoutError(f"Task {task_uid} не завершилась за {timeout} с, статус: {status}")

        time.sleep(poll_interval)


def add_doc_to_meili(
        docs: Union[str, os.PathLike, Sequence[dict]],
        index_name: str,
        *,
        timeout: int = 60,
        poll_interval: float = 1.0,
) -> str:
    # 1) Нормализуем вход
    if isinstance(docs, (str, os.PathLike)):
        with open(docs, "r", encoding="utf-8") as f:
            blocks = json.load(f)
    else:
        blocks = list(docs)

    # 2) Валидация
    if isinstance(blocks, dict):
        blocks = [blocks]
    if not isinstance(blocks, list) or not all(isinstance(x, dict) for x in blocks):
        raise ValueError("add_doc_to_meili: ожидается список словарей (list[dict]).")
    if not blocks:
        raise ValueError("add_doc_to_meili: список документов пуст.")

    # 3) Отправка + ожидание завершения
    try:
        task_info = client.index(index_name).add_documents(blocks)
        status, task = wait_for_task_completion(
            client, task_info, timeout=timeout, poll_interval=poll_interval
        )

        if status == "succeeded":
            uid = task.get("uid") or task.get("taskUid")
            msg = f"Документы добавлены в индекс '{index_name}'. task={uid}"
            logger.info(msg)
            return msg

        msg = f"Индексирование завершилось со статусом '{status}'. task={task}"
        logger.warning(msg)
        return msg

    except Exception as e:
        logger.exception("Ошибка добавления документов в Meilisearch")
        return f"Ошибка добавления документа(ов) в индекс '{index_name}': {e}"


def parse_iso(x: str | None) -> str | None:
    """
    Приводит дату и время из индекса meili в удобочитаемый формат
    :param x: дата из meili
    :return: дата в удобочитаемом формате
    """
    if not x: return "не установлена"
    try:
        return str(datetime.fromisoformat(x.replace("Z", "")).astimezone(timezone.utc).strftime("%A, %d %B %Y, %H:%M"))
    except Exception:
        return None


def meili_list_documents(
        index_name: str,
        return_type: Literal["ID", "All", "All_News"] = "ID",
        content_limit: int = 150,
        doc_count_limit: int = 200,
) -> Union[List[List[str]], List[str]]:
    """
    Возвращает:
      - "ID": список id
      - "All": список [id, valid_from, valid_to, is_permanent, title_or_default, cropped_content]
    """
    # Поля, которые важны в документе
    fields = ['id', 'title', 'content', 'valid_from', 'valid_to', 'is_permanent']

    data = client.index(index_name).get_documents({'limit': doc_count_limit, 'fields': fields})
    array_of_docs = data.results

    def safe_get(doc: Any, key: str, default: str = "") -> str:
        # 1) атрибут
        try:
            val = getattr(doc, key)
        except AttributeError:
            # 2) доступ по ключу
            try:
                val = doc[key]
            except Exception:
                val = default
        return "" if val is None else str(val)

    def crop(text: str, limit: int) -> str:
        return text[:limit] + "..." if len(text) > limit else text

    out_all: List[List[str]] = []
    out_all_news: List[List[str]] = []
    out_ids: List[str] = []

    for doc in array_of_docs:
        doc_id = safe_get(doc, "id", "")
        title = safe_get(doc, "title", "")
        if not title:
            title = "Без заголовка"
        content = crop(safe_get(doc, "content", ""), content_limit)
        valid_from = parse_iso(safe_get(doc, "valid_from", ""))
        valid_to = parse_iso(safe_get(doc, "valid_to", ""))
        if safe_get(doc, "is_permanent", ""):
            is_permanent = valid_to = "Бессрочная" # в случае,
            # если новость/акция "вечная" отключаем отображение конечной даты
        else:
            is_permanent = "Временная"

        out_ids.append(doc_id)
        out_all.append([doc_id, title, content, ])
        out_all_news.append([doc_id, title, is_permanent, valid_from, valid_to, content, ])

    # return out_all if return_type == "All" else out_ids
    if return_type == "All":
        return out_all
    elif return_type == "All_News":
        return out_all_news
    else:
        return out_ids


def search_meili(
        index_name: str,
        query: str,
        limit: int = 3,
        highlight: str = None,
        highlight_fields: str = '*',
        output_mode: str = "full",
        max_chars: int | None = None,
) -> str:
    """
    Выполняет поисковый запрос в указанном индексе Meilisearch и возвращает
    результаты в одном из режимов:
    - full: метаданные + контент (обратная совместимость, поведение по умолчанию);
    - content_only: только текстовые фрагменты контента.

    Функция ищет документы, соответствующие переданному запросу, и формирует
    удобочитаемую строку с информацией о каждом найденном документе. В вывод
    включаются: ID документа, заголовок, имя файла, номер страницы и фрагмент
    содержимого (используя поле '_formatted' для подсветки).

    :param index_name: Имя индекса Meilisearch, в котором выполняется поиск.
                       Не должно быть пустым.
    :param query: Строка поискового запроса.
    :param limit: Максимальное количество возвращаемых результатов (по умолчанию 3).
    :param highlight: Параметр для тегов подсветки (в текущей реализации не используется).
    :param highlight_fields: Поля, по которым Meilisearch выполняет подсветку.
                             По умолчанию '*' — все поля.
    :param output_mode: Режим вывода:
                        - "full" (default): ID/заголовок/файл/страница + контент;
                        - "content_only": только контент без сервисных полей.
    :param max_chars: Опциональное ограничение длины ответа в символах.
                      Если задано и ответ длиннее, хвост обрезается.

    :return: Строка с результатами поиска, где документы разделены
             '\n---------\n'. Возможные варианты возврата:
             - форматированный список найденных документов;
             - сообщение "Совпадений не найдено, cформулируйте запрос иначе",
             - текст ошибки при некорректном индексе или сбое поиска.

    :raises: Исключения не пробрасываются наружу — все ошибки перехватываются
             и возвращаются в виде текстового сообщения.
    """

    if not index_name:
        return (
            "Ошибка: не выбран индекс, в котором следует осуществлять поиск"
        )

    try:
        search_result = client.index(index_name).search(query, {
            "limit": limit,
            # "highlightPreTag": highlight,
            # "highlightPostTag": highlight,
            "attributesToHighlight": [highlight_fields],
        })

        hits = search_result.get("hits", [])

        # --------------------------------------------
        def fix_none_err(v, default="не указан"):
            return default if v is None else str(v)

        # --------------------------------------------

        mode = str(output_mode or "full").strip().lower()
        if mode not in {"full", "content_only"}:
            mode = "full"

        # Собираем все куски контента:
        contents = []
        for doc in hits:
            # doc['_formatted'] может не всегда быть, поэтому используем .get(...)
            fmt = doc.get("_formatted", {})
            _id = fix_none_err(doc.get("id"))
            _file_name = fix_none_err(doc.get("file_name", "не указан"))
            _title = fix_none_err(doc.get("title", "не указан"))
            _page_number = fix_none_err(doc.get("page_number", 1))
            content_src = fmt.get("content")
            if content_src is None:
                content_src = doc.get("content", "")
            content_str = str(content_src or "").strip()

            if mode == "content_only":
                if content_str:
                    contents.append(content_str)
                continue

            contents.append("ID документа: " + _id)
            contents.append("Заголовок: " + _title)
            contents.append("Имя файла: " + _file_name)
            contents.append("Номер страницы: " + str(_page_number))
            contents.append(content_str)

        # Склеиваем в итоговую строку
        if mode == "content_only":
            combined_text = "\n\n".join(contents).strip()
        else:
            combined_text = "\n---------\n".join(contents)

        if isinstance(max_chars, int) and max_chars > 0 and len(combined_text) > max_chars:
            combined_text = combined_text[:max_chars].rstrip() + "…"

        if len(combined_text) == 0:
            combined_text = "Совпадений не найдено, cформулируйте запрос иначе"
        return combined_text

    except Exception as e:
        print(f"Error searching in index '{index_name}': {e}")
        return f"Ошибка поисковой системы Meilisearch: {e}"


# -----------------------------
# Прямые HTTP - запросы
# -----------------------------

def show_list_indexes(detail_mode: str = "full") -> list:
    """
    Отправляет GET-запрос к Meilisearch для получения списка всех индексов.

    :param detail_mode:
        - "full": вернуть полный список словарей индексов
                  (каждый содержит 'uid', 'createdAt', 'updatedAt', 'primaryKey').
        - "uid":  вернуть только список значений 'uid'.
                  Любое иное значение трактуется как "full".
    :return: Список индексов (в виде словарей) или список строк 'uid',
             в зависимости от выбранного режима. В случае ошибки возвращает пустой список.
    """
    endpoint = f"{c.MEILI_URL}/indexes"
    headers = {"Authorization": f"Bearer {c.MASTER_KEY}"}

    try:
        response = requests.get(endpoint, headers=headers, timeout=3)
        if response.status_code == 200:
            data = response.json()
            indexes = data.get("results", [])

            if detail_mode == "uid":
                # Extract just the 'uid' fields
                uids = [idx.get("uid") for idx in indexes]
                # print("Index UIDs found:", uids)
                return uids
            else:
                # detail_mode == "full" or any other unexpected value
                print("Indexes found:", indexes)
                return indexes
        else:
            print("Error listing indexes:", response.text)
            return []
    except requests.exceptions.RequestException as e:
        print(f"Error requesting indexes: {e}")
        return []


def create_index(index_uid: str, primary_key: str = "id") -> None:
    """
    Создаёт новый индекс в Meilisearch с указанным UID и primary key.

    :param index_uid: Уникальный идентификатор создаваемого индекса.
    :param primary_key: Ключ документа, который Meilisearch будет считать первичным.
                        По умолчанию "id".
    :return: None. Результат создания выводится в консоль. При сетевой ошибке
             выводится сообщение об исключении.
    """
    endpoint = f"{c.MEILI_URL}/indexes"
    headers = {"Authorization": f"Bearer {c.MASTER_KEY}"}
    payload = {
        "uid": index_uid,
        "primaryKey": primary_key,
    }
    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=3)
        # Можно обработать статус ответа — например, 201 говорит о создании,
        # но в любом случае выведем результат (или при необходимости вернём)
        print(f"Info about creating index '{index_uid}': {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Error creating index '{index_uid}': {e}")


def delete_index(index_uid: str) -> None:
    """
    Удаляет индекс Meilisearch по его UID.

    :param index_uid: Уникальный идентификатор индекса, который требуется удалить.
    :return: None. Результат операции выводится в консоль. При сетевой ошибке
             выводится сообщение об исключении.
    """
    endpoint = f"{c.MEILI_URL}/indexes/{index_uid}"
    headers = {"Authorization": f"Bearer {c.MASTER_KEY}"}

    try:
        response = requests.delete(endpoint, headers=headers, timeout=10)
        # if response.status_code == 204:
        #     print(f"Index '{index_uid}' deleted successfully.")
        # else:
        print(f"Info about deleting index '{index_uid}': {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Error deleting index '{index_uid}': {e}")


def get_meili_list_documents(index_uid: str, limit: int = 20, offset: int = 0) -> list:
    """
    Возвращает список документов из указанного индекса Meilisearch с пагинацией.

    :param index_uid: UID индекса, из которого необходимо получить документы.
    :param limit: Количество документов для выборки (по умолчанию 20).
    :param offset: Смещение, откуда начинать выборку (по умолчанию 0).
    :return: Список документов (dict). В случае ошибки — пустой список.
    """
    endpoint = f"{c.MEILI_URL}/indexes/{index_uid}/documents"
    headers = {"Authorization": f"Bearer {c.MASTER_KEY}"}
    params = {"limit": limit, "offset": offset}

    try:
        response = requests.get(endpoint, headers=headers, params=params, timeout=10)
        if response.status_code == 200:
            # print("Response status code:", str(response.status_code))
            data = response.json()
            documents = data.get("results", [])

            # print(f"Documents in index '{index_uid}':")
            # print(json.dumps(data, indent=2, ensure_ascii=False))

            return documents
        else:
            print(f"Error listing documents for index (without rising an exception) '{index_uid}': {response.text}")
            return []
    except requests.exceptions.RequestException as glde:
        print(f"Error listing documents for index (risen exception) '{index_uid}': {glde}")
        return []


def delete_meili_document(index_uid: str, doc_id: str) -> None:
    """
   Удаляет конкретный документ из индекса по его ID.

    :param index_uid: UID индекса, из которого должен быть удалён документ.
    :param doc_id: Идентификатор документа, подлежащего удалению.
    :return: None. Информация о результате удаления выводится в консоль.
             При ошибке выводится текст исключения.
    """
    endpoint = f"{c.MEILI_URL}/indexes/{index_uid}/documents/{doc_id}"
    headers = {"Authorization": f"Bearer {c.MASTER_KEY}"}

    try:
        response = requests.delete(endpoint, headers=headers, timeout=10)
        if response.status_code == 202:
            data = response.json()
            print(f"Document '{doc_id}' deletion enqueued.")
            print(json.dumps(data, indent=2, ensure_ascii=False))
        else:
            print(f"Error deleting document '{doc_id}' in index '{index_uid}': {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Error deleting document '{doc_id}' in index '{index_uid}': {e}")


# --------------------------------------------
# Функции, посвященные редактированию
# документов в индексе Meili
# --------------------------------------------

def get_document_by_id(index_name: str, doc_id: str) -> Optional[Dict[str, Any]]:
    """
    Возвращает документ из Meilisearch по его ID.

    Выполняет запрос:
        GET /indexes/{index}/documents/{id}

    :param index_name: Имя индекса.
    :param doc_id: ID документа.
    :return: dict с данными документа, если найден;
             None — если документ отсутствует (404) или произошла ошибка.
    """
    url = f"{c.MEILI_URL}/indexes/{index_name}/documents/{doc_id}"
    headers = {"Authorization": f"Bearer {c.MASTER_KEY}"}
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 404:
            return None
        print("get_document_by_id error:", r.status_code, r.text)
        return None
    except requests.RequestException as e:
        print("get_document_by_id exception:", e)
        return None


def upsert_document(index_name: str, doc: Dict[str, Any]) -> str:
    """
     Добавляет или обновляет документ в индексе Meilisearch по его primary key.

    Запрос:
        POST /indexes/{index}/documents
    Документ отправляется в массиве из одного элемента.

    :param index_name: Имя индекса.
    :param doc: Документ для сохранения (dict).
    :return: "OK" при успешной вставке/обновлении,
             либо строка с текстом ошибки.
    """
    url = f"{c.MEILI_URL}/indexes/{index_name}/documents"
    headers = {"Authorization": f"Bearer {c.MASTER_KEY}", "Content-Type": "application/json"}
    try:
        r = requests.post(url, headers=headers, json=[doc], timeout=20)
        if r.status_code in (200, 202):
            return "OK"
        return f"ERR {r.status_code}: {r.text}"
    except requests.RequestException as e:
        return f"ERR: {e}"


# ------------------------------------
# Обработка новостей / акций / промо
# ------------------------------------


def _now_ts_utc() -> int:
    return int(datetime.now(timezone.utc).timestamp())


def search_news_active(
        index_name: str = "news",
        keyword: str | None = None,
        *,
        now_ts: int | None = None,
        limit: int = 20,
        sort: list[str] | None = None,  # например ["from_ts:desc"]
) -> list[dict]:
    """
   Возвращает активные на текущий момент новости/акции.

    Документ считается активным, если текущий момент (или переданный now_ts)
    попадает в интервал [from_ts, to_ts], и doc_type = "news".

    :param index_name: Имя индекса новостей.
    :param keyword: Ключевое слово для поиска (может быть None).
    :param now_ts: Текущая временная отметка (timestamp).
                   Если не указано — вычисляется автоматически.
    :param limit: Максимальное количество результатов.
    :param sort: Параметры сортировки (по умолчанию ["from_ts:desc"]).
    :return: Список найденных документов (dict).
    """

    ts = _now_ts_utc() if now_ts is None else int(now_ts)
    # Пересечение интервалов: [from_ts, to_ts] с точкой now
    # + явный тип документа для чистоты
    flt = f'from_ts <= {ts} AND to_ts >= {ts} AND doc_type = "news"'

    # -------------------example-----------------------

    # search_result = client.index(index_name).search(query, {
    #     "limit": limit,
    #     # "highlightPreTag": highlight,
    #     # "highlightPostTag": highlight,
    #     "attributesToHighlight": [highlight_fields],
    # })

    res = client.index(index_name).search(keyword or "", {
        "filter": flt,
        "limit": limit,
        "sort": sort or ["from_ts:desc"],  # сначала свежие старты
    })
    # res["hits"] — список документов
    return res.get("hits", [])


# -----поиск новостей за период-----------------------------------
def search_news_by_period(
        #     Не факт, что пригодится, но пусть будет.

        index_name: str,
        keyword: str | None,
        start_ts: int,
        end_ts: int,
        limit: int = 50,
        sort: list[str] | None = None,
) -> list[dict]:
    """
    Выполняет поиск новостей/акций, период активности которых пересекается
    с заданным интервалом [start_ts, end_ts].

    :param index_name: Имя индекса.
    :param keyword: Ключевое слово для поиска (может быть None).
    :param start_ts: Начало временного интервала поиска.
    :param end_ts: Конец временного интервала поиска.
    :param limit: Максимальное количество результатов.
    :param sort: Параметры сортировки (по умолчанию ["from_ts:asc"]).
    :return: Список документов, пересекающих заданный период.
    """
    # Пересечение интервалов: [from_ts, to_ts] ∩ [start_ts, end_ts] ≠ Ø
    flt = f"from_ts <= {end_ts} AND to_ts >= {start_ts} AND doc_type = 'news'"
    res = client.index(index_name).search(keyword or "", {
        "filter": flt,
        "limit": limit,
        "sort": sort or ["from_ts:asc"],
    })
    return res.get("hits", [])


def main():
    """
    Пример использования. Настроить как нужно.
    """
    # 1) есть ли в принципе документы типа news
    from datetime import datetime, timezone
    now_ts = int(datetime.now(timezone.utc).timestamp())
    flt = f'from_ts <= {now_ts} AND to_ts >= {now_ts} AND doc_type = "news"'
    search_result = client.index("news").search("торакоцентез", {
        "limit": 10,
        "filter": flt,

        # "highlightPreTag": highlight,
        # "highlightPostTag": highlight,
        # "attributesToHighlight": [highlight_fields],
    })
    print(json.dumps(search_result, indent=2, ensure_ascii=False))

    NEWS_SETTINGS = {
        "searchableAttributes": ["title", "content", "keywords"],
        "displayedAttributes": ["*"],
        "filterableAttributes": ["from_ts", "to_ts", "is_permanent", "doc_type", "keywords"],
        "sortableAttributes": ["from_ts", "to_ts"],
    }
    ensure_index(client, "news", "id", NEWS_SETTINGS, )

    s = client.index("news").get_settings()
    print("searchable:", s.get("searchableAttributes"))

    # idx = client.index("news").search( {})
    # print(idx.search("", filter='doc_type = "news"', limit=3))
    #
    # # 2) есть ли документы с from_ts/to_ts (и какие значения)
    # hits = idx.search("", filter='doc_type = "news"', limit=50).get("hits", [])
    # print([(h.get("id"), h.get("from_ts"), h.get("to_ts")) for h in hits])
    #
    # # 3) что даёт «активные сейчас» без keyword

    # print(idx.search("", filter=flt, limit=3))

    print("=======")
    # doc: dict = {'id': 'skidka_50_na_manipulyaciyu_lor_hirurgiya_p1_b1', 'doc_id': 'skidka_50_na_manipulyaciyu_lor_hirurgiya', 'page': 1, 'block_id': 1, 'type': 'text', 'title': 'Скидка 50% на манипуляцию ЛОР, хирургия (+ check)', 'content': 'Скидка 50% на манипуляцию ЛОР, хирургия (+check2).\n_\nСкидка предоставляется на прием специалиста при прохождения данных манипуляций у доктора.\n_\nВНИМАНИЕ!   Пациент должен иметь на руках  протокол консультации врача, где указано, что  рекомендовано та или иная манипуляция (с него снимают копию и вклеивают в карту пациентки).\n  Если  протокола/направления от врача нет (и соответственно нет рекомендации для проведения данной манипуляции), то пациент оплачивает полную стоимость приема!\n_\nЗапись в Мед.центре: в примечании пишем 50%манипуляция\n_\nПродолжительность акции: не указана.', 'html': None, 'csv': None, 'keywords': [], 'created_at': '2025-10-13T17:12:38Z'}
    #
    print("=======")
    print(get_document_by_id("main_index", "obsluzhivanie_sotrudnikov_t_banka_po_chekapam_renessans_p1_b1"))
    # print(upsert_document("news", doc))
    # s_r = search_meili("news", "прием флеболога бесплатно")
    # print("=======")
    # print(s_r)


if __name__ == '__main__':
    doc = get_document_by_id("news", "novosti_za_1710")
    print(doc)
    print("Начало действия новости: ", parse_iso(doc.get("valid_from")))
    print("Окончание действия новости: ", parse_iso(doc.get("valid_to")))
    if bool(doc.get("is_permanent", False)):
        print("Бессрочная новость: ", bool(doc.get("is_permanent", False)))
    else:
        print("Cрочная новость")
    now = datetime.now()
    formatted_full = now.strftime("%A, %d %B %Y, %H:%M")  # Понедельник, 15 Декабрь 2025, 14:30
    print("Сейчас: ", formatted_full)

    info = meili_list_documents("news", "All", 50)
    for i in info:
        print(i)
