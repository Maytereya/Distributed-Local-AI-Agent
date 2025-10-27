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
# from gradio_interfaice import waiter

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
    wait_fn(task) — ваша обвязка ожидания задач Meili (taskUid/uid).
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


# def init_meili_index(index_name="main_index"):
#     """
#     Функция сообщения правильных атрибутов главному индексу
#     :param _client:
#     :param index_name:
#     :return:
#     """
#     try:
#         client.get_index(index_name)
#     except Exception:
#         client.create_index(index_name, {"primaryKey": "id"})
#
#     client.index(index_name).update_settings({
#         "searchableAttributes": ["title", "content", "keywords", "html", "csv"],
#         "displayedAttributes": ["*"],
#         "filterableAttributes": ["doc_id", "type", "page", "block_id", "keywords"],
#         "sortableAttributes": ["page", "block_id", "created_at"]
#     })


# def ensure_news_index_settings(index):
#     index.update_settings({
#         "filterableAttributes": list({"from_ts", "to_ts", "is_permanent", "type", "tags", "keywords"}),
#         "sortableAttributes": list({"from_ts", "to_ts"}),
#     })

# Удалить ветки hasattr(client, "tasks") и старые ключи в _extract_task_uid/_task_to_dict.

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


# def get_task_info(task_number: int) -> dict:
#     """
#     Retrieves information about a specific asynchronous task in Meilisearch.
#     Meilisearch uses tasks to handle operations such as adding, updating, or deleting documents.
#
#     :param task_number: The numeric ID of the task.
#     :return: A dictionary containing information about the requested task.
#     """
#     try:
#         task = client.get_task(task_number)
#         print("Task info:", task)
#         return task
#     except Exception as gte:
#         print(f"Error retrieving info for task #{task_number}: {gte}")
#         return {}


def meili_list_documents(
        index_name: str,
        return_type: Literal["ID", "All"] = "ID",
        content_limit: int = 150,
        doc_count_limit: int = 200,
) -> Union[List[List[str]], List[str]]:
    """
    Возвращает:
      - "ID": список id
      - "All": список [id, title_or_default, cropped_content]
    """

    data = client.index(index_name).get_documents({'limit': doc_count_limit, })
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
    out_ids: List[str] = []

    for doc in array_of_docs:
        doc_id = safe_get(doc, "id", "")
        title = safe_get(doc, "title", "")
        if not title:
            title = "Без заголовка"
        content = crop(safe_get(doc, "content", ""), content_limit)

        out_ids.append(doc_id)
        out_all.append([doc_id, title, content])

    return out_all if return_type == "All" else out_ids


def search_meili(index_name: str, query: str, limit: int = 3,
                 highlight: str = None, highlight_fields: str = '*') -> str:
    """
    Performs a search query in a specified Meilisearch index and returns formatted results.

    This function searches for documents matching the given query and formats the results
    into a human-readable string containing document metadata and content. Each result
    includes the document ID, title, filename, page number, and content.

    :param index_name: Name of the Meilisearch index to search in. Must not be empty.
    :param query: Search query string to match against searchable attributes.
    :param limit: Maximum number of search results to return. Defaults to 3.
    :param highlight: Highlight tag parameter (currently not used in implementation).
    :param highlight_fields: Fields to highlight in search results. Defaults to '*' (all fields).

    :return: A formatted string containing search results with document metadata and content,
             separated by '\n---------\n'. Returns an error message if:
             - The index_name is empty or not provided
             - No matches are found
             - An exception occurs during the search

    :raises: Does not raise exceptions directly; catches and returns error messages as strings.

    Example:
        >>> result = search_meili("main_index", "уретрит", limit=5)
        >>> print(result)
        ID документа: doc_123
        Заголовок: Medical Article
        Имя файла: medicine.pdf
        Номер страницы: 5
        Content text here...
        ---------
        ID документа: doc_124
        ...

    Note:
        - If no matches are found, returns: "Совпадений не найдено, cформулируйте запрос иначе"
        - Results are formatted with '_formatted' field from Meilisearch for proper highlighting
        - Missing document fields default to "не указан" (not specified) or appropriate defaults
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

        # Собираем все куски контента:
        contents = []
        for doc in hits:
            # doc['_formatted'] может не всегда быть, поэтому используем .get(...)
            fmt = doc.get("_formatted", {})
            _id = fix_none_err(doc.get("id"))
            _file_name = fix_none_err(doc.get("file_name", "не указан"))
            _title = fix_none_err(doc.get("title", "не указан"))
            _page_number = fix_none_err(doc.get("page_number", 1))
            content_str = fix_none_err(fmt.get("content", ""))

            contents.append("ID документа: " + _id)
            contents.append("Заголовок: " + _title)
            contents.append("Имя файла: " + _file_name)
            contents.append("Номер страницы: " + str(_page_number))
            contents.append(content_str)

        # Склеиваем в итоговую строку
        combined_text = "\n---------\n".join(contents)
        if len(combined_text) == 0:
            combined_text = "Совпадений не найдено, cформулируйте запрос иначе"
        return combined_text
        # return search_result

    except Exception as e:
        print(f"Error searching in index '{index_name}': {e}")
        return f"Ошибка поисковой системы Meilisearch: {e}"


# -----------------------------
# Raw HTTP request operations
# -----------------------------

def show_list_indexes(detail_mode: str = "full") -> list:
    """
    Sends a GET request to retrieve all indexes in Meilisearch.

    :param detail_mode:
        - "full": Print and return a list of index dictionaries
                  (each containing 'uid', 'createdAt', 'updatedAt', 'primaryKey').
        - "uid":  Print and return only a list of 'uid' values.
    :return: A list of either index dictionaries or just 'uid' strings,
             depending on 'detail_mode'. Returns an empty list on failure.
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
    Создаёт индекс в Meilisearch по обозначенному UID.

    :param primary_key: primary key for the new index.
    :param index_uid: Уникальный идентификатор (UID) для нового индекса.
    :return: None
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
    Deletes an entire index by UID.

    :param index_uid: The unique identifier of the index to delete.
    :return: None
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
    Sends a GET request to retrieve documents from a specific index, using pagination.

    :param index_uid: The unique identifier of the index.
    :param limit: How many documents to retrieve (default=20).
    :param offset: The starting offset (default=0).
    :return: A list of documents (dictionaries). Empty list on error.
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
    Deletes a specific document from an index by its document ID.

    :param index_uid: The index from which the document will be removed.
    :param doc_id: The ID of the document to delete.
    :return: None
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
    GET /indexes/{index}/documents/{id}
    Вернёт dict (документ) или None (если 404).
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
    POST /indexes/{index}/documents — add/replace по primaryKey (обычно 'id').
    На вход — один документ (мы отправляем массив из одного).
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
    Возвращает активные на текущий момент новости/акции (type="news", interval overlap).
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
    # Пересечение интервалов: [from_ts, to_ts] ∩ [start_ts, end_ts] ≠ Ø
    flt = f"from_ts <= {end_ts} AND to_ts >= {start_ts} AND type = 'news'"
    res = client.index(index_name).search(keyword or "", {
        "filter": flt,
        "limit": limit,
        "sort": sort or ["from_ts:asc"],
    })
    return res.get("hits", [])


def main():
    """
    Example usage. Adjust as needed.
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
    ensure_index(client, "news", "id", NEWS_SETTINGS, wait_fn=waiter)

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
    # print("=======")
    # print(get_document_by_id("news", "probnyi_dokument_so_vremenem_p1_b1"))
    # print(upsert_document("news", doc))
    s_r = search_meili("news", "прием флеболога бесплатно")
    # print("=======")
    # print(s_r)


if __name__ == '__main__':
    main()
