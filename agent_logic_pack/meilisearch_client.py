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
from collections.abc import Sequence
from typing import List, Literal, Any, Optional, Dict
from typing import Union

import meilisearch
import requests
# from httpx import AsyncClient, ConnectError
from tenacity import retry, stop_after_attempt, wait_fixed  # Для автоматических ретраев

#
from agent_logic_2 import config as c

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
except Exception as e:
    logger.error(f"❌ Ошибка подключения к MeiliSearch: {e}")


# TODO: Имя индекса задается жестко и не правится в gradio
def init_meili_index(index_name="main_index"):
    """
    Функция сообщения правильных атрибутов главному индексу
    :param _client:
    :param index_name:
    :return:
    """
    try:
        client.get_index(index_name)
    except Exception:
        client.create_index(index_name, {"primaryKey": "id"})

    client.index(index_name).update_settings({
        "searchableAttributes": ["title", "content", "keywords", "html", "csv"],
        "displayedAttributes": ["*"],
        "filterableAttributes": ["doc_id", "type", "page", "block_id", "keywords"],
        "sortableAttributes": ["page", "block_id", "created_at"]
    })


import time


def _task_to_dict(task):
    # Приводим Pydantic-модель к dict, если нужно
    if hasattr(task, "model_dump"):  # pydantic v2
        return task.model_dump()
    if hasattr(task, "dict"):  # pydantic v1
        return task.dict()
    if isinstance(task, dict):
        return task
    # последний шанс — собрать по атрибутам
    d = {}
    for attr in ("uid", "task_uid", "taskUid", "status", "type", "enqueued_at", "finished_at"):
        if hasattr(task, attr):
            d[attr] = getattr(task, attr)
    return d


def _extract_task_uid(task_info):
    """
    Поддерживает:
    - dict с 'taskUid' или 'uid'
    - Pydantic-модель TaskInfo с .task_uid / .uid
    """
    # Pydantic v1/v2
    if hasattr(task_info, "task_uid"):
        return task_info.task_uid
    if hasattr(task_info, "uid"):
        return task_info.uid

    # dict
    if isinstance(task_info, dict):
        return task_info.get("taskUid") or task_info.get("uid")

    # что-то ещё (модель без привычных полей) — пробуем через приведение к dict
    d = _task_to_dict(task_info)
    uid = d.get("taskUid") or d.get("uid")
    if uid is not None:
        return uid

    raise ValueError("Не удалось извлечь task uid из ответа Meilisearch.")


def wait_for_task_completion(client, task_info, *, timeout=60, poll_interval=1.0):
    """
    Универсальное ожидание завершения задачи Meilisearch.
    Поддерживает старые и новые версии Python-клиента.
    Возвращает: (status: str, task: dict)
    """
    task_uid = _extract_task_uid(task_info)
    start = time.time()

    while True:
        # В новых клиентах есть .tasks.get_task, в старых — client.get_task
        if hasattr(client, "tasks"):
            task = client.tasks.get_task(task_uid)
        else:
            task = client.get_task(task_uid)

        # Нормализуем к dict
        task_d = _task_to_dict(task)
        status = task_d.get("status")

        if status in {"succeeded", "failed", "canceled"}:
            return status, task_d

        if time.time() - start > timeout:
            raise TimeoutError(f"Задача {task_uid} не завершилась за {timeout} сек. Текущий статус: {status}")

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


def main():
    """
    Example usage. Adjust as needed.
    """

    print("=======")
    # doc: dict = {'id': 'skidka_50_na_manipulyaciyu_lor_hirurgiya_p1_b1', 'doc_id': 'skidka_50_na_manipulyaciyu_lor_hirurgiya', 'page': 1, 'block_id': 1, 'type': 'text', 'title': 'Скидка 50% на манипуляцию ЛОР, хирургия (+ check)', 'content': 'Скидка 50% на манипуляцию ЛОР, хирургия (+check2).\n_\nСкидка предоставляется на прием специалиста при прохождения данных манипуляций у доктора.\n_\nВНИМАНИЕ!   Пациент должен иметь на руках  протокол консультации врача, где указано, что  рекомендовано та или иная манипуляция (с него снимают копию и вклеивают в карту пациентки).\n  Если  протокола/направления от врача нет (и соответственно нет рекомендации для проведения данной манипуляции), то пациент оплачивает полную стоимость приема!\n_\nЗапись в Мед.центре: в примечании пишем 50%манипуляция\n_\nПродолжительность акции: не указана.', 'html': None, 'csv': None, 'keywords': [], 'created_at': '2025-10-13T17:12:38Z'}
    #
    # print("=======")
    # print(get_document_by_id("news", "skidka_50_na_manipulyaciyu_lor_hirurgiya_p1_b1"))
    # print(upsert_document("news", doc))
    # s_r = search_meili("news", "прием флеболога бесплатно")
    print("=======")
    # print(s_r)


if __name__ == '__main__':
    main()
