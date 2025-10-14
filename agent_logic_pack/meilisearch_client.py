"""
v0.3

This module provides two ways to work with Meilisearch:
1) Using the official Python client for adding documents and searching them.
2) Using raw HTTP requests for listing indexes, listing documents,
   and removing them (as well as deleting entire indexes).

"""

import json
# Retry section
# import time
import logging
from typing import List, Literal, Any, Union

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
    print("=== Index settings have been updated ===")


def add_doc_to_meili(blocks: List[dict], index_name: str) -> str:
    """
    Adds JSON documents from a local file to a Meilisearch index.
    If the index does not exist, it will be created automatically.

    :param blocks:
    :param doc_path: Path to the local JSON file containing the documents (list of dicts).
    :param index_name: Name of the target Meilisearch index.
    :return: None
    """

    try:
        task_info = client.index(index_name).add_documents(blocks)
        msg = f"Документ успешно поставлен в очередь на добавление в индекс '{index_name}': {task_info}"
        print(msg)
        return msg
    except Exception as e:
        msg = f"Ошибка добавления документа в индекс '{index_name}': {e}"
        print(msg)
        return msg


def get_task_info(task_number: int) -> dict:
    """
    Retrieves information about a specific asynchronous task in Meilisearch.
    Meilisearch uses tasks to handle operations such as adding, updating, or deleting documents.

    :param task_number: The numeric ID of the task.
    :return: A dictionary containing information about the requested task.
    """
    try:
        task = client.get_task(task_number)
        print("Task info:", task)
        return task
    except Exception as gte:
        print(f"Error retrieving info for task #{task_number}: {gte}")
        return {}


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

        # Собираем все куски контента:
        contents = []
        for doc in hits:
            # doc['_formatted'] может не всегда быть, поэтому используем .get(...)
            fmt = doc.get("_formatted", {})
            _id = doc.get("id")
            _file_name = doc.get("file_name", "не указан")
            _title = doc.get("title", "не указан")
            _page_number = doc.get("page_number", 1)
            content_str = fmt.get("content", "")

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


def create_index(index_uid: str, primary_key:str = "id") -> None:
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


def main():
    """
    Example usage. Adjust as needed.
    """

    print("=======")
    # res = meili_list_documents("main_index", return_type="All")
    # print(res)
    print(show_list_indexes("all"))
    # delete_index("try_0")
    # create_index("news")
    print("=======")
    print("=======")

    # s_r = search_meili("main_index", "уретрит")
    # print("=======")
    # print(s_r)


if __name__ == '__main__':
    main()
