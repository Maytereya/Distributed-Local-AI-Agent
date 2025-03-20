"""
v0.2

This module provides two ways to work with Meilisearch:
1) Using the official Python client for adding documents and searching them.
2) Using raw HTTP requests for listing indexes, listing documents,
   and removing them (as well as deleting entire indexes).

"""

import json

import requests
import meilisearch
# Retry section
# import time
import logging
# from httpx import AsyncClient, ConnectError
from tenacity import retry, stop_after_attempt, wait_fixed  # Для автоматических ретраев
#
import config as c

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


# client = meilisearch.Client(c.MEILI_URL, c.MASTER_KEY)


def add_doc_to_meili(doc_path: str, index_name: str) -> None:
    """
    Adds JSON documents from a local file to a Meilisearch index.
    If the index does not exist, it will be created automatically.

    :param doc_path: Path to the local JSON file containing the documents (list of dicts).
    :param index_name: Name of the target Meilisearch index.
    :return: None
    """
    try:
        with open(doc_path, mode="r", encoding="utf-8") as json_file:
            documents = json.load(json_file)
    except (IOError, json.JSONDecodeError) as e:
        print(f"Error reading/parsing JSON file '{doc_path}': {e}")
        return

    try:
        task_info = client.index(index_name).add_documents(documents)
        print(f"Documents have been enqueued for addition to index '{index_name}': {task_info}")
    except Exception as e:
        print(f"Error adding documents to index '{index_name}': {e}")


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
    except Exception as e:
        print(f"Error retrieving info for task #{task_number}: {e}")
        return {}


def search_meili(index_name: str, query: str, limit: int = 2,
                 highlight: str = None, highlight_fields: str = '*') -> str:
    """
    Performs a search on the given Meilisearch index using the Python client.

    :param index_name: Name of the Meilisearch index.
    :param query: The search query (keyword, phrase, etc.).
    :param limit: Maximum number of results to return.
    :param highlight: A tag to wrap around highlights (e.g. <em></em>).
    :param highlight_fields: Which fields to highlight. By default, '*'.
    :return: A dictionary of search results, as returned by Meilisearch.
    """
    try:
        search_result = client.index(index_name).search(query, {
            "limit": limit,
            # "highlightPreTag": highlight,
            # "highlightPostTag": highlight,
            "attributesToHighlight": [highlight_fields],
        })
        print("Search results:", search_result)

        hits = search_result.get("hits", [])

        # Собираем все куски контента:
        contents = []
        for doc in hits:
            # doc['_formatted'] может не всегда быть, поэтому используем .get(...)
            fmt = doc.get("_formatted", {})
            content_str = fmt.get("content", "")
            contents.append(content_str)

        # Склеиваем их в итоговую строку
        combined_text = "\n-----\n".join(contents)
        if len(combined_text) == 0:
            combined_text = "Совпадений не найдено, cформулируйте запрос иначе"
        return combined_text

    except Exception as e:
        print(f"Error searching in index '{index_name}': {e}")
        return "Ошибка поисковой системы meilisearch"


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
        response = requests.get(endpoint, headers=headers, timeout=10)
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

def create_index(index_uid: str) -> None:
    """
    Создаёт индекс в Meilisearch по обозначенному UID.

    :param index_uid: Уникальный идентификатор (UID) для нового индекса.
    :return: None
    """
    endpoint = f"{c.MEILI_URL}/indexes"
    headers = {"Authorization": f"Bearer {c.MASTER_KEY}"}
    payload = {
        "uid": index_uid
    }
    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=10)
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


def list_documents(index_uid: str, limit: int = 20, offset: int = 0) -> list:
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
            data = response.json()
            documents = data.get("results", [])
            print(f"Documents in index '{index_uid}':")
            print(json.dumps(data, indent=2, ensure_ascii=False))
            return documents
        else:
            print(f"Error listing documents for index '{index_uid}': {response.text}")
            return []
    except requests.exceptions.RequestException as e:
        print(f"Error listing documents for index '{index_uid}': {e}")
        return []


def delete_document(index_uid: str, doc_id: str) -> None:
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
    # Example: add documents to an index
    # path_to_doc = "/path/to/side_effects_guideline_list1.json"
    # add_doc_to_meili(path_to_doc, "side_effects_improved")

    # List indexes
    # indexes = show_list_indexes()
    # print(indexes)

    # List documents (limit=2)
    # docs = list_documents("side_effects_improved", limit=2)
    # print(docs)

    # Search in Meilisearch
    # result = search_meili("side_effects_improved", "headache", limit=3)
    # print("Search result:", result)

    # Delete a specific document
    # delete_document("side_effects_improved", "side_effects_guideline_for_RAG_paged_pdf_page_1")

    # Delete an entire index
    # delete_index("side_effects_improved")


if __name__ == '__main__':
    main()
