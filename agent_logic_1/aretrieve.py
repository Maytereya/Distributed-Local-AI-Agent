# Async Retriever for Chroma DB v 3.1
# Connection section

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")           # снять при онлайне
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")     # снять при онлайне
import logging
import uuid
from typing import List, Optional
from typing import Literal

# <<< ВАЖНО >>> тяжёлые модули импортируем позже, внутри функций
# from langchain_huggingface import HuggingFaceEmbeddings
# from sentence_transformers import SentenceTransformer

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings, Collection
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from tenacity import retry, stop_after_attempt, wait_fixed  # Для автоматических ретраев

from agent_logic_2 import config as c
from agent_logic_1 import formulate, embedding_filtration

# Model loading for embeddings
# from InstructorEmbedding import INSTRUCTOR
# i_model = INSTRUCTOR('hkunlp/instructor-large')
# model_only = "cointegrated/LaBSE-en-ru"
# model_only = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2' # эффективность под вопросом
# model_only = 'sentence-transformers/LaBSE'
# model_only = "sentence-transformers/distiluse-base-multilingual-cased-v1"
# ==== Medical models =====
# model_only = "dmis-lab/biobert-v1.1"
# model_only = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
# ==== Russian models =====
# model_only = "ai-forever/sbert_large_nlu_ru"
#

# --------------------------------------
# Отключение предупреждений о грядущем
# --------------------------------------
# warnings.filterwarnings(
#     "ignore", category=FutureWarning, module="transformers.tokenization_utils_base"
# )

# --------------------------------------
# Функционал подключения к
# Chroma server c логами и задержкой на 10 сек.
# Для того чтобы успеть понять и отладить
# --------------------------------------

#  Initialize logging for connection tries
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Retry Decorator
@retry(stop=stop_after_attempt(5), wait=wait_fixed(3))
def connect_to_chroma():
    """
        Пытается подключиться к серверу ChromaDB с автоматическими повторами.

        Использует параметры retry:
        - до 5 попыток подключения,
        - пауза 3 секунды между попытками.

        :return: Инициализированный chromadb.HttpClient при успешном подключении.
        :raises: Исключение, если все попытки подключения завершились неудачей.
        """
    logger.info("🔄 Подключение к ChromaDB...")
    return chromadb.HttpClient(host=c.chroma_host, port=c.chroma_port)


# Инициализация Chroma - клиента
try:
    chroma_client = connect_to_chroma()
    logger.info("✅ Успешное подключение к ChromaDB!")
    print(f"Chroma host: {c.chroma_host}, Chroma port: {c.chroma_port}")
except Exception as e:
    logger.error(f"❌ Ошибка подключения к ChromaDB: {e}")


# -----------------------------
# Выбор модели для эмбеддинга
# -----------------------------

def choose_model(model: Literal["distiluse", "sbert", "instructor", "default"] = "default",
                 return_type: Literal["model", "name"] = "model"):
    """
        Выбирает и при необходимости загружает модель эмбеддингов.

        :param model: Ключ модели:
            - "distiluse"  -> sentence-transformers/distiluse-base-multilingual-cased-v1
            - "sbert"      -> ai-forever/sbert_large_nlu_ru
            - "instructor" -> hkunlp/instructor-xl
            - "default"    -> cointegrated/LaBSE-en-ru
        :param return_type:
            - "name": вернуть только имя модели (строкой);
            - "model": вернуть загруженный объект SentenceTransformer.
        :return: Либо строка с именем модели, либо объект модели (SentenceTransformer на CPU).
        """

    model_mapping = {
        "distiluse": "sentence-transformers/distiluse-base-multilingual-cased-v1",
        "sbert": "ai-forever/sbert_large_nlu_ru",
        "instructor": "hkunlp/instructor-xl",
        "default": "cointegrated/LaBSE-en-ru"
    }
    selected_model_name = model_mapping.get(model, model_mapping["default"])

    if return_type == "name":
        return selected_model_name

    # ленивый импорт и явный CPU
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(selected_model_name, device='cpu')



# -----------------------------------
# Класс сервисных функций для Chroma
# -----------------------------------

class ChromaService:
    """
    Сервисный класс для работы с ChromaDB: подключение, диагностика
    и базовые операции с коллекциями.
    """

    def __init__(self, host: str, port: int):
        self.chroma_client = chromadb.HttpClient(host=host, port=port)

    def info_chroma(self):
        """
       Выводит в консоль служебную информацию о ChromaDB:
        текущую версию и количество коллекций.
        """
        print("Chroma current version: " + str(self.chroma_client.get_version()))
        print("Collections count: " + str(self.chroma_client.count_collections()))
        # print("Chroma heartbeat: " + str(round(self.chroma_client.heartbeat() / 3_600_000_000_000, 2)), " hours")

    def reset_chroma(self):
        """
        Полностью очищает ChromaDB и сбрасывает системный кэш.
        Использовать с осторожностью: все данные будут удалены.
        """
        self.chroma_client.reset()
        self.chroma_client.clear_system_cache()

    def display_collections(self, output_format: Literal["list", "str"] = "list") -> List[str] | str:
        """
        Возвращает список коллекций, сохранённых в ChromaDB.

        :param output_format:
            - "list": вернуть список имён коллекций (List[str]);
            - "str":  вернуть одну строку с именами коллекций, разделёнными переводами строк.
        :return: Список имён коллекций или строка с перечислением имён.
        """
        list_col = self.chroma_client.list_collections()

        names_list = [i.name for i in list_col]  # list comprehensive

        if output_format == "str":
            result = "\n".join(names_list)  # Соединяем имена в одну строку с переносами строк
        else:
            result = names_list

        return result


    def preconditioning(self, target_name: str):
        """
        Проверяет наличие коллекции с указанным именем и при необходимости удаляет её.

        Используется как подготовительный шаг перед созданием новой коллекции
        с тем же именем.

        :param target_name: Имя коллекции, которую требуется сбросить/подготовить.
        """
        # Имя коллекции = target_name
        list_col = self.chroma_client.list_collections()
        found = False
        for col in list_col:
            # Convert the object to string and extract the name value
            name_part = col.name.rstrip(")")

            if name_part == target_name:
                found = True
                break

        if found:
            print(f"Collection with name '{target_name}' exists, we'll delete it")
            self.chroma_client.delete_collection(target_name)
        else:
            print(f"Collection name '{target_name}' does not exist, we'll create it on the next step.")


# --------------------------------------------------------
# Альтернативная эмбеддинговая функция для русского языка
# --------------------------------------------------------

class HuggingFaceEmbeddingFunction(EmbeddingFunction[Documents]):
    """
    Обёртка для использования моделей SentenceTransformers в качестве
    embedding_function для ChromaDB.

    Поддерживает ленивую инициализацию модели и выбор пресетов
    (distiluse / sbert / instructor / default).
    """

    def set_model(self, model: Literal["distiluse", "sbert", "instructor", "default"] = "default"):
        """
        Задаёт пресет модели эмбеддингов для последующих вызовов.
        Модель фактически будет загружена только при первом вызове __call__.
        :param model: Ключ пресета модели ("distiluse", "sbert", "instructor", "default").
        """
        self._model_name = model
        self._model = None

    def __call__(self, input: Documents) -> Embeddings:
        """
        Вычисляет эмбеддинги для списка документов/строк.

        При первом вызове лениво загружает модель, заданную через set_model()
        (или "default", если set_model не вызывалась).

        :param input: Список строк (Documents), для которых нужно получить эмбеддинги.
        :return: Список эмбеддингов (List[List[float]]).
        """
        if not hasattr(self, '_model') or self._model is None:
            # реальная инициализация только на первый вызов
            self._model = choose_model(getattr(self, '_model_name', 'default'))
        return self._model.encode(input, show_progress_bar=True).tolist()



# ----------------------------------
# Загрузчики и сплиттер
# ----------------------------------

def web_txt_splitter(add_urls) -> List[Document]:
    """
    Загружает веб-страницы по списку URL и разбивает их содержимое на чанки.

    :param add_urls: Список URL, откуда нужно загрузить текст и разбить его на фрагменты.
    :return: Список объектов Document с разбитым текстом.
    """

    doc_splits: List[Document] = []
    if add_urls:  # Ensure the URL list is not empty
        docs = []
        for url in add_urls:
            loaded_docs = WebBaseLoader(url).load()
            if loaded_docs:
                docs.append(loaded_docs)

        # Flatten the list of loaded documents
        docs_list = [item for sublist in docs for item in sublist]

        if docs_list:  # Ensure the document list is not empty
            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=1000, chunk_overlap=100
            )
            doc_splits = text_splitter.split_documents(docs_list)
            print("Web document splitting done.")
        else:
            print("Loaded documents are empty.")
    else:
        print("No web documents passed.")

    return doc_splits


def txt_loader(path: str = "Upload/") -> List[Document]:
    """
    Загружает текстовые файлы из директории и разбивает их содержимое на чанки.

    :param path: Путь к директории, содержащей .txt-файлы.
    :return: Список объектов Document с разбитым текстом.
    """

    split_docs: List[Document] = []
    text_loader_kwargs = {"autodetect_encoding": True}

    loader = DirectoryLoader(path=path, glob="**/*.txt", loader_cls=TextLoader,
                             show_progress=True,
                             loader_kwargs=text_loader_kwargs)
    docs = loader.load()

    if docs:  # Ensure the list is not empty
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=3500, chunk_overlap=800
        )
        split_docs = text_splitter.split_documents(docs)
        print("Text document splitting done.")
    else:
        print("No text documents for splitting found in the specified path.")

    return split_docs


def pdf_loader(path: str) -> List[Document]:
    """
    Загружает PDF-документ постранично, сохраняя метаданные о номере страницы и пути.

    :param path: Путь к PDF-файлу.
    :return: Список объектов Document, по одному на страницу.
    """

    docs = []
    loader = PyPDFLoader(path)
    docs_lazy = loader.lazy_load()

    # Step counter
    step = 0

    for doc in docs_lazy:
        step += 1  # Increment step for each page processed
        print(f"\rStep {step}: Processing page...", end='', flush=True)
        docs.append(doc)

    print("\nAll PDF pages have been processed.")

    return docs


# -----------------------------------
# Работа с коллекциями
# -----------------------------------

def handle_collection(existed_collection: str) -> List[str] | str:
    """
    Получает информацию о существующей коллекции ChromaDB и возвращает
    список файлов с указанием страницы.

    :param existed_collection: Имя существующей коллекции.
    :return: Список строк вида "<имя_файла>, p.<номер_страницы>" или
             сообщение о том, что коллекция/документы недоступны.
    """
    try:
        collection = chroma_client.get_collection(name=existed_collection,
                                                  embedding_function=HuggingFaceEmbeddingFunction())
        if collection.count() == 0:
            return ["Коллекция не содержит документов"]

        # print("Common collection info:")
        peek = collection.peek(limit=300)  # returns a list of the first 300 items in the collection

        # Only get documents and ids
        # collection_info = collection.get(
        #     include=["uris"],
        # )

        documents_metadata = peek["metadatas"]
        if not documents_metadata:
            return ["Документы в коллекции не найдены"]

        # for metadata in documents_metadata:
        # print(documents_metadata)

        file_list = []
        for item in documents_metadata:
            page_num = item["page"]
            file_path = item["source"]
            file_name = os.path.basename(file_path)
            file_list.append(f"{file_name}, p.{page_num}")


        return file_list if file_list else ["Не найдено подходящих документов"]

    except Exception as e:
        return ["Доступ к коллекции невозможен"]


def create_collection(
        new_collection_name: str,
        model: Literal["distiluse", "sbert", "default"] = "default"
) -> Optional[Collection]:
    """
    Создаёт новую коллекцию в ChromaDB с указанным именем и моделью эмбеддингов.

    :param new_collection_name: Имя создаваемой коллекции.
    :param model: Выбранная модель эмбеддингов для коллекции.
                  "default" — LaBSE-en-ru, а также предопределённые варианты "sbert", "distiluse".
    :return: Объект Collection при успешном создании, иначе None.
    """
    embedding_function = HuggingFaceEmbeddingFunction()
    embedding_function.set_model(model)  # Set the embedding model.

    try:
        chroma_collection = chroma_client.create_collection(
            name=new_collection_name,
            embedding_function=embedding_function,
            metadata={"hnsw:space": "cosine"})  # Use cosine similarity as the default metric.

        if chroma_collection:
            print(f"Collection with name {new_collection_name} successfully created.")
            return chroma_collection

        else:
            print(f"Failed to create collection: {new_collection_name}")
            return None
    except Exception as ecc:
        print(f"An error occurred while creating the collection: {ecc}")
        return None


def remove_collection(collection_name: str, ):
    """
        Удаляет коллекцию из ChromaDB по имени.

        :param collection_name: Имя коллекции, которую необходимо удалить.
        :return: None. В случае ошибки выводит сообщение в консоль.
        """
    try:
        chroma_client.delete_collection(name=collection_name)
    except Exception as erc:
        print(f"An error occurred while deleting the collection: {erc}, stop |")
        return None


def add_data(
        exist_collection_name: str,
        upload_type: Literal["URL", "PDF", "TXT"],
        add_urls: List[str] | None = None,
        add_path: str | None = None,
        model: Literal["distiluse", "sbert", "instructor", "default"] = "default"
):
    """
    Добавляет данные в существующую коллекцию ChromaDB в зависимости от типа источника.

    :param exist_collection_name: Имя существующей коллекции, в которую будут загружены данные.
    :param upload_type: Тип загружаемых данных:
        - "URL": загрузить текст по списку URL;
        - "PDF": загрузить содержимое из одного PDF-файла;
        - "TXT": загрузить содержимое из текстовых файлов в директории.
    :param add_urls: Список URL (используется, если upload_type = "URL").
    :param add_path: Путь к файлу или директории (используется для "PDF" и "TXT").
    :param model: Модель эмбеддингов ("default" / "sbert" / "instructor" / "distiluse").
    :return: None. Сообщения о ходе и ошибках выводятся в консоль.
    """

    print(f"Adding data to collection: {exist_collection_name}")
    docs = []
    print(f"{add_path=}")
    print(type(add_path))

    try:
        # Fetch documents
        if upload_type == "URL" and add_urls is not None:
            print("Loading documents from URLs...")
            docs = web_txt_splitter(add_urls)
            if not docs:
                print("No documents found at the specified URLs.")
                return

        elif upload_type == "PDF" and add_path is not None:
            print("Loading PDF document...")
            docs = pdf_loader(add_path)
            if not docs:
                print("No content found in the PDF file.")
                return

        elif upload_type == "TXT" and add_path is not None:
            print("Loading TXT documents...")
            docs = txt_loader(add_path)
            if not docs:
                print("No text files found in the specified directory.")
                return

        else:
            print("No valid data provided for upload.")

        # Set embedding function
        embedding_function = HuggingFaceEmbeddingFunction()
        embedding_function.set_model(model)

        # Fetch the collection
        collection = chroma_client.get_collection(name=exist_collection_name,
                                                  embedding_function=embedding_function)
        if not collection:
            print(f"Collection {exist_collection_name} not found.")
            return

        total_chunks = len(docs)
        print(f"Total chunks: {total_chunks}")

        for i, doc in enumerate(docs, start=1):
            print(f"\rProcessing chunk: {i}/{total_chunks}", end="", flush=True)
            print()
            print(f"IDs: {str(uuid.uuid1())}")
            print(f"Metadata: {doc.metadata}")
            print(f"Type: {doc.type}")
            print(f"Content preview: {doc.page_content[:10]}")
            try:

                # Add documents to the collection
                collection.add(
                    ids=[str(uuid.uuid1())],
                    metadatas=doc.metadata,
                    documents=doc.page_content
                )
            except Exception as e:
                print(f"Failed to process document {i}/{total_chunks}: {e}")

        print()  # Print an empty line after processing all chunks
    except Exception as e:
        print(f"An error occurred while adding data: {e}")


# ------------------
# :: Chroma DB ::
# ------------------

def query_collection(
        existed_collection: str,
        question: str,
        contains: str = " ",
        n_results: int = 2,
        model: Literal["distiluse", "sbert", "instructor", "default"] = "default"
) -> Optional[List[Document]]:
    """
    Выполняет запрос к коллекции ChromaDB и возвращает подходящие документы.

    :param existed_collection: Имя коллекции, по которой выполняется поиск.
    :param question: Текст запроса (используется для эмбеддинга).
    :param contains: Фильтр по содержимому документа (where_document, оператор "$contains").
    :param n_results: Количество возвращаемых результатов.
    :param model: Модель эмбеддингов, используемая при поиске.
    :return: Список объектов Document с найденными фрагментами или None в случае ошибки.
    """
    embedding_function = HuggingFaceEmbeddingFunction()
    embedding_function.set_model(model)

    collection = chroma_client.get_collection(name=existed_collection,
                                              embedding_function=embedding_function)

    result = collection.query(
        query_texts=question,
        n_results=n_results,
        # where={"metadata_field": "is_equal_to_this"},
        where_document={"$contains": contains}  # Filter documents containing the specified text.
    )


    documents = [
        Document(page_content=doc, metadata=meta)
        for sublist in result['documents']
        for doc in sublist
        for sub_metadata in result["metadatas"]
        for meta in sub_metadata
    ]
    # Но можно возвращать List[str], str или Document при необходимости.
    return documents


# --------------------------
# :: Chroma Vector Store ::
# --------------------------

def vs_query(
        existed_collection: str,
        question: str,
        search_type: Literal["simil", "simil_score", "vector", "mmr"],
        model: Literal["distiluse", "sbert", "instructor", "default"] = "default",
        k: int = 3,
        filters: dict = None,
        fetch_k: int = 25,
        lambda_mult: float = 0.85
) -> List[Document]:
    """
    Выполняет векторный поиск по коллекции Chroma через LangChain VectorStore.

    :param existed_collection: Имя коллекции для поиска.
    :param question: Текст запроса.
    :param search_type: Тип поиска:
        - "simil": обычный similarity search;
        - "simil_score": similarity search с оценками похожести;
        - "vector": поиск по вручную рассчитанному вектору;
        - "mmr": поиск с Maximal Marginal Relevance (MMR).
    :param model: Модель эмбеддингов для построения векторов.
    :param k: Количество возвращаемых документов.
    :param filters: Фильтры для поиска по метаданным
                    (по умолчанию {"source": "pdf/side_effects_guidelines.pdf"}).
    :param fetch_k: Количество документов, выбираемых до MMR-отбора.
    :param lambda_mult: Коэффициент баланса между релевантностью и разнообразием
                        для MMR (чем ниже — тем выше упор на релевантность).
    :return: Список объектов Document с результатами поиска (может быть пустым при ошибке).
    """

    documents: List[Document] = []

    try:
        # ленивый импорт, чтобы не инициализировать transformers на импорт файла
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_chroma import Chroma

        model_name = choose_model(model, return_type="name")
        embedding_function = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},  # ключевая строчка
            encode_kwargs={"normalize_embeddings": True}
        )

        vector_store_from_client = Chroma(
            client=chroma_client,
            collection_name=existed_collection,
            embedding_function=embedding_function,
        )

        # Default filter if none provided
        if filters is None:
            filters = {"source": "pdf/side_effects_guidelines.pdf"}

        def print_results(all_results):
            for c_res in all_results:
                if isinstance(c_res, tuple):
                    c_res, score = c_res
                    print(f"* [SIM={score:.3f}] {c_res.page_content} [{c_res.metadata}]")
                else:
                    print(f"* {c_res.page_content} [{c_res.metadata}]")

        if search_type == "simil":
            documents = vector_store_from_client.similarity_search(
                question,
                k=k,
                filter=filters,
            )
            print("Search type: similarity search")
            # print_results(documents)

        elif search_type == "simil_score":
            results = vector_store_from_client.similarity_search_with_score(
                question,
                k=k,
                filter=filters,
            )
            print("Search type: similarity search with scores")
            # print_results(results)
            documents = [
                Document(page_content=doc.page_content, metadata=score)
                for doc, score in results
            ]

        elif search_type == "vector":
            documents = vector_store_from_client.similarity_search_by_vector(
                embedding=embedding_function.embed_query(question), k=k
            )
            print("Search type: search by vector")
            # print_results(documents)

        elif search_type == "mmr":
            retriever = vector_store_from_client.as_retriever(
                search_type="mmr",
                search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult, "filters": filters}
            )
            print("Search type: Maximal Marginal Relevance (MMR)")
            documents = retriever.invoke(question, )
            # print_results(documents)

    except Exception as e:
        print(f"An error occurred during vector store search using def vs_query(): {e}")

    return documents


# ------------------------------------
# Main retrieve function, include all
# ------------------------------------

async def main_retrieve_async(collection: str,
                              question: str,
                              search_type: Literal["vectorstore", "db"] = "vectorstore",
                              return_type: Literal["list", "str"] = "list",
                              k: int = 5,
                              n_results: int = 2,
                              threshold: float = 0.005,
                              ) -> List[Document] | str:
    """
    Основная асинхронная функция для поиска по Chroma (через VectorStore или напрямую по DB)
    с фильтрацией похожих документов.

    :param collection: Имя коллекции, в которой выполняется поиск.
    :param question: Вопрос пользователя (исходный текст запроса).
    :param search_type:
        - "vectorstore": поиск через Chroma VectorStore (LangChain) с MMR;
        - "db": прямой запрос к ChromaDB (collection.query) с фильтром по ключевому слову.
    :param return_type:
        - "list": вернуть список объектов Document;
        - "str": вернуть конкатенированный текст всех найденных документов.
    :param k: Количество документов в выдаче при поиске через VectorStore.
    :param n_results: Количество документов в выдаче при прямом поиске по ChromaDB.
    :param threshold: Порог отсечения сильно похожих документов при пост-фильтрации
                      (чем больше значение, тем больше документов остаётся).
    :return: Либо список Document, либо строка с объединённым содержимым
             (в зависимости от параметра return_type).
             При отсутствии совпадений возвращается заглушка
             с текстом "Совпадений не найдено, cформулируйте запрос иначе".
    """
    if not collection:
        return (
            "Ошибка: не выбрана коллекция, в которой следует осуществлять поиск"
        )
    print("Current info about Chroma:")
    chr_service = ChromaService(c.chroma_host, c.chroma_port)
    chr_service.info_chroma()
    print(f"Query to collection: {collection}")
    print("Question:", question)

    async def keyword_from_question(question_in):
        return await formulate.extract_keyword(question_in)

    keyword = await keyword_from_question(question)

    if search_type == "vectorstore":
        documents = vs_query(collection, question=question, search_type="mmr", k=k, )
        # Need two-step filtration
        print("vectorstore search type enabled")
        filtrated_docs = embedding_filtration.filtrate(keyword, documents, threshold=threshold)
    else:
        # Already filtrated in chroma_db algorythm
        print("chroma_db search type enabled")
        filtrated_docs = query_collection(collection, question, contains=keyword, n_results=n_results,
                                          model="default")


    if len(filtrated_docs) == 0:
        filtrated_docs = [
            Document(
                metadata={"source": "neiry-ai.ru"},
                page_content="Совпадений не найдено, cформулируйте запрос иначе",
            )]

    if return_type == "str":
        result_str: str = '\n'.join(
            '\n'.join(f_doc.page_content) if isinstance(f_doc.page_content, list)
            else f_doc.page_content
            for f_doc in filtrated_docs
        )
        return result_str

    return filtrated_docs


def main_add_to_chroma(
        # filename: str = "side_effects_guideline_for_RAG_paged.pdf",
        path_to_file: str = None,
        collection: str = "main_collection",
        doc_type: Literal["URL", "PDF", "TXT"] = "PDF",
) -> None:
    """
    Высокоуровневая функция для загрузки документов в коллекцию ChromaDB.

    Создаёт (при необходимости) и/или использует существующую коллекцию,
    после чего добавляет в неё данные указанного типа.

    :param path_to_file: Путь к файлу или директории, откуда загружаются данные.
                         Для "PDF" — PDF-файл, для "TXT" — директория с .txt-файлами.
                         Для "URL" путь обрабатывается внутри add_data (используются URL-списки).
    :param collection: Имя коллекции, в которую будут добавлены документы.
    :param doc_type: Тип загружаемого источника: "URL", "PDF" или "TXT".
    :return: None.
    """

    chroma_service = ChromaService(c.chroma_host, c.chroma_port)
    chroma_service.info_chroma()
    add_data(exist_collection_name=collection, upload_type=doc_type,
             add_path=path_to_file, )

    return None


# Test section
#
if __name__ == '__main__':
    print(':: TESTING ::')

    # PDF document to load pass
    # file_path = "pdf/taking_guidelines.pdf"
    # txt document directory pass
    # file_path = "../Upload/"

    # urls_rus = [
    #     "https://neiro-psy.ru/blog/monopobiya-kak-nazyvaetsya-strah-ostavatsya-odnomu-i-kak-s-nim-spravitsya",
    #     "https://neiro-psy.ru/blog/bipolyarnoe-rasstrojstvo-i-depressiya-ponimanie-razlichij",
    #     "https://neiro-psy.ru/blog/razdvoenie-lichnosti-kak-raspoznat-simptomy-i-obratitsya-za-pomoshchyu",
    # ]

    cs = ChromaService(c.chroma_host, c.chroma_port)
    collections = cs.display_collections(output_format="list")
    print(collections)
    disp = handle_collection("main_collection")
    print(disp)
