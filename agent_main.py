"""
V 3.0 (ex agent_main.py)
This module implements a state-based conversational agent as a class,
following the recommendations of DeepLearning.ai (Harris).
It is designed as version 6.0, referred to as "long logic," and introduces extended functionality for chat interactions.

Key Features:
1. **Stateful Chat Logic**:
   - Incorporates agent logic extended for chat-based applications.
   - Maintains chat history through the `history[]` variable for enhanced context awareness.
   - Note: The memory functionality of langGraph itself is not utilized in this implementation.

2. **Native Retriever Integration**:
   - Adds a native retriever layer on top of the Chroma DB vector database.
   - Enhances retrieval capabilities for more dynamic and precise responses.

3. **Ollama Embeddings**:
   - Ollama embeddings are not used in this configuration.

4. **Extended RAG Node Logic**:
   - Attempts to extend the reasoning capability of RAG (Retrieval-Augmented Generation) nodes using `i+` techniques.
   - Improves handling of complex queries and multistep reasoning.

"""

import asyncio
import copy
from pprint import pprint
# from dotenv import load_dotenv
from langgraph.graph import START, StateGraph, END
from typing import TypedDict, Optional, List
from langchain_core.documents import Document
import warnings

# agent logic pack modules:
from agent_logic_pack import aretrieve3 as retrieve, check, embedding_filtration as filtration, generate2 as generate, \
    formulate, search, routing2 as route

warnings.filterwarnings(
    "ignore", category=FutureWarning, module="transformers.tokenization_utils_base"
)

# _ = load_dotenv()


# import os

# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
# os.environ["LANGCHAIN_API_KEY"] = "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"


# ToDo: Проверить соответствие типов, чтобы не было List[str], List[Document] и Document там, где все д.б. List[Document]

class AgentState(TypedDict):
    """
        Represents the state of the agent during conversation.

        Attributes:
            question (str): The user's question.
            generation (str): The generated answer.
            web_search (str): Indicates whether to perform a web search ("yes" or "no").
            collection_name_1 (str): The name of the first collection for document retrieval.
            collection_name_2 (str): The name of the second collection for document retrieval.
            attempt_count (int): The number of attempts made to retrieve relevant documents.
            history (list): The chat history, including user and assistant messages.
            documents (Optional[List[Document]]): A list of retrieved documents.
        """
    question: str
    # enhanced_question: str
    # exact_word: str
    generation: str
    web_search: str
    collection_name_1: str  # Пробуем в связи с необходимостью использовать разные модели и разные способы добавления данных (рус/eng)
    collection_name_2: str
    # collection_name_3: str # Резерв
    attempt_count: int  # счетчик попыток обращения к RAG до выхода на Web Search
    history: list
    documents: Optional[List[Document]]  # С поправкой на ошибку несоответствия типа в модуле web_search


# Входная функция:
async def route_question(state: AgentState):
    """
    Routes the user's question to the appropriate module for further processing.

    Args:
        state (AgentState): The current state of the agent.

    Returns:
        str: The next node to call based on the data source.
    """

    print("---ROUTE QUESTION---")
    question = state["question"]
    print("Вопрос от пользователя: ", question)

    source = await route.route(question)

    if source["datasource"] == "web_search":
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "websearch"

    elif source["datasource"] == "vectorstore":
        print("---ROUTE QUESTION TO RAG CHAIN---")
        return "vectorstore"

    elif source["datasource"] == "chat":
        print("---ROUTE QUESTION TO CHAT WITH MEMORY---")
        return "chat"

    elif source["datasource"] == "exit":
        print("---ROUTE TO TERMINATE THE SESSION---")
        return "exit"


async def focus_question(state: AgentState):
    """
    Extracts the main focus of the user's question.

    Args:
        state (AgentState): The current state of the agent.

    Returns:
        dict: A dictionary with the focused question.
    """
    print("---FOCUS QUESTION---")
    question = state["question"]
    print("Вопрос от пользователя для фокусировки: ", question)
    enhanced_result = await formulate.extract_keyword(question)
    print("Сокращенная формулировка: ", enhanced_result)
    return {"question": enhanced_result}


async def extend_question(state: AgentState):
    """
    Reformulates and improves the user's question.

    Args:
        state (AgentState): The current state of the agent.

    Returns:
        dict: A dictionary with the extended question.
    """
    print("---FORMULATE QUESTION---")
    question = state["question"]
    print("User's question for reformulation: ", question)
    enhanced_result = await formulate.formulate(question)
    print("Enhanced formulation: ", enhanced_result)
    return {"question": enhanced_result}


async def retrieve_vs_1(state: AgentState):
    """
    Retrieves documents from Chroma vector store with high diversity.

    Args:
        state (AgentState): The current state of the agent.

    Returns:
        dict: Updated state with retrieved documents and the current question.
    """

    attempt_count = 1  # Инициализируем счетчик попыток обращения к RAG

    # state["collection_name"] = "txt-side-eff-cosine-distiluse-base-multilingual-cased-v1"

    print("Collection name: ", state["collection_name_1"])

    documents = retrieve.vs_query(existed_collection=state["collection_name_1"], question=state["question"],
                                  search_type="mmr", k=5, lambda_mult=0.25)

    print("---RETRIEVE FROM CHROMA Vector Store with high diversity of results---")
    question = state["question"]
    collection_name = state["collection_name_1"]
    print("Question: ", state["question"])
    print("Collection name: ", collection_name)
    # for document in documents:
    #     print(document.page_content[:200])

    return {"documents": documents, "question": question, "attempt_count": attempt_count}


async def retrieve_vs_2(state: AgentState):
    """
    Retrieve documents from Chroma vector store with lower diversity lambda_mult=0.85
    Args:
        state (dict): The current graph state
    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """

    documents = retrieve.vs_query(existed_collection=state["collection_name_1"], question=state["question"],
                                  search_type="mmr", k=5, lambda_mult=0.85)
    print("---RETRIEVE FROM VS with lower diversity---")
    question = state["question"]
    attempt_count = 2
    print("Вопрос: ", state["question"])
    # for document in documents:
    #     print(document.page_content)

    return {"documents": documents, "question": question, "attempt_count": attempt_count}


async def retrieve_db_3(state: AgentState):
    """
    Retrieve documents from Chroma Database using ____  model, chose for reserve attempt search
    Args:
        state (dict): The current graph state
    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """

    extracted_keyword = await formulate.extract_keyword(state["question"])

    documents = retrieve.query_collection(existed_collection=state["collection_name_2"], question=state["question"],
                                          model="distiluse", n_results=5, contains=extracted_keyword)
    print("---RETRIEVE FROM CHROMA DB---")
    question = state["question"]
    attempt_count = 3
    print("Вопрос: ", state["question"])
    for document in documents:
        print(document.page_content)

    return {"documents": documents, "question": question, "attempt_count": attempt_count}


# not async converted
def web_search(state: AgentState):
    """
    Performs a web search and appends results to the list of documents.

    Args:
        state (AgentState): The current state of the agent.

    Returns:
        dict: Updated state with web search results added to documents.
    """

    print("---TAVILY WEB SEARCH---")

    question = state["question"]
    documents = state["documents"]

    docs = search.web_search(question)

    web_results = Document(page_content=docs)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]

    print("Question: ", state["question"])
    print("Web search result: ", docs)

    return {"documents": documents, "question": question}


async def grade_documents(state: AgentState):
    """
    Determines whether the retrieved documents are relevant to the question.
    If any document is not relevant, we will set a flag to run web search.

    Args:
        state (dict): The current graph state.

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state.
    """
    print("---FILTRATE DOCUMENTS by Cosine Similarity---")
    filtered_documents = filtration.filtrate(state["question"], state["documents"])
    print("Количество отфильтрованных документов: ", len(filtered_documents))

    question = await formulate.formulate(state["question"])
    print("Formulated QUESTION: ", question)
    # documents = state["documents"]

    print("---CHECK DOCUMENT RELEVANCE TO formulated QUESTION---")
    # Score each doc
    scored_filtered_docs = []
    web_search = "yes"  # Устанавливаем значение "yes" по умолчанию

    if filtered_documents is not None:
        for d in filtered_documents:
            score = await check.grade(question=question, document=d.page_content)
            grade = score["score"]

            if grade.lower() == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                scored_filtered_docs.append(d)
                # Как только найден релевантный документ, устанавливаем web_search в "no"
                web_search = "no"
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")

    else:
        print("Document was not given, maybe because of connection error or total filtration")

    return {"documents": scored_filtered_docs, "question": question, "web_search": web_search}


# ! sync
def decide_to_generate(state: AgentState):
    """
    Определяет какое векторное хранилище будет на следующей итерации до тех пор, пока не будет достигнуто условие
    наибольшего числа итераций.
    Например, векторных хранилища/коллекций в одном хранилище = 3. Итерационный индекс в таком случае = 2 (начиная с 0).
    При достижении максимального значения индекса, поиск переходит к WebSearch, что означает, что в векторных хранилищах
    релевантной информации не найдено.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")

    web_search = state["web_search"]
    attempt_counter = state["attempt_count"]
    print("########################")
    print("decide_to_generate attempt_counter: ", attempt_counter)
    print("########################")

    if web_search == "yes" and attempt_counter == 1:  # Пока что захардкодим эту величину

        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            f"---DOCUMENTS ARE NOT RELEVANT TO QUESTION, "
            f"STARTING SEARCH NEXT COLLECTION. "
            f"Attempts: {attempt_counter}---"
        )
        return "retrieve_next_1"

    elif web_search == "yes" and attempt_counter == 2:  # Пока что захардкодим эту величину

        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            f"---DOCUMENTS ARE NOT RELEVANT TO QUESTION, "
            f"STARTING SEARCH NEXT COLLECTION. "
            f"Attempts: {attempt_counter}---"
        )
        return "retrieve_next_2"

    elif web_search == "yes" and attempt_counter == 3:

        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            f"---DOCUMENTS ARE NOT RELEVANT TO QUESTION, "
            f"STARTING WEB SEARCH. "
            f"Attempts: {attempt_counter}---"
        )
        return "websearch"

    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"


# !
async def generate_final(state: AgentState):
    """
    Generates an answer using RAG or web search results.

    Args:
        state (AgentState): The current state of the agent.

    Returns:
        dict: Updated state with the generated answer.
    """
    print("---GENERATE answer using RAG or WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]
    # generation: list[Document] = []
    generation: str = ""
    # history = state["history"]
    # print("==== документы, на основании которых будет генерация ======")
    # for document in documents:
    #     print(document)
    # print("==========")
    # Потенциальное избавление от циклических ссылок
    # history_copy = copy.deepcopy(history)

    try:
        # generation based on RAG or WEB_Search tool
        generation = await generate.generate_answer(question, documents, )
    except Exception as e:
        print("Error during generation: ", e)

    return {"documents": documents, "question": question, "generation": generation, }


async def chat(state: AgentState):
    """
    Generate chat

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---CHAT WITH U---")
    question = state["question"]
    history = state["history"]

    # Потенциальное избавление от циклических ссылок
    history_copy = copy.deepcopy(history)

    # chat generation
    generation = await generate.chat(question, history_copy)
    return {"question": question, "generation": generation, "history": history}


# !
async def grade_generation_v_documents_and_question(state: AgentState):
    """
    Checks if the generated answer is grounded in the retrieved documents and answers the question.

    Args:
        state (AgentState): The current state of the agent.

    Returns:
        str: The next node to call ("useful", "not useful", "not supported").
    """

    print("---CHECK HALLUCINATIONS---")

    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    # history = state["history"]
    grade: str = "no"
    try:
        score = await check.hallucinations_checker(documents, generation)
        grade = score["score"]

        # print("check.hallucinations_checker, score: ", score)
        print("check.hallucinations_checker, grade: ", grade)
    except Exception as e:
        print("Error in hallucination checker, check.hallucinations_checker: ", e)

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")

        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        try:
            score = await check.answer_grader(question, generation)
            grade = score["score"]
            # print("check.answer_grader, score: ", score)
            print("check.answer_grader, grade: ", grade)

        except Exception as e:
            print("Ошибка получения данных от check.answer_grader: ", e)

        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---Nothing found, RE-TRY---")
        return "not supported"


class Agent:
    """
    Represents an intelligent agent for handling user queries via a state graph.
    """

    def __init__(self, system=""):
        self.system = system
        graph = StateGraph(AgentState)

        graph.add_node("websearch", web_search)  # web search
        graph.add_node("focus", focus_question)  # выделение главного слова в формулировке запроса к базе данных
        # graph.add_node("extend", extend_question)  # переформулирование и улучшение вопроса
        graph.add_node("retrieve_vs_1", retrieve_vs_1)  # retrieve from Chroma vector store mmr only
        graph.add_node("retrieve_vs_2", retrieve_vs_2)  # retrieve from Chroma DB vector search
        graph.add_node("retrieve_db_3", retrieve_db_3)  # retrieve from Chroma DB vector search with sbert model
        graph.add_node("grade_documents", grade_documents)  # grade documents
        graph.add_node("generate", generate_final)  # generate
        graph.add_node("chat", chat)  # chat

        graph.add_conditional_edges(
            START,
            route_question,
            {
                "websearch": "websearch",
                # "vectorstore": "retrieve_vs_1",
                "vectorstore": "focus",
                "chat": "chat",
                "exit": END,
            },
        )
        graph.add_edge("focus", "retrieve_vs_1")  # добавляем ребро улучшения запроса
        graph.add_edge("retrieve_vs_1", "grade_documents")

        # добавим еще одну ветку, связанную с последовательным перебором коллекций перед переходом на
        # websearch
        graph.add_conditional_edges(
            "grade_documents",
            decide_to_generate,
            {
                "websearch": "websearch",
                "generate": "generate",
                "retrieve_next_1": "retrieve_vs_2",
                "retrieve_next_2": "retrieve_db_3",
            },
        )
        graph.add_edge("websearch", "generate")

        # усложним логику дополнительным ветвлением для поиска разными способами с двумя моделями.
        graph.add_edge("retrieve_vs_2", "grade_documents")
        graph.add_edge("retrieve_db_3", "grade_documents")

        graph.add_conditional_edges(
            "generate",
            grade_generation_v_documents_and_question,
            {
                "not supported": "generate",
                "useful": END,
                # "not useful": "websearch",
                "not useful": "retrieve_db_3",
            },
        )
        graph.add_edge("chat", END)

        self.graph = graph.compile()  #


async def agent_conversation(agent: Agent):
    """Handles user interaction with the agent."""

    # Инициализация пустой истории
    history = []
    # Первое название коллекции, которое следует передать в graph через окно загрузки
    name_of_collection_1: str = "25_10_2024_LaBSE-en-ru_pdf"
    name_of_collection_2: str = "23_10_2024_distiluse_txt"
    # attempt_counter: int = 0

    print("Начнем беседу с агентом. Введите ваш вопрос.")
    while True:
        question = input("You: ")
        if question.lower() in ["exit", "quit", "e", "q", "выход", "в"]:
            print("Goodbye!")
            break
        # Добавляем новый вопрос в историю
        history.append({"role": "user", "content": question})

        # Для каждого нового вопроса создаем состояние агента
        inputs = {"question": question, "history": history, "collection_name_1": name_of_collection_1,
                  "collection_name_2": name_of_collection_2,
                  }

        # Запускаем цикл агента для каждого вопроса
        result = await run_agent(agent, inputs)

        # Добавляем ответ агента в историю
        history.append({"role": "assistant", "content": result})

        print(f"Agent: {result}")


async def run_agent(agent: Agent, inputs):
    """Запуск агента для обработки вопроса и получения результата."""
    app = agent.graph

    # Асинхронная обработка графа
    async for output in app.astream(inputs):
        for key, value in output.items():
            pprint(f"Закончено выполнение узла: {key}")
    # Возвращаем сгенерированный ответ
    return value["generation"]


# Основная функция для запуска чат-бота
async def main():
    # Создаем агента
    agent = Agent()

    # Запускаем процесс взаимодействия с агентом
    await agent_conversation(agent)


# Запуск программы
if __name__ == "__main__":
    asyncio.run(main())
