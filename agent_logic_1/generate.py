from langchain_core.documents import Document

from agent_logic_2 import config as c
import time
from ollama import AsyncClient

ollama_aclient = AsyncClient(host=c.ollama_url)

# Выбор llm
llm = c.ll_model_small


# Post-processing
# def format_docs(docs):
#     """Convert Document to string
#         This option need for some functions inside async_graph_operator.py: generate_final
#     """
#     return "\n\n".join(doc.page_content for doc in docs)


async def chat(question: str, history: list = None):
    """
    Just chat with the short-term history
    """

    if history is None:
        history = []

    prompt = ('<|begin_of_text|><|start_header_id|>system<|end_header_id|> '
              'You are an assistant for question-answering tasks. Answer the question in plain text format. '
              'If you do not know the answer, simply state that you do not know. '
              'Use a maximum of five sentences and keep your answer concise. '
              'If relevant, reference the history of previous conversations in your answer. '
              'If needed, reason with the user about information found in the conversation history to provide clarity or address follow-up questions. '
              '<|eot_id|><|start_header_id|>user<|end_header_id|> '
              f'Question: {question}. \n\n'
              f'History of previous conversations: {history} \n\n'
              'Answer: '
              '<|eot_id|><|start_header_id|>assistant<|end_header_id|>')

    # async & .generate
    start_time = time.time()
    aresult = await ollama_aclient.generate(
        model=llm,
        prompt=prompt,
        # format="json",
        # options=opt,
        # keep_alive=-1,

    )

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Async request timing client-server is: {elapsed_time:.2f} sec")
    print(f"Eval_duration of answer generation: {aresult['eval_duration'] / 1_000_000_000}")
    #

    return aresult['response']


async def generate_answer(question: str, documents_in: list[Document], ) -> str:  # list[Document]:
    """
    Generate the final answer of the agent in a question-answering cycle.
    """
    # if history is None:
    #     history = []

    formatted_docs = "\n\n".join(
        [f"Document {i + 1}:\n{doc.page_content}" for i, doc in enumerate(documents_in)]
    )

    print("===================")
    print(question)
    print("===================")
    print(formatted_docs)
    print("===================")

    prompt = (f'<|begin_of_text|><|start_header_id|>system<|end_header_id|> '
              'You are an assistant tasked with answering user questions based on relevant context retrieved from a vector database. '
              'The context provided consists of documents found based on their similarity to the question. If multiple '
              'documents are available, select the most relevant one and generate your response based on that document. '
              'Use the information provided to create a clear and concise response in plain text.'
              'If you do not know the answer, simply respond with "I do not know" without any explanations!'
              # 'Limit your response to a maximum of six sentences.'
              # 'If previous conversation history exists, reference it in your answer. If there is just user question, you must ignore it. '
              # '<|eot_id|><|start_header_id|>user<|end_header_id|> '
              f'Question: {question}. \n\n'
              f'Context: \n\n'
              f' {formatted_docs}. \n\n'
              # f'History of previous conversations (if available): {history} \n\n'
              # 'Answer: '
              '<|eot_id|><|start_header_id|>assistant<|end_header_id|>')

    # async & .generate
    start_time = time.time()
    aresult = await ollama_aclient.generate(
        model=llm,
        prompt=prompt,
        # format="json",
        # options=opt,
        # keep_alive=-1,

    )

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Async request timing client-server is: {elapsed_time:.2f} sec")
    print(f"Eval_duration of answer generation: {aresult['eval_duration'] / 1_000_000_000}")
    #

    print("aresult['response']: ", aresult['response'])
    # print('aresult["context"]: ',aresult["context"])
    print('aresult["model"]: ', aresult["model"])

    return aresult['response']
