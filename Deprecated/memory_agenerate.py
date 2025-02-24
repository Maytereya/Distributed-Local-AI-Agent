from langchain_core.documents import Document

import time
from ollama import AsyncClient
from converters import json_converter as j
import config as c

ollama_aclient = AsyncClient(host=c.ollama_url)
# Выбор llm
llm = c.ll_model_big

# Post-processing
def format_docs(docs):
    """Convert Document to string
        This option need for some functions inside async_graph_operator.py: generate_final
    """
    return "\n\n".join(doc.page_content for doc in docs)


async def chat(question: str, history=None):
    """
Just chat with the short-term history
    """
    # if history is None:
    #     history = [{"role": "user", "content": "no history"}]
    if history is None:
        # Память!
        history = []
    prompt = ('<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for '
              'question-answering tasks. Answer the question in '
              'plain text format. If you do not know the answer, just say that you do not know. Use three sentences '
              'maximum and keep the answer concise. History of previous conversations should be referenced, '
              f'if applicable, and included in the answer. <|eot_id|><|start_header_id|>user<|end_header_id|> '
              f'Question: {question}. History of previous conversations: {history} Answer: '
              '<|eot_id|><|start_header_id|>assistant<|end_header_id|>')

    # async & .generate
    start_time = time.time()
    aresult = await ollama_aclient.generate(
        model=c.ll_model,
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


async def generate_answer(question: str, documents: list[Document], history=None) -> list[Document]:
    """
    Generate the answer if the agent
    Пока рекордсмен по продолжительности генерации...
    """
    # if history is None:
    #     history = [{"role": "user", "content": "no history"}]
    if history is None:
        history = []
    prompt = ('<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for '
              'question-answering tasks. Use the following pieces of retrieved context to answer the question in '
              'plain text format. If you do not know the answer, just say that you do not know. Use three sentences '
              'maximum and keep the answer concise. History of previous conversations should be referenced, '
              f'if applicable, and included in the answer.  <|eot_id|><|start_header_id|>user<|end_header_id|> '
              f'Question: {question}. Context: {documents}. History here: {history} Answer: '
              '<|eot_id|><|start_header_id|>assistant<|end_header_id|>')

    # async & .generate
    start_time = time.time()
    aresult = await ollama_aclient.generate(
        model=c.ll_model,
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
    print("aresult['response'] type is: ", type(aresult))
    return aresult['response']


async def grade(question: str, document: str):
    """
    Grade the relevance of the retrieved document.
    """
    prompt = (f'<|begin_of_text|><|start_header_id|>system<|end_header_id|> '
              'You are tasked with assessing the relevance of a retrieved document to a user’s question. '
              'If the document contains concepts or keywords closely related to the user’s question, consider it relevant. '
              'The assessment does not need to be overly strict—the aim is to filter out irrelevant documents, not to demand exact matches. '
              'Return a binary "yes" or "no" to indicate whether the document is relevant. '
              'Provide your answer as a JSON object with a single key "score" and no additional text. '
              'Example: {"score": "yes"} or {"score": "no"}.'
              '<|eot_id|><|start_header_id|>user<|end_header_id|> '
              f'Here is the retrieved document: \n\n{document} \n\n'
              f'Here is the user question: {question} \n\n'
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
    print(f"Eval_duration: {aresult['eval_duration'] / 1_000_000_000}")
    #
    json_result = j.str_to_json(aresult['response'])
    print("Grade retrieved response: " + str(json_result))

    return json_result


async def hallucinations_checker(documents, generation):
    """Hallucination Grader"""

    prompt = ("<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether an "
              "answer is grounded in / supported by a set of facts. Give a binary 'yes' or 'no' score to indicate "
              "whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON "
              'with a single key "score" and no preamble or explanation. '
              'Example#1: {"score": "yes"}, example#2: {"score": "no"}. '
              f'<|eot_id|><|start_header_id|>user<|end_header_id|> Here are the facts: \n\n {documents} \n\n Here is '
              f'the answer: \n\n {generation} \n\n <|eot_id|><|start_header_id|>assistant<|end_header_id|>')

    # async & .generate
    start_time = time.time()
    aresult = await ollama_aclient.generate(
        model=c.ll_model,
        prompt=prompt,
        # format="json",
        # options=opt,
        # keep_alive=-1,

    )

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Async request timing client-server is: {elapsed_time:.2f} sec")
    # print('Предыдущий результат: 227.32 секунд (LTE, MSK)')
    print(f"Eval_duration: {aresult['eval_duration'] / 1_000_000_000}")

    json_result = j.str_to_json(aresult['response'])
    print("Hallucinations checker response: " + str(json_result))
    return json_result


async def answer_grader(question: str, generation):
    """Answer Grader"""

    prompt = ('<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether an '
              'answer is useful to resolve a question. Give a binary score "yes" or "no" to indicate whether the '
              'answer is useful to resolve a question. Provide the binary score as a JSON with a single key "score" '
              'and no preamble or explanation. '
              'Example#1: {"score": "yes"}, example#2: {"score": "no"}. '
              '<|eot_id|><|start_header_id|>user<|end_header_id|> Here is the answer: '
              f'\n\n {generation} \n\n Here is the question: \n\n {question} \n\n '
              '<|eot_id|><|start_header_id|>assistant<|end_header_id|>')

    # async & .generate
    start_time = time.time()
    aresult = await ollama_aclient.generate(
        model=c.ll_model,
        prompt=prompt,
        # format="json",
        # options=opt,
        # keep_alive=-1,

    )

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Async request timing client-server is: {elapsed_time:.2f} sec")
    print(f"Eval_duration: {aresult['eval_duration'] / 1_000_000_000}")

    json_result = j.str_to_json(aresult['response'])
    print("Answer grader response: " + str(json_result))

    return json_result
