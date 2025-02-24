import time
from ollama import AsyncClient
from converters import json_converter as j
import config as c

ollama_aclient = AsyncClient(host=c.ollama_url)


# Post-processing
def format_docs(docs):
    """Convert Document to string
        This option need for some functions inside async_graph_operator.py: generate_final
    """
    return "\n\n".join(doc.page_content for doc in docs)


async def generate_answer(question: str, documents):
    """
    Generate the answer if the agent
    Пока рекордсмен по продолжительности генерации...
    """
    prompt = ('<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for '
              'question-answering tasks. Use the following pieces of retrieved context to answer the question in '
              'plain text format. If you do not know the answer, just say that you do not know. Use three sentences '
              'maximum and keep the answer concise <|eot_id|><|start_header_id|>user<|end_header_id|> '
              f'Question: {question}. Context: {documents}. Answer: '
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


async def grade(question: str, document: str):
    """
    Grade the relevance of the retrieved document.
    """
    prompt = ('<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing relevance of a '
              'retrieved document to a user question. If the document contains keywords related to the user question, '
              'grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous '
              'retrievals. Give a binary score "yes" or "no" score to indicate whether the document is relevant to '
              'the question. Provide the binary score as a JSON with a single key "score" and no preamble or explanation.'
              'Example#1: {"score": "yes"}, example#2: {"score": "no"}.'
              '<|eot_id|><|start_header_id|>user<|end_header_id|> Here is the retrieved document: \n\n'
              f'{document} \n\n'
              f'Here is the user question: {question} \n\n'
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
    # print('Предыдущий результат: 40.06 секунд (LTE, MSK)')
    print(f"Eval_duration: {aresult['eval_duration'] / 1_000_000_000}")

    json_result = j.str_to_json(aresult['response'])
    print("Answer grader response: " + str(json_result))

    return json_result
