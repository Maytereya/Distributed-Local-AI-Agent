# Check hallucinations module v 1.0

import json
from langchain_core.documents import Document
from typing import Dict, Optional
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

import config as c

import time
from ollama import AsyncClient, Options
from langchain_ollama import \
    ChatOllama  # Used for managing temperature and JSONOutputParser
from converters import json_converter as j

# Initialize Ollama async client
ollama_aclient = AsyncClient(host=c.ollama_url)

# Setting LLM options
# options = Options(temperature=0, num_ctx=128000)  # Higher context length, might slow performance
options = Options(temperature=0, )  # Default settings for generation

# Selecting the model to use
llm = c.ll_model_small


async def grade(question: str, document: str) -> Optional[Dict[str, str]]:
    """
    Assess the relevance of a retrieved document to a user's question.

    The function evaluates whether the given document is relevant to the user's question.
    It returns the result as a JSON object with a single key "score" having the value "yes" or "no".

    :param question: The user's question to evaluate against.
    :param document: The retrieved document to assess.
    :return: A JSON object indicating the relevance of the document.
    """

    prompt = (f'''<|begin_of_text|><|start_header_id|>system<|end_header_id|>
              You are tasked with assessing the relevance of a retrieved document to a user’s question.
              If the document contains concepts or keywords closely related to the user’s question, consider it relevant.
              The assessment does not need to be overly strict. 
              The aim is to filter out irrelevant documents, not to demand exact matches.
              Return a binary "yes" or "no" to indicate whether the document is relevant.
              Provide your answer as a JSON object with a single key "score" and no additional text.
              Example: {{"score": "yes"}} or {{"score": "no"}}.
              Here is the retrieved document: \n\n{document} \n\n
              <|eot_id|><|start_header_id|>user<|end_header_id|>
              Here is the user question: {question} \n\n
              <|eot_id|><|start_header_id|>assistant<|end_header_id|>''')

    # async & .generate
    start_time = time.time()
    aresult = await ollama_aclient.generate(
        model=llm,
        prompt=prompt,
        format="json",
        options=options,
        # keep_alive=-1,

    )

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Async request timing client-server is: {elapsed_time:.2f} sec")
    print(f"Eval_duration: {aresult['eval_duration'] / 1_000_000_000}")
    #
    json_result = j.str_to_json(aresult['response'])

    print("Module Check. Grade retrieved response: " + str(json_result))

    return json_result


async def hallucinations_checker(documents_in: list[Document], generation: str) -> Optional[Dict[str, str]]:
    """
    Check if the generated answer is grounded in the provided facts.

    This function evaluates whether the generated response is supported by a list of documents.
    It returns a JSON object with a single key "score" having the value "yes" or "no".

    :param documents_in: A list of Document objects to evaluate against.
    :param generation: The generated response to verify.
    :return: A JSON object indicating whether the answer is grounded in the provided facts.
    """

    # prompt = ('<|begin_of_text|><|start_header_id|>system<|end_header_id|> '
    #           'You are a grader assessing whether an answer is grounded in / supported by a set of facts, which are in Russian. '
    #           'You must provide your answer as a JSON object with a single key "score". Do not provide any explanations, translations, reformulations, or additional information. '
    #           'If the answer is supported by the Russian text, return {"score": "yes"}. If it is not, return {"score": "no"}. '
    #           'Do not include any other content besides the JSON object. '
    #           '<|eot_id|><|start_header_id|>user<|end_header_id|> '
    #           f'Here is the user question: \n\n{question} \n\n'
    #           f'Here are the provided facts (use only Russian text for evaluation): \n\n{documents} \n\n'
    #           f'Here is the generated answer: \n\n{generation} \n\n'
    #           '<|eot_id|><|start_header_id|>assistant<|end_header_id|>')

    formatted_docs = "\n\n".join(
        [f"Document {i + 1}:\n{doc.page_content}" for i, doc in enumerate(documents_in)]
    )

    prompt = (f'''<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether
              an answer is grounded in / supported by a set of facts. Give a binary "yes" or "no" score to indicate
              whether the answer is grounded in / supported by a set of facts. 
              Provide the binary score as a JSON with a single key "score" and no preamble or explanation.
              If the answer is supported by the set of facts, return {{"score": "yes"}}. If it is not,
              return {{"score": "no"}}.
              Here are the facts:
              \n ------- \n
              {formatted_docs}
              \n ------- \n
              Here is the answer:
              \n ------- \n
              {generation} <|eot_id|><|start_header_id|>assistant<|end_header_id|>''')

    # async & .generate
    start_time = time.time()
    aresult = await ollama_aclient.generate(
        model=llm,
        prompt=prompt,
        format="json",
        options=options,
        # keep_alive=-1,

    )

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Async request timing client-server is: {elapsed_time:.2f} sec")
    # print('Предыдущий результат: 227.32 секунд (LTE, MSK)')
    print(f"Eval_duration: {aresult['eval_duration'] / 1_000_000_000}")

    json_result = j.str_to_json(aresult['response'])
    print("Module Check. Hallucinations checker response: " + aresult['response'])

    # Handle cases where response length is unexpected
    if len(json.dumps(json_result)) != 16:
        json_result = j.str_to_json('{"score": "no"}')

    return json_result


async def hallucinations_checker_v2(documents, generation):
    """
    A variant of hallucinations_checker using ChatOllama and JsonOutputParser.

    :param documents: A string containing facts to evaluate against.
    :param generation: The generated response to verify.
    :return: None (directly uses the prompt to check hallucination grading).
    """

    loc_llm = ChatOllama(model=llm, format="json", temperature=0, )

    # Prompt
    prompt = PromptTemplate(
        template=""" <|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether 
        an answer is grounded in / supported by a set of facts. Give a binary 'yes' or 'no' score to indicate 
        whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a 
        single key 'score' and no preamble or explanation. <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here are the facts:
        \n ------- \n
        {documents} 
        \n ------- \n
        Here is the answer: {generation}  <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["generation", "documents"],
    )

    hallucination_grader = prompt | loc_llm | JsonOutputParser()
    hallucination_grader.invoke({"documents": documents, "generation": generation})


async def answer_grader(question: str, generation: str) -> Optional[Dict[str, str]]:
    """
    Assess the usefulness of a generated answer to a given question.

    The function evaluates whether the answer resolves the question effectively.
    It returns a JSON object with a single key "score" having the value "yes" or "no".

    :param question: The user's question to evaluate against.
    :param generation: The generated response to assess.
    :return: A JSON object indicating whether the answer addresses the question.
    """

    # prompt = ('<|begin_of_text|><|start_header_id|>system<|end_header_id|> '
    #           'You are a grader assessing whether an answer is useful to resolve a question. Give a binary score "yes" or "no" to indicate whether the answer is useful to resolve a question. '
    #           'Provide the binary score as a JSON with a single key "score" and no preamble or explanation. '
    #           'Example: {"score": "yes"} or {"score": "no"}. '
    #           '<|eot_id|><|start_header_id|>user<|end_header_id|> '
    #           f'Here is the generated answer: \n\n{generation} \n\n'
    #           f'Here is the question: \n\n{question} \n\n'
    #           '<|eot_id|><|start_header_id|>assistant<|end_header_id|>')

    # prompt = ('<|begin_of_text|><|start_header_id|>system<|end_header_id|> '
    #           'You are evaluating whether a generated answer addresses the given question. '
    #           'Provide a binary score: "yes" if the answer directly addresses the question, and "no" if it does not.'
    #           'Important: If the generated answer is "I do not know", assign it a score of "yes" as this response '
    #           'indicates that the question has been addressed. Return only the score as JSON with a single key "score" '
    #           'and no additional text or explanation. Examples: {"score": "yes"} or {"score": "no"}.'
    #
    #           # '<|eot_id|><|start_header_id|>user<|end_header_id|> '
    #           f'Here is the generated answer: \n\n{generation} \n\n'
    #           f'Here is the question: \n\n{question} \n\n'
    #           '<|eot_id|><|start_header_id|>assistant<|end_header_id|>')

    # prompt = ('<|begin_of_text|><|start_header_id|>system<|end_header_id|> '
    #           'You are evaluating whether a generated answer addresses the given question. '
    #           'Provide a binary score: "yes" if the answer contains keywords from question, and "no" in all other cases.'
    #           'Important: If the generated answer is "I do not know", assign it a score of "yes" as this response '
    #           'indicates that the question has been addressed. Return only the score "yes" or "no" as JSON with a '
    #           'single key "score"'
    #           'and no additional text or explanation. Examples: {"score": "yes"} or {"score": "no"}.'
    #
    #           # '<|eot_id|><|start_header_id|>user<|end_header_id|> '
    #           f'Here is the generated answer: \n\n{generation} \n\n'
    #           f'Here is the question: \n\n{question} \n\n'
    #           '<|eot_id|><|start_header_id|>assistant<|end_header_id|>')

    prompt = (f'''<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are evaluating whether a generated answer addresses the given question. 
                  Provide a binary score: "yes" if the answer contains keywords from question, and "no" in all other cases.
                  Important: If the generated answer is "I do not know", assign it a score of "yes" as this response indicates that the question has been addressed.
                  Return only the score "yes" or "no" as JSON with a single key "score" and no additional text or explanation.
                  Examples: {{"score": "yes"}} or {{"score": "no"}}. 
                  <|eot_id|><|start_header_id|>user<|end_header_id|> 
                  Here is the generated answer: \n\n{generation} \n\n
                  Here is the question: \n\n{question} \n\n
                  <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    ''')  # different way to shield: .format(generation=generation, question=question)

    # async & .generate
    start_time = time.time()
    aresult = await ollama_aclient.generate(
        model=llm,
        prompt=prompt,
        format="json",
        options=options,
        # keep_alive=-1,

    )

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Async request timing client-server is: {elapsed_time:.2f} sec")
    print(f"Eval_duration: {aresult['eval_duration'] / 1_000_000_000}")

    json_result = j.str_to_json(aresult['response'])
    print("Module Check. Answer grader response: " + str(json_result))

    return json_result
