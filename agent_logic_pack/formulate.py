import asyncio

from agent_logic_2 import config as c
from ollama import AsyncClient, Options

ollama_aclient = AsyncClient(host=c.ollama_url)
options = Options(temperature=1, )

# Выбор llm
llm = c.ll_model_big


async def formulate(question: str, ):
    """
    Formulate a question of the user
    :param question: Сырой запрос пользователя
    :return: Обработанный запрос пользователя для облегчения поиска в векторной базе и фомулирования правильного запроса
    """

    prompt = ('<|begin_of_text|><|start_header_id|>system<|end_header_id|> '
              'You are transforming the user’s query into a clear, complete, and unambiguous request related to '
              'medical topics, specifically psychiatry, psychopharmacology, and the side effects of medications. '
              'Guidelines for formulating the final query: '
              'If the user’s query contains the name of a medication, '
              'structure the final query so that it begins with the medication’s name, followed by phrases to '
              'look up its effects, potential side effects, and methods for managing these effects or symptoms. '
              'The medication name should always come first. '
              'If the query concerns a symptom, syndrome, or disorder, structure the final query so that it begins '
              'with the name of the symptom, followed by “causes,” “treatment,” and “management” in logical sequence. '
              'The name of the symptom, syndrome, or disorder should be at the beginning. '
              'The final query format should be only in Russian.'
              'The final output format should strictly follow the instruction format without any duplication '
              'in English.'
              'If you know the meaning of abbreviations, such as ЗНС (злокачественный нейролептический синдром) or '
              'СС (серотониновый синдром), expand them in full but keep the abbreviation at the beginning of the output.'
              'Do not use any introductory words like "Формулировка", "formulation" etc. '
              'For example: "Клозапин, побочные эффекты, способы коррекции"'
              # 'User queries are provided in Russian, and the answer should also be in Russian.'
              '<|eot_id|><|start_header_id|>user<|end_header_id|> '
              f'Question: {question}. \n\n'
              '<|eot_id|><|start_header_id|>assistant<|end_header_id|>')

    aresult = await ollama_aclient.generate(
        model=llm,
        prompt=prompt,
        # format="json",
        options=options,
        # keep_alive=-1,

    )

    print(f"Eval_duration of answer generation: {aresult['eval_duration'] / 1_000_000_000}")
    #
    # print("Формулировка: ")
    # print(aresult['response'])
    return aresult['response']


async def extract_keyword(query: str, ) -> str:
    """
    :param query: Формулировка, содержащая ключевое слово.
    :return: Только ключевое слово для поиска в ChromaDB или фильтрации результатов при поиске vectorstore
    """

    prompt = ('<|begin_of_text|><|start_header_id|>system<|end_header_id|> '
              'Identify the main keyword in the user query, focusing on the noun that represents the core element of the query, such as the name of a medication, symptom, or syndrome. '
              'Guidelines for extracting the main keyword: Select a single keyword that serves as the primary subject of the query. '
              'If the query includes the name of a medication, extract the medication name. '
              'If the query concerns a symptom or syndrome, extract the full name of the symptom or syndrome. '
              'Examples: Query: "Побочные эффекты лекарства в таблетках клозапина, способы коррекции" '
              'Extracted keyword: "Клозапин" '
              'Query: "Симптомы злокачественного нейролептического и как лечить?"'
              'Extracted keyword: "Злокачественный нейролептический синдром" '
              'The output should only include the main keyword,'
              'without additional text or explanations.'
              '<|eot_id|><|start_header_id|>user<|end_header_id|> '
              f'Query: {query}. \n\n'
              '<|eot_id|><|start_header_id|>assistant<|end_header_id|>')

    aresult = await ollama_aclient.generate(
        model=llm,
        prompt=prompt,
        # format="json",
        options=options,
        # keep_alive=-1,

    )

    print(f"Eval_duration of answer generation: {aresult['eval_duration'] / 1_000_000_000}")
    #
    print("Ключевое слово: ")
    print(aresult['response'])
    return aresult['response']


async def main(question: str, ):
    query = await formulate(question)
    await extract_keyword(query)


if __name__ == "__main__":
    q = input("Question: ")
    asyncio.run(main(q))
