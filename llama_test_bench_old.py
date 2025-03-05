import asyncio
from ollama import AsyncClient, Options

ollama_async_client = AsyncClient(host="http://192.168.1.42:11434")
options = Options(
    temperature=0.4
    #     Тут может быть довольно много всяких опций, смотри в ollama API.
)

# Выбор llm
llm = "llama3.3:70b-instruct-q8_0"


async def formulate(question: str, ):
    """
    Отвечалка на вопросы без памяти.

    :param question: Вопрос.
    :return: Ответ на вопрос.
    """

    prompt = (f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
              You are helpful assistant. You are call centre employee in private hospital with "Наука" brande name. 
              Use russian language only for your answers. 
              Your goal is to answer the questions about medical conditions. 
              You need to use simple sentences. You must recommend to visit the doctor.           
              <|eot_id|><|start_header_id|>user<|end_header_id|> 
              Question: {question}. \n\n
              <|eot_id|><|start_header_id|>assistant<|end_header_id|>""")

    aresult = await ollama_async_client.generate(
        model=llm,
        prompt=prompt,
        # format="json",
        options=options,
        keep_alive=-1,

    )

    print(f"Eval_duration of answer generation: {aresult['eval_duration'] / 1_000_000_000}")
    #

    return aresult['response']


async def main(question: str, ):
    query = await formulate(question)
    print(f"{llm}answer: \n\n {query}")
    # await formulate(query)


if __name__ == "__main__":
    q = input("Ask your question: ")
    asyncio.run(main(q))
