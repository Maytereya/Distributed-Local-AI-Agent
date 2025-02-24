from unittest import result

import chromadb
import asyncio
import config as c
from ollama import AsyncClient, Options

# Выбор llm
llm = c.ll_model_big

prompt = """Расскажи кратко о себе, используй не более 3 абзацев текста"""

c.choose_exec_directory("root")
ollama_aclient = AsyncClient(host=c.ollama_url)
options = Options(temperature=0.5, )


async def test_prompt():
    aresult = await ollama_aclient.generate(
        model=llm,
        prompt=prompt,
        # format="json",
        options=options,
        # keep_alive=-1,

    )
    print(f"Eval_duration of answer generation: {aresult['eval_duration'] / 1_000_000_000}")

    print("Тестовый эбаут майселф от Llama 3.*: ", aresult['response'])

    return None


if __name__ == '__main__':
    print(':: TESTING ::')
    print("c.chroma_host =", c.chroma_host)
    chroma_client = chromadb.HttpClient(host=c.chroma_host, port=c.chroma_port)

    if chroma_client.heartbeat() > 0:
        print("ChromaDB Healthy")
    print("Версия Chroma:", chroma_client.get_version())

    listing = chroma_client.list_collections()
    i = 0
    for collection in listing:
        i += 1
        print(i, collection.name)

    print(" ============================= ")
    asyncio.run(test_prompt())
