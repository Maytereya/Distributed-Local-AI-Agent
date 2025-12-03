import asyncio

from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole

from agent_logic_2 import config as c


async def gigachad_echo_async(system: str, prompt: str, ):
    pl = Chat(
        messages=[
            Messages(
                role=MessagesRole.SYSTEM,
                content=system,
            ),
            Messages(
                role=MessagesRole.USER,
                content=prompt,
            ),
        ],
        update_interval=0.1,
        temperature=0.4,
        max_tokens=1000,
    )
    response_str: str = ""
    response_list = []
    async with GigaChat(credentials=c.giga_authorization, verify_ssl_certs=False) as giga:

        async for chunk in giga.astream(pl):
            yield chunk



async def main():
    system_prmt = "Ты - вежливый ассистент"
    question = "Скажи базовые данные о себе: твое название/имя, с какими данными и запросами ты работаешь лучше всего?"
    await gigachad_echo_async(system_prmt, question)


if __name__ == "__main__":
    print(asyncio.run(main()))
