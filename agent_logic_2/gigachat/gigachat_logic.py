
import asyncio

from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole
from agent_logic_2 import config as c

def gigachad_echo():
    # Укажите ключ авторизации, полученный в личном кабинете, в интерфейсе проекта GigaChat API
    with GigaChat(credentials=c.giga_authorization, verify_ssl_certs=False) as giga:
        response = giga.chat("Какие изменяемые факторы влияют на продолжительность жизни человека?")
        print(response.choices[0].message.content)


"""Пример работы с чатом"""

payload = Chat(
    messages=[
        Messages(
            role=MessagesRole.SYSTEM,
            content="Ты внимательный бот-психолог, который помогает пользователю решить его проблемы."
        )
    ],
    temperature=0.7,
    max_tokens=200,
)

PAYLOAD = Chat(
    messages=[
        Messages(
            role=MessagesRole.SYSTEM,
            content="Ты - умный ИИ ассистент, который всегда готов помочь пользователю.",
        ),
        Messages(
            role=MessagesRole.ASSISTANT,
            content="Как я могу помочь вам?",
        ),
        Messages(
            role=MessagesRole.USER,
            content="Напиши подробный доклад на тему жизни Пушкина в Москве",
        ),
    ],
    update_interval=0.1,
)


async def gigachad_echo_async(system: str, prompt: str, ) -> str:
    pl = Chat(
        messages=[
            Messages(
                role=MessagesRole.SYSTEM,
                content=system,
            ),
            # Messages(
            #     role=MessagesRole.ASSISTANT,
            #     content="Как я могу помочь вам?"),
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
            response_list.append(chunk.choices[0].delta.content)
            if response_list:
                response_str: str = "\n\n---\n\n".join(response_list)
                # print(chunk.choices[0].delta.content, flush=True)
            else:
                response_str = "GigaChat не отвечает"
        return response_str


system_prmt = "Ты - вежливый ассистент"
question = "Скажи базовые данные о себе: твое название/имя, с какими данными и запросами ты работаешь лучше всего?"


async def main():
    await gigachad_echo_async(system_prmt, question)


if __name__ == "__main__":
    print(asyncio.run(main()))
