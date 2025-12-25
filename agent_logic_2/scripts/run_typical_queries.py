"""
Быстрый прогон типовых запросов через router_preprocessor.
Печатает ответы на все запросы подряд, чтобы быстро оценить качество поиска.
"""

from __future__ import annotations

import asyncio

from agent_logic_2.router_preprocessor import process_routing_request

QUERIES = [
    "Митрошкина",
    "Митрошкина Мария",
    "Кто такая Митрошкина Врач ????",
    "Карасев расписание",
    "Рыжова график работы",
    "Покажи всех узистов",
    "Врачи УЗИ приходящие, работает с детьми",
    "Урологи принимающие по дмс",
    "Хирурги работающие с детьми",
    "Хирург по дмс",
    "Кто делает торакоцентез",
    "Лор врачи приходящие",
]


async def main() -> None:
    for idx, query in enumerate(QUERIES, 1):
        print("=" * 80)
        print(f"[{idx}] Запрос: {query}")
        try:
            answer, _ = await process_routing_request(query)
        except Exception as exc:
            print(f"Ошибка: {exc}")
            continue
        print("Ответ:")
        print(answer)


if __name__ == "__main__":
    asyncio.run(main())
