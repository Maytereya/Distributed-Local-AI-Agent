"""Тема реестра перебивает метку только по смыслу — подтверждает LLM (решение владельца 18.10; принцип 2).

Реестр тем (`topic_registry.yaml`) находит тему по словам реплики — сравнением основ слов. Это даёт кандидата, но не
смысл: «Ново-Садовая 13» совпадает с «новость» и «новости» (основа «новос» начинается с «ново»), и реплику с адресом
NLU-метка OTHER меняет на акции. На живой переписке (6049 реплик пациентов) такие совпадения только по основе — 18,
почти все — улицы («Ново-Садовая», «Ново-Вокзальная»).

Под `[MESSENGER_ROUTER] topic_override_llm = on`: когда тема перебила бы метку OTHER (единственное место, где слова
реестра меняют маршрут), модуль выбора (`choose`, задача `topic`) получает реплику и название темы и отвечает, о ней ли
реплика по смыслу. Подтвердила — перебиваем, как раньше; «ни одна», сбой или мусор — метка остаётся (тема — только
подсказка в флагах). Подтверждение и подсказка (без смены метки) LLM не зовут.
"""

from __future__ import annotations

from typing import Any

from .choice import Option, Status, Task, choose
from .runtime_config import config as _cfg

TOPIC_TASK = Task(name="topic", max_picks=1, num_predict=32, timeout_s=5.0, queue_timeout_ms=1500)


def enabled() -> bool:
    """`topic_override_llm = on`."""

    return str(getattr(_cfg, "MR_TOPIC_OVERRIDE_LLM", "off") or "").strip().lower() == "on"


async def confirms(user_text: str, topic_match: Any) -> bool:
    """Реплика по смыслу — об этой теме. Сбой LLM или «ни одна» — нет (метку не перебиваем)."""

    topic_id = str(getattr(topic_match, "topic_id", "") or "")
    title = str(getattr(topic_match, "title", "") or topic_id)
    choice = await choose(TOPIC_TASK, user_text, [Option(id=topic_id, text=title)])
    return choice.status is Status.PICKED and topic_id in choice.ids
