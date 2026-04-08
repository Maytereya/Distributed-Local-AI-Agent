from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional

from agent_logic_1 import aretrieve as retrieve
from agent_logic_1 import meilisearch_client as meilisearch
from agent_logic_2 import config as c
from agent_logic_2.router_preprocessor import routing
from messenger_simulator import RuntimeOptions as MessengerRuntimeOptions
from messenger_simulator import stream_message as stream_messenger_message


async def echo_ai_router(message, history, session_state, ai_feed: Literal["local", "cloud"] = "local"):
    """
    Подключает роутер и стримит ответ.
    :param ai_feed: Что подключаем: локальную LLM или облачную.
    :param message: Текст запроса пользователя
    :param history: (не используется, можно убрать)
    :param session_state: словарь сессии, хранит pending и history
    :yields: два значения — текущий кусок ответа и обновлённый session_state
    """
    session_state = session_state or {}
    try:
        # routing возвращает AsyncGenerator[(partial_response, session), None]
        async for partial, session_state in routing(message, sess=session_state, ai_feed=ai_feed):
            # Каждая итерация — новое состояние и новый кусок ответа
            yield partial, session_state

    except (asyncio.CancelledError, GeneratorExit):
        # Стоп из ChatInterface: ничего не шлем в UI, просто даём отмене подняться —
        # это закроет стримовые соединения ниже по стеку (включая Ollama).
        print("Стоп в echo_ai_router ПРОИЗОШЕЛ")
        raise

    except Exception as e:
        # При ошибке тоже стримим её сразу
        yield f"⚠️ Ошибка обработки запроса в Call-Center-Ai: {e}", session_state


def _resolve_messenger_api_url() -> str:
    # 1) Явный override через env (если нужен быстрый hotfix без правки config.ini)
    env_value = os.getenv("MESSENGER_API_URL", "").strip()
    if env_value:
        return env_value

    # 2) Канонический источник: [MESSENGER_ROUTER.<ENV>] в config.ini
    from agent_logic_2 import config as c

    cfg_value = str(getattr(c, "MESSENGER_API_URL", "") or "").strip()
    if cfg_value:
        return cfg_value

    # 3) Доп. fallback через structured settings
    structured = getattr(getattr(c, "settings", None), "messenger_router", None)
    if structured is not None:
        value = str(getattr(structured, "url", "") or "").strip()
        if value:
            return value

    # 4) Последний защитный fallback
    return "http://localhost:8000/api/messenger-generate"


def _resolve_messenger_host_header() -> Optional[str]:
    value = os.getenv("MESSENGER_API_HOST_HEADER", "").strip()
    return value or None


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    txt = raw.strip().lower()
    if txt in {"1", "true", "yes", "on"}:
        return True
    if txt in {"0", "false", "no", "off"}:
        return False
    return default


def _build_messenger_runtime_options() -> MessengerRuntimeOptions:
    llm_mode = os.getenv("MESSENGER_LLM_MODE", "hybrid").strip().lower()
    if llm_mode not in {"strict", "hybrid", "rich"}:
        llm_mode = "hybrid"

    try:
        retries = int(os.getenv("MESSENGER_SELF_CHECK_MAX_RETRIES", "1").strip())
    except Exception:
        retries = 1
    retries = min(2, max(0, retries))

    try:
        queue_timeout_ms = int(os.getenv("MESSENGER_QUEUE_TIMEOUT_MS", "30000").strip())
    except Exception:
        queue_timeout_ms = 30000
    queue_timeout_ms = min(120000, max(1000, queue_timeout_ms))

    return MessengerRuntimeOptions(
        llm_mode=llm_mode,
        self_check=_env_bool("MESSENGER_SELF_CHECK", default=False),
        self_check_max_retries=retries,
        queue_timeout_ms=queue_timeout_ms,
    )


def _is_flush_boundary(chunk_text: str) -> bool:
    if not chunk_text:
        return False
    return chunk_text[-1] in {".", "!", "?", "\n", ";", ":", "…"}


def _collection_exists(collection_name: str) -> bool:
    name = str(collection_name or "").strip()
    if not name:
        return False
    try:
        chroma_service = retrieve.ChromaService(c.chroma_host, c.chroma_port)
        return name in chroma_service.display_collections(output_format="list")
    except Exception:
        # Если проверить не удалось (например, временно недоступен Chroma),
        # не блокируем запрос на этом уровне.
        return True


def _format_messenger_attachments(attachments: List[Dict[str, Any]]) -> str:
    if not attachments:
        return ""

    lines = ["", "", "Вложения:"]
    for att in attachments:
        if not isinstance(att, dict):
            continue
        raw_url = att.get("url")
        raw_name = att.get("name")
        raw_type = att.get("type")
        url = str(raw_url or "").strip()
        name = str(raw_name or "").strip() or (url if url else "attachment")
        att_type = str(raw_type or "file").strip() or "file"
        if url:
            lines.append(f"- [{name}]({url}) ({att_type})")
        else:
            lines.append(f"- {name} ({att_type})")

    return "\n".join(lines) if len(lines) > 3 else ""


async def echo_messenger_ai(message: str, session_id: str) -> AsyncGenerator[str, None]:
    full_text = ""
    pending = ""
    attachments: List[Dict[str, Any]] = []
    seen_attachments: set[str] = set()
    handoff = False
    emitted = False
    last_flush = time.monotonic()

    flush_interval_s = 0.12
    flush_min_chars = 24

    try:
        async for chunk in stream_messenger_message(
            url=_resolve_messenger_api_url(),
            session_id=session_id,
            text=message,
            host_header=_resolve_messenger_host_header(),
            runtime_options=_build_messenger_runtime_options(),
        ):
            if chunk.error:
                raise RuntimeError(chunk.error)

            if chunk.text:
                pending += chunk.text

            if chunk.attachments:
                for att in chunk.attachments:
                    try:
                        key = json.dumps(att, ensure_ascii=False, sort_keys=True)
                    except Exception:
                        key = str(att)
                    if key in seen_attachments:
                        continue
                    seen_attachments.add(key)
                    attachments.append(att)

            if chunk.handoff:
                handoff = True

            now = time.monotonic()
            should_flush = bool(
                pending
                and (
                    _is_flush_boundary(pending)
                    or len(pending) >= flush_min_chars
                    or (now - last_flush) >= flush_interval_s
                )
            )
            if should_flush:
                full_text += pending
                pending = ""
                last_flush = now
                emitted = True
                yield full_text

        if pending:
            full_text += pending
            emitted = True
            yield full_text

        tail = _format_messenger_attachments(attachments)
        if handoff:
            tail += "\n\n[Система] Диалог передан оператору."
        if tail:
            final_text = f"{full_text}{tail}".strip()
            if final_text != full_text:
                emitted = True
                yield final_text

        if not emitted:
            yield "(no text)"

    except (asyncio.CancelledError, GeneratorExit):
        raise
    except Exception as e:
        yield f"⚠️ Ошибка обработки запроса в Messengers-Ai: {e}"


async def chroma_echo(
    message: str,
    history: List[Dict],
    collection: str,
    threshold_value: float,
    slider_value_n_results: int,
    slider_value_k,
    radio_value,
) -> str:
    """
    Main chroma call func. Its return the pieces of text from uploaded to chroma docs.
    :param collection: Str. Chosen collection name.
    :param message: Str. The users question.
    :param history: Obligate parameter for correct gradio executes.
    :param threshold_value:
    :param slider_value_n_results:
    :param slider_value_k:
    :param radio_value: Type of search established.
    :return: String of filtrated text from ChromaDB.
    """
    return await retrieve.main_retrieve_async(
        question=message,
        collection=collection,
        return_type="str",
        threshold=threshold_value,
        n_results=slider_value_n_results,
        k=slider_value_k,
        search_type=radio_value,
    )


async def meili_echo(
    message: str,
    history: List[Dict],
    index: str,
    limit: int,
) -> str:
    """
    Поддерживает прямое обращение к серверу Meilisearch.
    :param message:
    :param history:
    :param index:
    :param limit:
    :return: String - результаты поиска
    """
    return meilisearch.search_meili(query=message, index_name=index, limit=limit)


def _resolve_free_talk_stream():
    """
    Импортирует stream_free_talk(_with_state) из package `localragagent`.

    В dev-режиме проект часто запускается без editable install,
    поэтому добавляем `<repo>/src` в sys.path как fallback.
    """
    try:
        from localragagent.freetalk import runner as free_talk_runner

        stream = getattr(free_talk_runner, "stream_free_talk_with_state", None)
        if callable(stream):
            return stream
        return free_talk_runner.stream_free_talk
    except ModuleNotFoundError:
        src_dir = Path(__file__).resolve().parents[3] / "src"
        if src_dir.exists():
            src_str = str(src_dir)
            if src_str not in sys.path:
                sys.path.insert(0, src_str)
        from localragagent.freetalk import runner as free_talk_runner

        stream = getattr(free_talk_runner, "stream_free_talk_with_state", None)
        if callable(stream):
            return stream
        return free_talk_runner.stream_free_talk


async def universal_echo(
    message: str,
    history: List[Dict],
    radio_value: str,  # "Free-talk-Ai", "Call-Center-Ai", "gigachat", "meilisearch", "vectorstore", "db"
    threshold_value: float,
    slider_value_n_results: int,
    slider_value_k: int,
    collection: str,
    meili_index: str,
    messenger_session_id: Optional[str] = None,
):
    messenger_session_id = str(messenger_session_id or "").strip() or None

    if radio_value == "Free-talk-Ai":
        if not history:
            session_id = f"gr_ft_{uuid.uuid4().hex[:12]}"
        else:
            current = str(messenger_session_id or "").strip()
            session_id = current if current.startswith("gr_ft_") else f"gr_ft_{uuid.uuid4().hex[:12]}"
        stream_free_talk = _resolve_free_talk_stream()
        current_session_id = session_id
        async for item in stream_free_talk(
            message=message,
            history=history,
            session_id=session_id,
        ):
            if isinstance(item, tuple) and len(item) == 2:
                partial = str(item[0] or "")
                candidate = str(item[1] or "").strip()
                if candidate.startswith("gr_ft_"):
                    current_session_id = candidate
            else:
                partial = str(item or "")
            yield partial, current_session_id
        return

    if radio_value == "Call-Center-Ai":
        session_state: dict = {}
        ai_feed: Literal["local", "cloud"] = "local"
        async for partial, session_state in echo_ai_router(message, history, session_state, ai_feed=ai_feed):
            yield partial, messenger_session_id
        return

    if radio_value == "Messengers-Ai":
        if not history:
            session_id = f"gr_mr_{uuid.uuid4().hex[:12]}"
        else:
            current = str(messenger_session_id or "").strip()
            session_id = current if current.startswith("gr_mr_") else f"gr_mr_{uuid.uuid4().hex[:12]}"
        async for partial in echo_messenger_ai(message, session_id=session_id):
            yield partial, session_id
        return

    if radio_value == "gigachat":
        session_state: dict = {}
        ai_feed: Literal["local", "cloud"] = "cloud"
        async for partial, session_state in echo_ai_router(message, history, session_state, ai_feed=ai_feed):
            yield partial, messenger_session_id
        return

    if radio_value == "meilisearch":
        result = await meili_echo(
            message=message,
            history=history,
            index=meili_index,
            limit=slider_value_k,
        )
        yield result, messenger_session_id
        return

    # сюда попадём только если radio_value == "vectorstore" или "db"
    if not collection:
        yield "⚠️ Не выбрана коллекция для поиска в Chroma.", messenger_session_id
        return
    if not _collection_exists(collection):
        yield f"⚠️ Коллекция '{collection}' не найдена. Обновите список коллекций.", messenger_session_id
        return

    result = await chroma_echo(
        message=message,
        history=history,
        collection=collection,
        threshold_value=threshold_value,
        slider_value_n_results=slider_value_n_results,
        slider_value_k=slider_value_k,
        radio_value=radio_value,
    )
    yield result, messenger_session_id
