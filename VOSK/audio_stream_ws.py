import asyncio
import json

import numpy as np
import websockets
from scipy.signal import resample_poly

import agent_logic_2.config as c

GRAMMAR = [
    # Фамилия и варианты
    "Дразнин", "дразнин",
    "Дразнина", "Дразнину", "Дразнином", "Дразнине",
    # Рядом стоящие слова для биграмм
    "доктор Дразнин", "к доктору Дразнину",
    # Доменные термины
    "медкарта", "амбулаторный", "невролог", "Медцентр", "Нейро-пси"
]
TARGET_SR = 16000
SEND_MS = 500  # 200–400 мс — оптимально


def _to_mono_float32(x):
    x = np.asarray(x)
    if x.ndim == 2 and x.shape[1] > 1:
        x = x.mean(axis=1)
    return x.astype(np.float32, copy=False)


def _normalize_gain(x, target_rms=0.05, max_gain=8.0):
    rms = float(np.sqrt(np.mean(x ** 2) + 1e-12))
    if rms < 1e-6:
        return x
    g = min(max_gain, target_rms / rms)
    y = x * g
    return np.clip(y, -1.0, 1.0)


def _resample_to_16k(x, sr):
    if sr == 16000: return x
    if sr == 48000: return resample_poly(x, up=1, down=3)
    if sr == 44100: return resample_poly(x, up=160, down=441)
    return resample_poly(x, up=16000, down=sr)


async def _ensure_ws(state):
    if state.get("ws") and not state.get("closing"):
        return state["ws"]
    ws = await websockets.connect(c.VOSK_URL, max_size=1 << 20)
    await ws.send(json.dumps({"config": {"sample_rate": TARGET_SR, "phrase_list": GRAMMAR}, "words": 1}, ))
    state["ws"] = ws
    state["closing"] = False
    return ws


async def _recv_nonblock(ws, timeout=0.001):
    try:
        msg = await asyncio.wait_for(ws.recv(), timeout=timeout)
        return msg
    except asyncio.TimeoutError:
        return None


async def flush_ws(asr_state):
    """Отправить EOF и добрать финал."""
    ws = asr_state.get("ws")
    if not ws:
        return asr_state.get("text", ""), asr_state
    # добросить хвост буфера, если есть
    acc = asr_state.get("acc")
    if acc:
        await ws.send(bytes(acc))
        acc.clear()
    await ws.send(json.dumps({"eof": 1}))
    # дочитать финал
    try:
        last = await asyncio.wait_for(ws.recv(), timeout=1.0)
        j = json.loads(last)
        final = (j.get("text") or "").strip()
        if final:
            asr_state["text"] = (asr_state.get("text", "") + " " + final).strip()
            asr_state["last_final"] = final  # ← запомнить последний законченный фрагмент
    except asyncio.TimeoutError:
        pass
    asr_state["closing"] = True
    await ws.close()
    asr_state["ws"] = None
    return asr_state.get("text", ""), asr_state


# === основной обработчик для gr.Audio(streaming=True) ===
async def vosk_ws_stream(audio_chunk, asr_state):
    """
    Возвращает (live_text, state) на каждый чанк.
    Держит открытым один WebSocket на сессию.
    """
    if asr_state is None:
        asr_state = {"text": "", "acc": bytearray(), "ws": None, "closing": False}

    # распаковка чанка
    if isinstance(audio_chunk, tuple):
        sr, data = audio_chunk
    else:
        sr, data = 16000, audio_chunk

    # пустой тик: просто показываем то, что есть
    if data is None or len(data) == 0:
        # не закрываем сокет без явной команды; можно сделать отдельную кнопку flush
        return asr_state.get("text", ""), asr_state

    # подготовка сэмплов
    f = _to_mono_float32(data)
    f = _normalize_gain(f, target_rms=0.05, max_gain=8.0)
    f16 = _resample_to_16k(f, sr)
    pcm16 = (np.clip(f16, -1, 1) * 32767).astype(np.int16).tobytes()

    # буферизация до SEND_MS
    asr_state["acc"].extend(pcm16)
    need_bytes = int(TARGET_SR * (SEND_MS / 1000.0)) * 2
    ws = await _ensure_ws(asr_state)

    if len(asr_state["acc"]) >= need_bytes:
        feed = bytes(asr_state["acc"])
        asr_state["acc"].clear()
        await ws.send(feed)

    # читаем 0..N сообщений без блокировки
    text = asr_state.get("text", "")
    while True:
        msg = await _recv_nonblock(ws, timeout=0.0001)
        if not msg:
            break
        j = json.loads(msg)
        partial = (j.get("partial") or "").strip()
        final = (j.get("text") or "").strip()
        if final:
            text = (text + " " + final).strip()
            asr_state["text"] = text
            asr_state["last_final"] = final  # ← запомнить последний законченный фрагмент
        elif partial:
            # временный view: база + partial
            text = (asr_state.get("text", "") + " " + partial).strip()

    return text, asr_state
