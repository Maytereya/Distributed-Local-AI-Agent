import json
import logging

import librosa
import numpy as np
import websockets

from agent_logic_2 import config as c


def enable_debug_logging():
    # консольный логгер
    logging.basicConfig(
        level=logging.INFO,
    )


TARGET_SR = 16000


async def ws_transcribe(audio, meta_output:bool=False):
    """
    audio: (sr, numpy.ndarray) от Gradio (type='numpy')
    Отправляем один цельный кусок аудио по WebSocket на /stream,
    получаем JSON, возвращаем текст.
    """
    if audio is None:
        return "Нет аудио 😅"

    sr, data = audio

    # в лог — посмотреть, что выдаёт Gradio
    logging.log(logging.INFO, f"Client audio sr={sr}, shape={data.shape}")

    # Приводим к mono
    if data.ndim > 1:
        data = data.mean(axis=1)

    # 👉 СНАЧАЛА приводим к float32 и нормализуем, если было целое
    if np.issubdtype(data.dtype, np.integer):
        # например, int16 → [-1, 1]
        max_val = np.iinfo(data.dtype).max
        data = data.astype(np.float32) / max_val
    else:
        data = data.astype(np.float32)

    # ресемплим до 16 kHz
    if sr != TARGET_SR:
        data = librosa.resample(y=data, orig_sr=sr, target_sr=TARGET_SR)
        # sr = TARGET_SR

    # Приводим к float32 на всяк случай повторно
    data = data.astype(np.float32)

    async with websockets.connect(c.WHISPER_URL, max_size=10_000_000) as ws:
        # Отправляем всё аудио одним чанком (как "стрим" из одного куска)
        await ws.send(data.tobytes())
        # Сигнал конца потока
        await ws.send("END")

        # Ждём ответ от сервера
        result_raw = await ws.recv()
        try:
            result = json.loads(result_raw)
        except json.JSONDecodeError:
            return f"Некорректный ответ сервера: {result_raw}"

        if "error" in result:
            return f"Ошибка на сервере: {result['error']}"

        text = result.get("text") or ""
        lang = result.get("language") or "unknown"
        tsec = result.get("time_sec")

        meta = f"[lang: {lang}, time: {tsec}s]" if tsec is not None else f"[lang: {lang}]"
        if meta_output:
            return f"{meta}\n\n{text}"
        else:
            return text