import asyncio
import numpy as np
from scipy.signal import resample_poly

from sber_asr_stream import SberStream, log, TARGET_SR, SEND_MS


# --- локальные хелперы аудио (как в VOSK) ---
def _to_mono_float32(x):
    x = np.asarray(x)
    if x.ndim == 2 and x.shape[1] > 1:
        x = x.mean(axis=1)
    return x.astype(np.float32, copy=False)

def _normalize_gain(x, target_rms=0.055, max_gain=8.0):
    rms = float(np.sqrt(np.mean(x * x) + 1e-12))
    if rms < 1e-6:
        return x
    g = min(max_gain, target_rms / rms)
    return np.clip(x * g, -1.0, 1.0)

def _resample_to_16k(x, sr):
    if sr == 16000: return x
    if sr == 48000: return resample_poly(x, up=1, down=3)
    if sr == 44100: return resample_poly(x, up=160, down=441)
    return resample_poly(x, up=16000, down=sr)


# --- Gradio streaming callback ---
async def asr_stream_sber(audio_chunk, asr_state):
    """
    Поведение как у VOSK-версии:
      - один аккумулятор acc_pcm (bytearray) в state;
      - на каждом тике готовим PCM16LE@16k и добавляем в acc_pcm;
      - при >= SEND_MS отправляем ВСЁ содержимое acc_pcm через s.send_bytes(), затем acc_pcm.clear().
    """
    if not isinstance(asr_state, dict):
        asr_state = {}

    # запуск/реюз стрима
    s: SberStream | None = asr_state.get("sber")
    if (s is None) or getattr(s, "closed", False):
        s = SberStream()
        await s.start()
        asr_state["sber"] = s
        asr_state.setdefault("text", "")
        asr_state.setdefault("partial", "")
        log.info("Sber_stream (re)created")

    # общий аккумулятор PCM16LE@16k
    acc = asr_state.setdefault("acc_pcm", bytearray())

    # ждём (sr, ndarray) от gr.Audio(type="numpy")
    if not (isinstance(audio_chunk, tuple) and len(audio_chunk) == 2):
        live = (asr_state.get("text", "") + " " + asr_state.get("partial", "")).strip()
        return live, asr_state

    sr, data = audio_chunk
    log.info("asr_stream_sber: sr=%s shape=%s dtype=%s",
             sr, getattr(data, "shape", None), getattr(data, "dtype", None))

    # 1) моно + float32
    x = _to_mono_float32(data)
    # 2) нормализация
    x = _normalize_gain(x, target_rms=0.055, max_gain=8.0)
    # 3) ресемпл → 16k
    x16 = _resample_to_16k(x, sr)
    # 4) в PCM16LE
    pcm16 = (np.clip(x16, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()

    # накапливаем
    acc.extend(pcm16)

    # отправляем целиком по порогу
    need_bytes = int(TARGET_SR * (SEND_MS / 1000.0)) * 2
    if len(acc) >= need_bytes:
        await s.send_bytes(bytes(acc))  # запись в WAV делается внутри send_bytes
        acc.clear()

    # дать listener'у шанс обновить partial (важно для Gradio)
    await asyncio.sleep(0)

    live = (s.text_final + " " + s.partial).strip()
    asr_state["text"], asr_state["partial"] = s.text_final, s.partial
    return live, asr_state


# --- Завершение фразы / кнопка STOP ---
async def sber_flush(asr_state):
    """Добросить хвост acc_pcm и отправить EOF, вернуть финальный текст."""
    log.info("flush begins")
    s: SberStream | None = asr_state.get("sber") if isinstance(asr_state, dict) else None
    if s:
        acc = asr_state.get("acc_pcm")
        if acc:
            await s.send_bytes(bytes(acc))
            acc.clear()
        await s.eof()
        log.info("Sber_stream awaited eof")

    final_text = (s.text_final if s else "").strip()
    log.info("Sber_flush awaited final text: %s", final_text)

    if isinstance(asr_state, dict):
        asr_state["partial"] = ""
        # не реюзать закрытый поток
        asr_state.pop("sber", None)
        # acc_pcm оставляем пустым — на следующий запуск

    return final_text, asr_state or {}


# --- Сброс из кнопки ---
def sber_reset_state(asr_state):
    """Аккуратно закрыть предыдущий поток (не блокируя UI) и вернуть чистый state."""
    async def _close(s: SberStream):
        try:
            await s.eof()
        except Exception:
            pass

    if isinstance(asr_state, dict) and asr_state.get("sber"):
        try:
            asyncio.create_task(_close(asr_state["sber"]))
        except RuntimeError:
            pass

    return "", {}  # пустой текст и новый чистый state