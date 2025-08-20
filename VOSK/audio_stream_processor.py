# --- VOSK init (глобально, 1 раз) ---
import json as pyjson

import numpy as np
from scipy.signal import resample_poly
from vosk import Model, KaldiRecognizer

import agent_logic_2.config as c

SAMPLE_RATE = 16000  # норма модели / входа
TARGET_SR = 16000
BUF_SECONDS = 0.60  # 500 мс буферизации

_vosk_model = None


def _get_vosk_model():
    global _vosk_model
    if _vosk_model is None:
        _vosk_model = Model(c.VOSK_model_path)
    return _vosk_model


def _mk_recognizer():
    rec = KaldiRecognizer(_get_vosk_model(), TARGET_SR)
    rec.SetWords(True)
    return rec


def _ensure_state(asr_state):
    # asr_state: {"rec":..., "text":..., "acc": bytearray()}
    if asr_state is None:
        asr_state = {}
    if "rec" not in asr_state:
        asr_state["rec"] = _mk_recognizer()
    if "text" not in asr_state:
        asr_state["text"] = ""
    if "acc" not in asr_state:
        asr_state["acc"] = bytearray()
    return asr_state


def _to_mono_float32(x):
    # Gradio иногда даёт стерео: shape (n, 2)
    if x is None:
        return None
    x = np.asarray(x)
    if x.ndim == 2 and x.shape[1] > 1:
        x = x.mean(axis=1)
    return x.astype(np.float32, copy=False)


def _resample_to_16k(data, sr):
    if sr == TARGET_SR:
        return data
    # 48000 -> 16000: down=3, up=1
    # на прочих частотах resample_poly всё равно годится
    return resample_poly(data, up=1, down=3)
    # return resample_poly(data, up=TARGET_SR, down=sr)


def vosk_stream(audio_chunk, asr_state):
    """
    Gradio .stream дергает это на каждый чанк.
    Возвращаем (live_text, state) один раз на чанк.
    """
    try:
        # Достаём sr и массив
        if isinstance(audio_chunk, tuple):
            sr, data = audio_chunk
        else:
            sr, data = TARGET_SR, audio_chunk  # fallback

        asr_state = _ensure_state(asr_state)
        rec = asr_state["rec"]

        # Пустой тик — просто вернём накопленное
        if data is None or len(data) == 0:
            yield asr_state["text"], asr_state
            return

        # Моно float32
        f = _to_mono_float32(data)

        # Ресемпл к 16k
        f16 = _resample_to_16k(f, sr)

        # float32 [-1..1] -> int16 bytes
        pcm16 = (np.clip(f16, -1, 1) * 32767).astype(np.int16).tobytes()

        # Буферизация по времени
        asr_state["acc"].extend(pcm16)
        need_bytes = int(TARGET_SR * BUF_SECONDS) * 2  # 2 байта на сэмпл

        if len(asr_state["acc"]) < need_bytes:
            # слишком мало — попробуем показать partial
            pres = pyjson.loads(rec.PartialResult())
            partial = (pres.get("partial") or "").strip()
            view = asr_state["text"]
            if partial:
                view = (view + " " + partial).strip()
            yield view, asr_state
            return

        # Кормим распознаватель накопленным
        print("before feed", len(asr_state["acc"]))
        feed = bytes(asr_state["acc"])
        asr_state["acc"].clear()
        print("after clear", len(asr_state["acc"]))

        # ДЛЯ ДЕБАГА: печать после очистки — acc будет 0, это ок
        print(f"chunk sr={sr} bytes={len(feed)} acc={len(asr_state['acc'])}")

        if rec.AcceptWaveform(feed):
            res = pyjson.loads(rec.Result())
            full = (res.get("text") or "").strip()
            if full:
                asr_state["text"] = (asr_state["text"] + " " + full).strip()
            yield asr_state["text"], asr_state
        else:
            pres = pyjson.loads(rec.PartialResult())
            partial = (pres.get("partial") or "").strip()
            view = asr_state["text"]
            if partial:
                view = (view + " " + partial).strip()
            yield view, asr_state

    except Exception as e:
        yield f"ASR error: {e}", asr_state
