import numpy as np
from scipy.signal import resample_poly

import asyncio
import logging

import grpc
import numpy as np
from google.protobuf.duration_pb2 import Duration  # Поставил IDE ignore потому что не распознается.

import agent_logic_2.config as c
import sber_proto.recognitionv2_pb2 as pb
import sber_proto.recognitionv2_pb2_grpc as pb_grpc

log = logging.getLogger("asr_sber")

TARGET_SR = 16000
# Пока самое значимое влияние на распознавание
SEND_MS = 1200  # 1000–1500 мс

from sber_dump import PcmDump, DumpMeta



class SberStream:
    """Минимальный bidi-стрим: options → audio_chunk(bytes...) → EOF.
       Все препроцедуры и сегментацию делаем снаружи (в asr_stream_sber)."""

    def __init__(self):
        self.q: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=16)
        self.text_final = ""
        self.partial = ""
        self._closing = False
        self._call = None
        self._listener_task = None
        self._channel = None
        self._stub = None
        self.closed = False
        self._dump = None  # PcmDump | None

    async def start(self):
        # TLS
        ssl = grpc.ssl_channel_credentials(
            open(c.SBER_CA, "rb").read() if getattr(c, "SBER_CA", None) else None
        )

        # канал
        self._channel = grpc.aio.secure_channel(
            c.SBER_HOST,
            ssl,
            options=[
                ("grpc.keepalive_time_ms", 20000),
                ("grpc.keepalive_timeout_ms", 10000),
                ("grpc.keepalive_permit_without_calls", 1),
                ("grpc.http2.max_pings_without_data", 0),
            ],
        )
        self._stub = pb_grpc.SmartSpeechStub(self._channel)

        # авторизация через metadata
        md = (("authorization", f"Bearer {c.SBER_TOKEN}"),)

        # запуск стрима
        self._call = self._stub.Recognize(
            self._gen(),
            metadata=md,
            timeout=300.0,
            wait_for_ready=True,
        )

        # дампер (если включён)
        if c.SBER_DUMP_AUDIO:
            meta = DumpMeta(
                sample_rate=TARGET_SR,
                channels=1,
                sample_width=2,
                send_ms=SEND_MS,
                target_rms=0.055,
                max_gain=8.0,
                note="postproc in UI: mono+normalize+resample16k -> PCM16"
            )
            out_dir = c.SBER_DUMP_DIR
            log.info("Dumping audio to WAV into: %r", out_dir)
            self._dump = PcmDump(out_dir, meta, prefix="sber_out")
        else:
            self._dump = None

        self._listener_task = asyncio.create_task(self._listen())
        self.closed = False
        log.info("SberStream started")

    async def _gen(self):
        """Сначала отправляем RecognitionOptions, затем читаем из очереди PCM-байты."""
        log.info("GEN start")
        try:
            opts = pb.RecognitionOptions(
                audio_encoding=pb.RecognitionOptions.PCM_S16LE,
                sample_rate=TARGET_SR,
                channels_count=1,
                enable_partial_results=pb.OptionalBool(enable=True),
                enable_multi_utterance=pb.OptionalBool(enable=False),
                no_speech_timeout=Duration(seconds=3),
                max_speech_timeout=Duration(seconds=12),
                normalization_options=pb.NormalizationOptions(
                    force_cyrillic=pb.OptionalBool(enable=True)
                ),
                language="ru-RU",
                # model=getattr(c, "SBER_MODEL", ""),
            )
            log.info("GEN send options")
            yield pb.RecognitionRequest(options=opts)

            total = 0
            while True:
                chunk = await self.q.get()  # bytes или None
                if chunk is None:
                    log.info("GEN EOF (total=%d)", total)
                    break
                total += len(chunk)
                log.debug("GEN send %d bytes (total=%d)", len(chunk), total)
                yield pb.RecognitionRequest(audio_chunk=chunk)

        except Exception:
            log.exception("GEN fatal before first yield")
            raise

    async def send_bytes(self, pcm_bytes: bytes):
        """Принять готовый PCM16LE@16k (без перекрытий) и отправить в gRPC.
           Параллельно пишем тот же поток в WAV-дамп (если включён)."""
        if not pcm_bytes:
            return
        if self._dump is not None:
            self._dump.write(pcm_bytes)
        await self.q.put(pcm_bytes)
        secs = len(pcm_bytes) / (TARGET_SR * 2)
        log.info("SEND BYTES: dur=%.3fs, bytes=%d", secs, len(pcm_bytes))

    async def _listen(self):
        try:
            async for resp in self._call:
                if resp.HasField("transcription"):
                    tr = resp.transcription
                    txt = (tr.results[0].normalized_text or tr.results[0].text) if tr.results else ""
                    if tr.eou:
                        if txt:
                            self.text_final = (self.text_final + " " + txt).strip()
                        self.partial = ""
                        log.info("FINAL: %r", txt)
                    else:
                        self.partial = txt
                        log.debug("PART: %r", txt)
        except grpc.aio.AioRpcError as e:
            log.error("RPC error: %s %s", e.code().name, e.details())
        except Exception:
            log.exception("Listener error")
        finally:
            code = details = None
            try:
                code = await self._call.code()
                details = await self._call.details()
            except Exception:
                pass
            log.info("listener done (code=%s, details=%s)", code, details)

    async def eof(self):
        """Закрыть поток: отправить None в очередь, дождаться listener, закрыть канал и дампер."""
        if self._closing:
            return
        self._closing = True
        self.closed = True

        await self.q.put(None)

        try:
            if self._listener_task:
                await asyncio.wait_for(self._listener_task, timeout=2.0)
        except asyncio.TimeoutError:
            self._listener_task.cancel()

        if self._channel:
            await self._channel.close()

        if self._dump is not None:
            try:
                self._dump.close()
            except Exception:
                pass

# -----------------------------------
# Собственно скрипт преобразования
# -----------------------------------

def _to_mono_float32(x):
    x = np.asarray(x)
    if x.ndim == 2 and x.shape[1] > 1:
        x = x.mean(axis=1)
    return x.astype(np.float32, copy=False)

def _normalize_gain(x, target_rms=0.055, max_gain=8.0):
    rms = float(np.sqrt(np.mean(x * x) + 1e-12))
    if rms < 1e-6:
        return x, rms, rms
    g = min(max_gain, target_rms / rms)
    y = np.clip(x * g, -1.0, 1.0)
    rms_out = float(np.sqrt(np.mean(y * y) + 1e-12))
    return y, rms, rms_out

def _resample_to_16k(x, sr):
    if sr == 16000: return x
    if sr == 48000: return resample_poly(x, up=1, down=3)
    if sr == 44100: return resample_poly(x, up=160, down=441)
    return resample_poly(x, up=16000, down=sr)

async def asr_stream_sber(audio_chunk, asr_state):
    """UI-колбэк. Полностью повторяем логику VOSK:
       - один аккумулятор acc_pcm в state,
       - готовим PCM16LE@16k (моно, нормализовано),
       - как только набралось ≥ SEND_MS — отправляем ВСЁ и очищаем аккумулятор.
    """
    if not isinstance(asr_state, dict):
        asr_state = {}

    # старт/переиспользование стрима
    s: SberStream | None = asr_state.get("sber")
    if (s is None) or getattr(s, "closed", False):
        s = SberStream()
        await s.start()
        asr_state["sber"] = s
        asr_state.setdefault("text", "")
        asr_state.setdefault("partial", "")
        log.info("Sber_stream (re)created")

    # аккумулятор для "готового" PCM (ровно то, что уйдёт в сеть)
    acc = asr_state.setdefault("acc_pcm", bytearray())

    # ожидаем (sr, ndarray)
    if not (isinstance(audio_chunk, tuple) and len(audio_chunk) == 2):
        live = (asr_state.get("text","") + " " + asr_state.get("partial","")).strip()
        return live, asr_state

    sr, data = audio_chunk
    log.info("asr_stream_sber: sr=%s shape=%s dtype=%s",
             sr, getattr(data, "shape", None), getattr(data, "dtype", None))

    # 1) моно + float32
    x = _to_mono_float32(data)

    # 2) нормализация (мягкая)
    x, rms_in, rms_out = _normalize_gain(x, target_rms=0.055, max_gain=8.0)
    log.debug("ui-norm: rms_in=%.5f -> rms_out=%.5f", rms_in, rms_out)

    # 3) ресемпл → 16k
    x16 = _resample_to_16k(x, sr)

    # 4) PCM16LE
    pcm16 = (np.clip(x16, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()

    # накапливаем и отправляем целиком по достижении порога
    acc.extend(pcm16)
    need_bytes = int(TARGET_SR * (SEND_MS / 1000.0)) * 2  # bytes на SEND_MS
    if len(acc) >= need_bytes:
        await s.send_bytes(bytes(acc))  # запись в WAV происходит внутри send_bytes
        acc.clear()

    # дать шанс listener'у обновить partial
    await asyncio.sleep(0)

    live = (s.text_final + " " + s.partial).strip()
    asr_state["text"], asr_state["partial"] = s.text_final, s.partial
    return live, asr_state