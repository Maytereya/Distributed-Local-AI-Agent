from __future__ import annotations
import json, wave, uuid, time
from pathlib import Path
from dataclasses import dataclass, asdict

@dataclass
class DumpMeta:
    sample_rate: int
    channels: int
    sample_width: int
    send_ms: int
    target_rms: float | None = None
    max_gain: float | None = None
    note: str | None = None

class PcmDump:
    """
    Пишет WAV (PCM16 mono @16k) + метаданные JSON.
    Вызывается из SberStream после всей обработки (моно→нормализация→16k→PCM16).
    """
    def __init__(self, out_dir: str | Path, meta: DumpMeta, prefix: str = "sber_out"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{prefix}_{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:8]}"
        self.wav_path = self.out_dir / f"{stem}.wav"
        self.json_path = self.out_dir / f"{stem}.json"
        self.meta = meta
        self._w: wave.Wave_write | None = None
        self._bytes = 0

    def open(self):
        if self._w is None:
            self._w = wave.open(str(self.wav_path), "wb")
            self._w.setnchannels(self.meta.channels)      # 1
            self._w.setsampwidth(self.meta.sample_width)  # 2 байта
            self._w.setframerate(self.meta.sample_rate)   # 16000

    def write(self, pcm_bytes: bytes):
        if not pcm_bytes:
            return
        self.open()
        self._w.writeframes(pcm_bytes)
        self._bytes += len(pcm_bytes)

    def close(self):
        if self._w is not None:
            self._w.close()
            self._w = None
        # пишем JSON с метой и длительностью
        duration_s = self._bytes / (self.meta.sample_rate * self.meta.sample_width * self.meta.channels)
        payload = asdict(self.meta) | {"duration_sec": round(duration_s, 3), "bytes": self._bytes}
        self.json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    def __enter__(self): self.open(); return self
    def __exit__(self, exc_type, exc, tb): self.close()