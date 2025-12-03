import asyncio, math, numpy as np
from sber_asr_stream import SberStream

async def main():
    s = SberStream()
    await s.start()
    # 1 сек синуса 440 Гц @ 48k → send_wave сам приведёт к 16k и PCM
    sr = 48000
    t = np.arange(sr, dtype=np.float32) / sr
    data = 0.1 * np.sin(2*math.pi*440*t).astype(np.float32)
    await s.send_wave(sr, data)
    await s.eof()
    print("RESULT:", s.text_final)

if __name__ == "__main__":
    asyncio.run(main())