# smoke_sber_grpc.py
import os
import grpc
from google.protobuf import duration_pb2
# print(duration_pb2.Duration)  # ← если выведет <class 'google.protobuf.duration_pb2.Duration'> — ОК
# print(duration_pb2.Duration(seconds=5))  # → seconds: 5

import agent_logic_2.config as c
from sber_salut_operations import sber_proto as rec_pb2, sber_proto as rec_grpc

BUNDLE = os.path.abspath(c.SBER_CA)

ssl_cred = grpc.ssl_channel_credentials(open(BUNDLE, "rb").read())
tok_cred = grpc.access_token_call_credentials(c.SBER_TOKEN)

channel = grpc.secure_channel(
    c.SBER_HOST, grpc.composite_channel_credentials(ssl_cred, tok_cred)
)
stub = rec_grpc.SmartSpeechStub(channel)

# ← ВАЖНО: создаём через kwargs
opts = rec_pb2.RecognitionOptions(
    audio_encoding=rec_pb2.RecognitionOptions.AudioEncoding.PCM_S16LE,
    sample_rate=16000,
    channels_count=1,
    enable_partial_results=rec_pb2.OptionalBool(enable=False),
    enable_multi_utterance=rec_pb2.OptionalBool(enable=False),
    no_speech_timeout=duration_pb2.Duration(seconds=7),
    max_speech_timeout=duration_pb2.Duration(seconds=20),
    # hints (по желанию):
    # hints=rec_pb2.Hints(words=["Дразнин","медкарта"], enable_letters=True, eou_timeout=Duration(seconds=0)),
)

wav_path = os.path.abspath("../data/sample.wav")  # 16kHz mono PCM S16LE
with open(wav_path, "rb") as f:
    audio = f.read()

reqs = [
    rec_pb2.RecognitionRequest(options=opts),
    rec_pb2.RecognitionRequest(audio_chunk=audio),
]

for resp in stub.Recognize(iter(reqs)):
    if resp.HasField("transcription"):
        tr = resp.transcription
        for hyp in tr.results:
            txt = hyp.normalized_text or hyp.text
            print(("FINAL: " if tr.eou else "PART:  "), txt)