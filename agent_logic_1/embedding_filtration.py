# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
import os
from typing import List

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from langchain_core.documents import Document

# читаем настройки из ini через ваш config.py
from agent_logic_2 import config as c

logger = logging.getLogger(__name__)

# -----------------------------------
# Настройки из config.ini (с дефолтами)
# -----------------------------------
def _cfg(section: str, key: str, default: str) -> str:
    try:
        return c.config[section].get(key, default)
    except Exception:
        return default

def _cfg_bool(section: str, key: str, default: bool) -> bool:
    val = _cfg(section, key, str(default)).strip().lower()
    return val in ("1", "true", "yes", "on")

# Раздел и ключи — FILTRATION Сделано пока в обход config.py
FILTRATION_MODEL = _cfg("FILTRATION", "model", "cointegrated/rubert-tiny2")
FILTRATION_LOCAL_ONLY = _cfg_bool("FILTRATION", "local_only", True)   # офлайн-режим
FILTRATION_DEVICE = _cfg("FILTRATION", "device", "auto").lower()      # auto|cpu|cuda
FILTRATION_CACHE = _cfg("FILTRATION", "cache_dir", "") or None        # путь к кешу
MAX_LENGTH = int(_cfg("FILTRATION", "max_length", "256"))
BATCH_SIZE = int(_cfg("FILTRATION", "batch_size", "16"))

# Доп. настройка токенайзеров — через ini, чтобы не лезть в ENV
TOKENIZERS_PARALLELISM = _cfg("FILTRATION", "tokenizers_parallelism", "false")
os.environ["TOKENIZERS_PARALLELISM"] = TOKENIZERS_PARALLELISM

# Синглтоны (лениво)
_TOKENIZER: AutoTokenizer | None = None
_MODEL: AutoModel | None = None
_DEVICE: torch.device | None = None

def _pick_device() -> torch.device:
    if FILTRATION_DEVICE == "cpu":
        return torch.device("cpu")
    if FILTRATION_DEVICE == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if FILTRATION_DEVICE == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _load_filtration_model() -> tuple[AutoTokenizer, AutoModel, torch.device]:
    """Ленивая загрузка модели/токенайзера на CPU/CUDA, с поддержкой офлайна и кэша."""
    global _TOKENIZER, _MODEL, _DEVICE
    if _TOKENIZER is not None and _MODEL is not None and _DEVICE is not None:
        return _TOKENIZER, _MODEL, _DEVICE

    _DEVICE = _pick_device()

    hf_kwargs = {
        "local_files_only": FILTRATION_LOCAL_ONLY,
        "trust_remote_code": False,
    }
    if FILTRATION_CACHE:
        os.makedirs(FILTRATION_CACHE, exist_ok=True)
        hf_kwargs["cache_dir"] = FILTRATION_CACHE

    try:
        logger.info(
            f"[filtration] Loading '{FILTRATION_MODEL}' "
            f"(local_only={FILTRATION_LOCAL_ONLY}, device={_DEVICE.type}) ..."
        )
        _TOKENIZER = AutoTokenizer.from_pretrained(FILTRATION_MODEL, **hf_kwargs)
        _MODEL = AutoModel.from_pretrained(FILTRATION_MODEL, **hf_kwargs)
        _MODEL = _MODEL.to(_DEVICE).eval()
        logger.info("[filtration] Model loaded successfully.")
        return _TOKENIZER, _MODEL, _DEVICE

    except Exception as e:
        # Не валим сервис — подняли понятную ошибку, верхний код может решить стратегию
        msg = (
            f"[filtration] Не удалось загрузить '{FILTRATION_MODEL}'. "
            f"offline={FILTRATION_LOCAL_ONLY}. "
            f"Подготовьте кэш (config.ini→FILTRATION.cache_dir и смонтируйте том в контейнер) "
            f"или временно поставьте local_only=false для прогрева. Ошибка: {e}"
        )
        logger.error(msg)
        raise RuntimeError(msg) from e

def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = (last_hidden_state * mask).sum(1)
    counts = mask.sum(1).clamp(min=1e-9)
    return summed / counts

def _embed_texts(texts: List[str]) -> torch.Tensor:
    """Batched эмбеддинги -> [N, D], L2-нормированные. Без градентов, с обрезкой длины."""
    tokenizer, model, device = _load_filtration_model()

    all_embeds = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            enc = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            emb = _mean_pool(out.last_hidden_state, enc["attention_mask"])
            emb = F.normalize(emb, p=2, dim=1)
            all_embeds.append(emb.detach().to("cpu"))

    return torch.cat(all_embeds, dim=0) if all_embeds else torch.empty(0, 0)

def get_embedding(text: str) -> torch.Tensor:
    embs = _embed_texts([text])
    return embs if embs.ndim == 2 else embs.unsqueeze(0)

def filtrate(keyword: str, texts: List[Document] | None, threshold: float = 0.005) -> List[Document]:
    """
    Динамический порог: пропускаем документы с sim >= max(sim) - threshold.
    При ошибке загрузки модели — возвращаем вход как есть (не блокируем сервис).
    """
    print("- Section FILTRATE -")
    print("keyword:", keyword)

    if not texts:
        print("Document list is empty or None — nothing to filtrate.")
        return []

    try:
        k_emb = _embed_texts([keyword])                           # [1, D]
        d_emb = _embed_texts([d.page_content for d in texts])     # [N, D]
        if d_emb.numel() == 0:
            return []

        sims = (k_emb @ d_emb.T).squeeze(0).tolist()              # косинус = dot (нормировано)
        max_sim = max(sims) if sims else -1.0
        dynamic_threshold = max_sim - threshold
        print("Dynamic Threshold:", dynamic_threshold)

        kept = [doc for doc, sim in zip(texts, sims) if sim >= dynamic_threshold]
        return kept

    except Exception:
        logger.exception("[filtration] Ошибка в фильтрации — возвращаю документы без фильтрации.")
        return texts  # мягкий фолбэк; можно заменить на [] для строгого режима
