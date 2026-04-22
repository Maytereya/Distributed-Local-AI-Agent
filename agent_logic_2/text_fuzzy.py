from __future__ import annotations

from difflib import SequenceMatcher
from typing import Iterable


def normalize_text_for_fuzzy(text: str) -> str:
    """
    Нормализует текст для нестрогого сравнения:
    - приводит к нижнему регистру;
    - заменяет «ё» на «е»;
    - удаляет пунктуацию и символы;
    - схлопывает пробелы.
    """
    if text is None:
        return ""
    s = str(text).lower().strip()
    s = s.replace("ё", "е")

    out: list[str] = []
    prev_space = False
    for ch in s:
        if ch.isalnum():
            out.append(ch)
            prev_space = False
        else:
            if not prev_space:
                out.append(" ")
                prev_space = True
    return "".join(out).strip()


def _bigrams(tokens: list[str]) -> list[str]:
    """Возвращает список склеенных биграмм из токенов."""
    if len(tokens) < 2:
        return []
    return ["".join(tokens[i:i + 2]) for i in range(len(tokens) - 1)]


def _max_pair_ratio(left: Iterable[str], right: Iterable[str]) -> float:
    """Ищет максимальную схожесть среди пар элементов двух коллекций."""
    best = 0.0
    for a in left:
        for b in right:
            r = SequenceMatcher(None, a, b).ratio()
            if r > best:
                best = r
                if best >= 0.99:
                    return best
    return best


def fuzzy_ratio(a: str, b: str) -> float:
    """
    Возвращает наилучший коэффициент схожести между двумя строками.
    Сравнивает нормализованные варианты: базовый, склеенный, по токенам и биграммам.
    """
    a_norm = normalize_text_for_fuzzy(a)
    b_norm = normalize_text_for_fuzzy(b)
    if not a_norm or not b_norm:
        return 0.0

    scores = [SequenceMatcher(None, a_norm, b_norm).ratio()]

    a_join = a_norm.replace(" ", "")
    b_join = b_norm.replace(" ", "")
    if a_join and b_join:
        scores.append(SequenceMatcher(None, a_join, b_join).ratio())

    a_tokens = a_norm.split()
    b_tokens = b_norm.split()
    if a_tokens and b_tokens:
        scores.append(_max_pair_ratio(a_tokens, b_tokens))

    a_bi = _bigrams(a_tokens)
    b_bi = _bigrams(b_tokens)
    if a_bi and b_bi:
        scores.append(_max_pair_ratio(a_bi, b_bi))

    return max(scores)


def fuzzy_match(a: str, b: str, threshold: float = 0.85) -> bool:
    """
    Проверяет, что схожесть >= порога.
    Порог 0.85–0.9 обычно безопасен для коротких медицинских терминов.
    """
    return fuzzy_ratio(a, b) >= threshold
