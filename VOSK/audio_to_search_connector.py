import re
def _extract_keywords(text: str, top_k: int = 6) -> str:
    """
    Обработка от лишних слов, букв и звуков стрима, возвращенного из websocket
    :param text: чанк текста на обработку
    :param top_k:
    :return:
    """
    tokens = re.findall(r"[A-Za-zА-Яа-яёЁ0-9\-]+", text)
    stop = {"и", "в", "на", "по", "из", "к", "с", "что", "это", "для", "как", "а", "но", "же"}
    freq = {}
    for t in tokens:
        t = t.lower()
        if t in stop or len(t) < 3:
            continue
        freq[t] = freq.get(t, 0) + 1
    kws = [w for w, _ in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_k]]
    return " ".join(kws) if kws else text