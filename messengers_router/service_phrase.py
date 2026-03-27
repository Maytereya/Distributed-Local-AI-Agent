"""Domain-функции извлечения процедурных фраз из пользовательского текста.

Модуль не зависит от policy/infrastructure и может безопасно
использоваться из разных слоёв (classifier/services/entity_grounder).
"""

from __future__ import annotations

import re

from .city import match_city

SERVICE_ANCHORS = (
    "узи",
    "экг",
    "холтер",
    "мрт",
    "кт",
    "фгдс",
    "фдгс",
    "фкс",
    "эндоскоп",
    "эндоскопия",
    "гастроскоп",
    "ректороманоскоп",
    "эзофагогастродуоденоскоп",
    "рентген",
    "флюорограф",
    "колоноскоп",
)

SERVICE_BOUNDARY_WORDS = {
    "в", "во", "на", "к", "ко", "по", "из", "до", "после",
    "и", "или", "но",
    "сегодня", "завтра", "понедельник", "вторник", "среда", "четверг",
    "пятница", "суббота", "воскресенье",
    "января", "февраля", "марта", "апреля", "мая", "июня",
    "июля", "августа", "сентября", "октября", "ноября", "декабря",
}

SERVICE_UPPERCASE = {"узи", "экг", "мрт", "кт", "фгдс", "фкс", "уздг"}
SERVICE_CANONICAL_TOKENS = {
    "фдгс": "фгдс",
    "фгс": "фгдс",
    "егдс": "фгдс",
    "эгдс": "фгдс",
}
SERVICE_GENERIC_STOPWORDS = {
    "хочу",
    "хотел",
    "хотела",
    "нужно",
    "надо",
    "можно",
    "пожалуйста",
    "записаться",
    "записать",
    "запись",
    "врач",
    "врачу",
    "врача",
    "доктор",
    "доктору",
    "доктора",
    "специалист",
    "специалисту",
    "специалиста",
    "прием",
    "приём",
    "услуга",
    "услуги",
    "услугу",
    "процедура",
    "процедуры",
    "процедуру",
}
_SERVICE_FOLLOWUP_RE = re.compile(
    r"\b(?:услуг\w*|процедур\w*|обследовани\w*|на|по|к)\s+([a-zа-яё0-9\- ]{2,80})",
    re.I,
)
_SERVICE_ACTION_RE = re.compile(
    r"\b(?:выполняет|делает|проводит|сделать|провести|выполнить|удалить|удаление|убрать)\s+([a-zа-яё0-9\- ]{2,80})",
    re.I,
)
_SERVICE_SINGLE_WORD_RE = re.compile(r"^\s*([a-zа-яё][a-zа-яё0-9\-]{3,})\s*$", re.I)


def extract_service_phrase(text: str) -> str | None:
    """
    Выделяет компактную фразу услуги/процедуры из пользовательского текста.

    Пример:
    - "записаться на узи печени в самаре" -> "УЗИ печени"

    :param text: текст сообщения пользователя
    :return: нормализованная фраза услуги или None
    """
    if not isinstance(text, str) or not text.strip():
        return None

    raw = text.strip()
    low = text.lower()
    m = re.search(r"\b(" + "|".join(SERVICE_ANCHORS) + r")\b", low)

    def _normalize_tokens(tokens: list[str]) -> str | None:
        if not tokens:
            return None
        service_tokens: list[str] = []
        for idx, tok in enumerate(tokens):
            token = tok.strip().lower()
            token = SERVICE_CANONICAL_TOKENS.get(token, token)
            if not token:
                continue
            if len(service_tokens) >= 5:
                break
            if token in SERVICE_BOUNDARY_WORDS:
                break
            if ":" in token or re.search(r"\d", token):
                break
            if idx == 0 and token in SERVICE_GENERIC_STOPWORDS:
                continue
            service_tokens.append(token)
        if not service_tokens:
            return None
        if len(service_tokens) == 1 and service_tokens[0] in SERVICE_GENERIC_STOPWORDS:
            return None
        first = service_tokens[0]
        first_out = first.upper() if first in SERVICE_UPPERCASE else first.capitalize()
        if len(service_tokens) == 1:
            return first_out
        return " ".join([first_out, *service_tokens[1:]])

    if m:
        tail = low[m.start():]
        tokens = re.findall(r"[a-zа-яё0-9:-]+", tail)
        normalized = _normalize_tokens(tokens)
        if normalized:
            return normalized

    # Fallback: поддержка услуг без "якорей" (пример: "на торакоцентез").
    # Берем фразу после маркеров "услуга/процедура/на/по" и нормализуем.
    f = _SERVICE_FOLLOWUP_RE.search(low)
    if f:
        cand = f.group(1).strip()
        cand_tokens = re.findall(r"[a-zа-яё0-9:-]+", cand)
        normalized = _normalize_tokens(cand_tokens)
        if normalized and not match_city(normalized):
            return normalized

    # Дополнительный fallback для фраз вида:
    # "какой врач выполняет кольпоскопию".
    a = _SERVICE_ACTION_RE.search(low)
    if a:
        cand = a.group(1).strip()
        cand_tokens = re.findall(r"[a-zа-яё0-9:-]+", cand)
        normalized = _normalize_tokens(cand_tokens)
        if normalized and not match_city(normalized):
            return normalized

    # Однословный ввод в active clarify ("торакоцентез").
    sw = _SERVICE_SINGLE_WORD_RE.match(raw)
    if sw:
        token = sw.group(1).strip().lower()
        token = SERVICE_CANONICAL_TOKENS.get(token, token)
        if token not in SERVICE_GENERIC_STOPWORDS and not match_city(token):
            return token.upper() if token in SERVICE_UPPERCASE else token.capitalize()

    return None
