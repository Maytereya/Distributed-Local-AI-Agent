"""Какого врача назвал бот: специальности МИС в тексте ответа против разметки «симптом → врач».

Приоритет владельца 09.10: на жалобу бот правильно рекомендует врача. Словарь — роли приёма
из среза врачей МИС (тот же список, из которого выбирает бот), составные роли («хирург,
флеболог») разбиты на отдельные специальности. Разговорные формы («лор», «гинеколог»,
«окулист») и падежи («к неврологу») приводятся к названию МИС.

Вердикты по разметке `core` (к кому идут в первую очередь) и `ok` (тоже разумно):
- `right` — названы только допустимые, есть хотя бы один из `core`;
- `acceptable` — названы только допустимые, но из `ok`;
- `partial` — среди названных есть и допустимые, и лишние;
- `wrong` — допустимых среди названных нет;
- `none` — специальность не названа.
"""

from __future__ import annotations

import re
from typing import Iterable

# Разговорное название → специальность МИС (только однозначные).
ALIASES = {
    "лор": "оториноларинголог",
    "отоларинголог": "оториноларинголог",
    "гинеколог": "акушер-гинеколог",
    "дерматолог": "дерматовенеролог",
    "венеролог": "дерматовенеролог",
    "окулист": "офтальмолог",
    "проктолог": "колопроктолог",
    "аллерголог": "аллерголог-иммунолог",
    "андролог": "уролог-андролог",
    "ортопед": "травматолог-ортопед",
    "травматолог": "травматолог-ортопед",
    "кардиохирург": "сердечно-сосудистый хирург",
}
_SUFFIX = r"(?:у|а|ом|ам|ы|ов|ами|ах|е|и)?"


def _form_pattern(form: str) -> str:
    """Шаблон названия с падежами: склоняется каждая часть — через пробел и через дефис
    («мануальному терапевту», «урологу-андрологу»)."""

    out = []
    for piece in re.split(r"(\s+|-)", form):
        if not piece or piece.isspace():
            out.append(r"\s+" if piece else "")
        elif piece == "-":
            out.append("-")
        elif len(piece) > 5 and form.count(" ") and piece.endswith(("ый", "ий", "ой", "ая")):
            out.append(re.escape(piece[:-2]) + r"[а-яё]{0,4}")
        else:
            out.append(re.escape(piece) + _SUFFIX)
    return "".join(out)


def atoms(specialties: Iterable[str]) -> list[str]:
    """Отдельные специальности из ролей МИС («хирург, флеболог» → две)."""

    return sorted({a.strip().lower() for s in specialties for a in str(s).split(",") if a.strip()})


class SpecialtyMatcher:
    def __init__(self, vocabulary: Iterable[str]) -> None:
        names = {a: a for a in atoms(vocabulary)}
        names.update({alias: target for alias, target in ALIASES.items() if target in names})
        # Длинные — первыми: «уролог-андролог» раньше «уролог», «мануальный терапевт» раньше «терапевт».
        self._patterns = [
            (re.compile(rf"(?<![а-яё\-]){_form_pattern(form)}(?![а-яё\-])"), target)
            for form, target in sorted(names.items(), key=lambda kv: -len(kv[0]))
        ]

    def named(self, text: str) -> list[str]:
        """Специальности, названные в тексте, в порядке появления, без повторов."""

        low = str(text or "").lower().replace("ё", "е")
        found: list[tuple[int, str]] = []
        taken: list[tuple[int, int]] = []
        for pattern, target in self._patterns:
            for m in pattern.finditer(low):
                if any(a < m.end() and m.start() < b for a, b in taken):
                    continue
                taken.append((m.start(), m.end()))
                found.append((m.start(), target))
        out: list[str] = []
        for _, target in sorted(found):
            if target not in out:
                out.append(target)
        return out


def specialty_verdict(named: list[str], core: Iterable[str], ok: Iterable[str], roles: Iterable[str] = ()) -> str:
    core_set, ok_set = set(core), set(ok)
    allowed = core_set | ok_set
    shown = set(named)
    # Составная роль МИС («сердечно-сосудистый хирург, флеболог, лимфолог») — один врач: если допустима
    # одна её часть, остальные части лишними не считаются.
    for role in roles:
        parts = set(atoms([role]))
        if len(parts) > 1 and parts & shown & allowed:
            shown -= parts - allowed
    if not shown:
        return "none"
    if not shown & allowed:
        return "wrong"
    if shown - allowed:
        return "partial"
    return "right" if shown & core_set else "acceptable"
