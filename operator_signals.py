"""Shared narrow negation guard; bare 'no' needs an actual operator offer."""
import re

_NEGATED=re.compile(r"(?:\b(?:не\s+(?:нуж\w*|хочу|надо)|без)\s+(?:\w+\s+){0,3}(?:оператор\w*|человек\w*)"
    r"|\bне\s+(?:переключ\w*|перевод\w*|соедин\w*|зови|позов\w*)[\w\s,]{0,40}\b(?:оператор\w*|человек\w*)"
    r"|\bоператор\w*\s+(?:мне\s+)?не\s+(?:нуж\w*|надо))",re.I)


def operator_refused(text):
    return bool(_NEGATED.search(str(text or "")))
