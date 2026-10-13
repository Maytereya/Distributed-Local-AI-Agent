"""Храповик CI: список тестов, зелёных без локальных срезов и config.ini, только растёт.

CI (`.github/workflows/tests.yml`, Python 3.12) гоняет файлы из `tests/hermetic_files.txt`. До 10.10
там было 7 файлов; копия CI (3.12 + requirements-ci.txt, без конфига и срезов) показала, что
герметичны 102. Список не должен тихо сжиматься, а каждый его файл — существовать.
"""

from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_FLOOR = 117  # 13.10.2026; поднимать в том же коммите, где файл добавлен в список


def test_hermetic_list_only_grows_and_every_file_exists():
    listed = [line.strip() for line in (_ROOT / "tests" / "hermetic_files.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(listed) == len(set(listed)), "дубли в списке"
    assert len(listed) >= _FLOOR, f"список герметичных тестов сжался: {len(listed)} < {_FLOOR}"
    missing = [f for f in listed if not (_ROOT / f).is_file()]
    assert missing == [], missing
