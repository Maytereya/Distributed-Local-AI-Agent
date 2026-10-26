"""Встречный свип `strong_family_letter` (BUG-2026-10-27-STRONG-DROPS-LETTER): где выключатель меняет сильное совпадение.

Офлайн, мир стенда (`~/Dev/dla-private/bench/worlds/<дата>`, сеть закрыта): прайс и словарь синонимов МИС — срезы мира. Для
каждой фразы — `strong_rows` при `strong_family_letter = off` и `on`; печатаются фразы, где строки разошлись. Фразы:
- `name` — каждое название прайса (дословное название обязано остаться сильным);
- `synonym` — каждый ключ словаря синонимов;
- `family` — «<слово семейства> <буква>» по всем буквам семейства, кириллицей и латиницей, и слово без буквы;
- `swap` — ключ синонима с буквой семейства, где буква заменена на другую букву семейства или снята;
- `designator` — «<слово> <различитель>» для различителей вне буквенных семейств (цифры, «b12», «igg», «т4»): `on` не должен
  ничего менять, а сильная строка не должна нести чужой различитель без названного.

Первой строкой — размер словаря синонимов (ловушка CLAUDE.md «ночной словарь пуст»): 0 — замер недействителен.

Запуск: `./venv/bin/python scripts/sweep_strong_family_letter.py [--world 20261008] [--parse on|off] [--tsv путь]`.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evalbench import world as ev_world  # noqa: E402

_LOOKALIKE = {"a": "а", "b": "в", "c": "с", "e": "е", "k": "к", "d": "д", "g": "г"}
_DESIGNATOR = re.compile(r"^(?:[a-zа-я]{1,2}\d{1,3}|\d{1,2}|ig[agmde])$")  # буквы — набор `family`


def _other_designator(name: str, anchor: str, token: str) -> bool:
    """Строка несёт после `anchor` другой различитель и не несёт названный (`token`)."""

    from messengers_router.services._biomaterial import _tokens

    from messengers_router.services._prices_helpers import _fold_family_letter

    tokens = [_fold_family_letter(tok) for tok in _tokens(name)]
    token = _fold_family_letter(token)
    if token in tokens:
        return False
    return any(prev == anchor and other != token and _DESIGNATOR.match(other) for prev, other in zip(tokens, tokens[1:]))


def _phrases(rows: list[dict[str, Any]], vocab: Any, families: dict[str, frozenset[str]]) -> list[tuple[str, str]]:
    from messengers_router.services._biomaterial import _tokens
    from messengers_router.services._strong_match import _name, family_letters

    names = sorted({_name(row) for row in rows} - {""})
    out = [("name", name) for name in names] + [("synonym", key) for key in sorted(vocab.synonym_candidates)]
    forms: dict[str, set[str]] = {}
    for name in names:
        tokens = _tokens(name)
        for prev in tokens:
            if prev.isalpha() and len(prev) >= 6 and prev[:6] in families:
                forms.setdefault(prev[:6], set()).add(prev)
    for stem, letters in sorted(families.items()):
        for form in sorted(forms.get(stem, ())):
            out.append(("family", form))
            for letter in sorted(letters):
                for spelled in {letter, _LOOKALIKE.get(letter, letter)}:
                    out += [("family", f"{form} {spelled}"), ("family", f"анализ на {form} {spelled}")]
    for key in sorted(vocab.synonym_candidates):
        named = family_letters(key, families)
        if not named:
            continue
        tokens = _tokens(key)
        for pos, (prev, token) in enumerate(zip(tokens, tokens[1:]), start=1):
            if len(token) == 1 and prev[:6] in families:
                for other in sorted(families[prev[:6]]):
                    swapped = [*tokens[:pos], _LOOKALIKE.get(other, other), *tokens[pos + 1 :]]
                    out.append(("swap", " ".join(swapped)))
                out.append(("swap", " ".join([*tokens[:pos], *tokens[pos + 1 :]])))
    anchors: dict[str, set[str]] = {}
    for name in names:
        tokens = _tokens(name)
        for prev, token in zip(tokens, tokens[1:]):
            if prev.isalpha() and len(prev) >= 2 and _DESIGNATOR.match(token):
                anchors.setdefault(prev, set()).add(token)
    for anchor, tokens in sorted(anchors.items()):
        if len(tokens) >= 2:
            out += [("designator", f"{anchor} {token}") for token in sorted(tokens)]
    return list(dict.fromkeys(out))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--world", default="20261008")
    ap.add_argument("--parse", default="on", choices=("on", "off"), help="price_by_parse (прод с 26.10 — on)")
    ap.add_argument("--tsv", default="", help="куда записать все фразы с разошедшимися строками")
    args = ap.parse_args()
    with ev_world.FrozenWorld(ev_world.PRIVATE_ROOT / "worlds" / args.world, "day") as world:
        from agent_logic_2.nayka_api import api_price
        from messengers_router.runtime_config import config as cfg
        from messengers_router.services import _biomaterial
        from messengers_router.services._prices_helpers import SAMARA_PRICE_REGION_ID
        from messengers_router.services._strong_match import (
            _letter_families,
            _name,
            _names_other_letter,
            family_letters,
            strong_rows,
        )

        cfg.MR_PRICE_BY_PARSE = args.parse
        rows = [row for row in api_price.load_price_by_region(SAMARA_PRICE_REGION_ID) if isinstance(row, dict)]
        vocab = _biomaterial._vocabularies()
        families = _letter_families(rows)
        print(f"словарь синонимов: {len(vocab.synonym_candidates)} ключей; прайс: {len(rows)} строк; price_by_parse = {args.parse}")
        if not vocab.synonym_candidates:
            print("словарь пуст — замер недействителен")
            return 2
        print("семейства:", {stem: "".join(sorted(letters)) for stem, letters in sorted(families.items())})
        changed: list[tuple[str, str, list[str], list[str]]] = []
        foreign: list[tuple[str, str, list[str]]] = []
        sizes: Counter[str] = Counter()
        for kind, phrase in _phrases(rows, vocab, families):
            sizes[kind] += 1
            cfg.MR_STRONG_FAMILY_LETTER = "off"
            before = [_name(row) for row in strong_rows(phrase, rows)]
            cfg.MR_STRONG_FAMILY_LETTER = "on"
            after = [_name(row) for row in strong_rows(phrase, rows)]
            if before != after:
                changed.append((kind, phrase, before, after))
            named = family_letters(phrase, families)
            wrong = [name for name in after if named and _names_other_letter(name, named, families)]
            if kind == "designator":
                anchor, token = phrase.rsplit(" ", 1)
                wrong += [name for name in after if _other_designator(name, anchor, token)]
            if wrong:
                foreign.append((kind, phrase, wrong))
        print("фраз:", dict(sizes), "| сеть:", world.net.attempts)
        by_kind = Counter(kind for kind, *_ in changed)
        print(f"разошлось: {len(changed)} {dict(by_kind)}; чужая буква/различитель у сильной строки под on: {len(foreign)}")
        for kind, phrase, before, after in changed:
            print(f"  [{kind}] {phrase!r}: {before} → {after}")
        for kind, phrase, wrong in foreign:
            print(f"  ЧУЖАЯ [{kind}] {phrase!r}: {wrong}")
        if args.tsv:
            with open(args.tsv, "w", encoding="utf-8", newline="") as fh:
                out = csv.writer(fh, delimiter="\t")
                out.writerow(("kind", "phrase", "off", "on"))
                for kind, phrase, before, after in changed:
                    out.writerow((kind, phrase, " | ".join(before), " | ".join(after)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
