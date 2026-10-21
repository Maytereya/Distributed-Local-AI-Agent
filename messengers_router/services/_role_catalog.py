"""Каталог ролей приёма из среза врачей МИС и точный слой «слово пациента → роль» (ход (г), L-17, этап 1).

Роль — `companyUnit` врача (`unit_links[]`): «Врач-кардиолог», «Врач-сердечно-сосудистый хирург, флеболог»;
это роль приёма, а не филиал и не юрлицо. Каталог строится из среза при каждой загрузке — в коде нет списка
специальностей: новую роль МИС бот узнаёт по её же названию без правки кода, исчезнувшую — перестаёт.

Точный слой — механика над названиями МИС, без синонимов из кода:
- `exact` — названа роль целиком, в любом падеже («к сердечно-сосудистому хирургу»); роль с уточнением
  поглощает общую («пластическому хирургу» — не «хирург»);
- `partial` — названа часть составной роли («маммолог» при роли «гинеколог-маммолог»). Это кандидат для
  выбора LLM (`choose()`), а не ответ: «похожий, но другой врач» автоматически не подставляется (решение
  владельца 08.10).

Пустой результат — не «у нас нет»: что отвечать, решает вызывающий (дизайн
`docs/DESIGN_2026-10-20_step_g_roles_catalog.md`). Модуль к ответам пока не подключён (этап 1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

from ..russian_nlu import normalize_ru
from ._symptom_specialists import is_reception_role, specialty_display_name

_ADJ_ENDINGS = ("ый", "ий", "ой", "ая")
_VOWELS = "аеиоуыэюяй"
_SHORT = 4  # основа не длиннее — только целое слово («узи», «лор»), иначе «лорн…» и «узин…» совпали бы


def _tokens(text: str) -> tuple[str, ...]:
    """Слова текста: нижний регистр, ё → е, дефис и знаки — разделители."""

    return tuple("".join(ch if ch.isalnum() else " " for ch in normalize_ru(text)).split())


def _stem(token: str) -> str:
    """Основа слова названия роли, совпадающая с любым падежом: «хирург» → «хирург» («хирургу»),
    «пластический» → «пластическ», «диагностики» → «диагностик» («диагностика»)."""

    if len(token) > 5 and token.endswith(_ADJ_ENDINGS):
        return token[:-2]
    stem = token
    while len(stem) > _SHORT + 1 and stem[-1] in _VOWELS:
        stem = stem[:-1]
    return stem


def _hit(stem: str, words: Sequence[str]) -> bool:
    if len(stem) <= _SHORT:
        return stem in words
    return any(word.startswith(stem) for word in words)


@dataclass(frozen=True)
class Role:
    """Роль приёма: одна или несколько записей `companyUnit` с одним отображаемым названием."""

    name: str  # отображаемое: «сердечно-сосудистый хирург, флеболог»
    unit_ids: tuple[int, ...]
    mis_names: tuple[str, ...]  # как в МИС: «Врач-сердечно-сосудистый хирург, флеболог»
    atoms: tuple[str, ...]  # специальности роли: («сердечно-сосудистый хирург», «флеболог»)
    doctors: int  # врачей в регионе с этой ролью
    branch_ids: tuple[int, ...]  # филиалы (`region_ids`) этих врачей
    reception: bool  # к роли приходят на приём (не диагностика, не процедура, не служебная)


@dataclass(frozen=True)
class RoleMatch:
    exact: tuple[Role, ...] = ()
    partial: tuple[Role, ...] = ()


@dataclass(frozen=True)
class RoleCatalog:
    roles: tuple[Role, ...] = ()

    def by_unit(self, unit_id: int) -> Role | None:
        return next((role for role in self.roles if unit_id in role.unit_ids), None)

    def find(self, text: str) -> RoleMatch:
        """Роли, названные в тексте, — только по названиям МИС (см. docstring модуля)."""

        words = _tokens(text)
        if not words:
            return RoleMatch()
        full: list[tuple[Role, frozenset[str]]] = []
        part: list[tuple[Role, frozenset[str]]] = []
        for role in self.roles:
            for atom in role.atoms:
                stems = [_stem(token) for token in _tokens(atom)]
                hits = frozenset(stem for stem in stems if _hit(stem, words))
                if hits and len(hits) == len(set(stems)):
                    full.append((role, frozenset(stems)))
                    break
                # Часть составной роли — только существительное («маммолог»), не прилагательное («пластическ»),
                # и только у роли приёма: служебная «<фамилия> -прием» иначе ловит каждое «приём».
                nouns = {stem for stem, token in zip(stems, _tokens(atom)) if not token.endswith(_ADJ_ENDINGS)}
                if role.reception and len(stems) > 1 and hits & nouns:
                    part.append((role, hits))
        # Роль с уточнением поглощает общую: «сердечно-сосудистый хирург» — не ещё и «хирург».
        exact = [
            (role, stems) for role, stems in full if not any(stems < other for other_role, other in full if other_role != role)
        ]
        covered = frozenset().union(*(stems for _, stems in exact))
        partial = [role for role, hits in part if not hits <= covered and role not in {r for r, _ in exact}]
        return RoleMatch(exact=_unique(role for role, _ in exact), partial=_unique(partial))


def _unique(roles: Iterable[Role]) -> tuple[Role, ...]:
    return tuple(dict.fromkeys(roles))


def build_role_catalog(doctors: Sequence[dict[str, Any]], *, in_region: Callable[[dict[str, Any]], bool]) -> RoleCatalog:
    """Каталог ролей приёма врачей региона — чистая функция среза.

    Записи `companyUnit` с одним отображаемым названием («Врач-ревматолог», «Ревматолог (центр)») — одна роль.
    Порядок — по числу врачей, затем по названию: каталог не зависит от порядка строк среза.

    :param doctors: карточки врачей из среза МИС (`doctors_*.jsonl`)
    :param in_region: врач принимает в регионе бота (у вызывающего — признаки филиалов региона)
    :return: каталог; пустой срез — пустой каталог (решать «нет» по нему нельзя — ход (д))
    """

    acc: dict[str, dict[str, Any]] = {}
    for doctor in doctors:
        if not isinstance(doctor, dict) or not in_region(doctor):
            continue
        branches = [int(x) for x in (doctor.get("region_ids") or []) if str(x).strip().lstrip("-").isdigit()]
        seen: set[str] = set()
        for link in doctor.get("unit_links") or []:
            mis_name = " ".join(str((link or {}).get("company_unit_name") or "").split())
            name = specialty_display_name(mis_name)
            if not name:
                continue
            slot = acc.setdefault(name, {"ids": {}, "mis": {}, "doctors": 0, "branches": {}})
            unit_id = (link or {}).get("company_unit_id")
            if isinstance(unit_id, int):
                slot["ids"].setdefault(unit_id, None)
            slot["mis"].setdefault(mis_name, None)
            slot["branches"].update(dict.fromkeys(branches))
            if name not in seen:
                slot["doctors"] += 1
                seen.add(name)
    roles = [
        Role(
            name=name,
            unit_ids=tuple(sorted(slot["ids"])),
            mis_names=tuple(sorted(slot["mis"])),
            atoms=tuple(dict.fromkeys(a.strip() for a in name.split(",") if a.strip())),
            doctors=slot["doctors"],
            branch_ids=tuple(sorted(slot["branches"])),
            reception=is_reception_role(name),
        )
        for name, slot in acc.items()
    ]
    roles.sort(key=lambda role: (-role.doctors, role.name))
    return RoleCatalog(roles=tuple(roles))


__all__ = ["Role", "RoleCatalog", "RoleMatch", "build_role_catalog"]
