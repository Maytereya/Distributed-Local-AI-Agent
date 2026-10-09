"""Цены в тексте ответа против прайса среза — независимо от кода цен бота.

Резолвер цен бота оракулом быть не может: он и есть то, что меряем. Независимая проверка
опирается на два факта о данных (проверено на срезе 09.10: 3671 строка, 3543 названия, ни
одно название не встречается с двумя разными ценами) и о рендере (название строки прайса
печатается дословно, сумма — `N NNN руб.`):

- каждая сумма в строке ответа ищет в той же строке самое длинное название прайса с этой
  ценой — нашлось: пара (serviceId, цена);
- сумма, которой нет ни у одной строки прайса и ни у одной цены врача, — «цена без
  источника» (принцип 1: факты печатает код из МИС);
- сумма есть в данных, но название в строке не найдено — «без привязки» (строки врачей
  «ФИО — N руб.», склейки): считается отдельно, в метрику не идёт.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

_AMOUNT_RE = re.compile(r"(?<![\d.,])(\d{1,3}(?:[\s  ]\d{3})+|\d+)(?:[.,]\d{1,2})?\s*(?:руб|₽)", re.I)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").replace(" ", " ").replace("ё", "е").lower()).strip()


def _amount(raw: str) -> int:
    return int(re.sub(r"\D", "", raw))


@dataclass(frozen=True)
class PricePair:
    amount: int
    service_id: int | None
    name: str | None
    status: str  # matched | unattributed | unsourced


class PriceCatalog:
    """Прайс региона из среза мира + множество цен врачей."""

    def __init__(self, rows: list[dict], doctor_costs: set[int] | None = None) -> None:
        self.by_cost: dict[int, list[tuple[str, int]]] = {}
        self.names: dict[int, str] = {}
        for row in rows:
            try:
                cost = int(round(float(row.get("cost"))))
                sid = int(row.get("serviceId"))
            except (TypeError, ValueError):
                continue
            name = _norm(row.get("serviceName"))
            if not name:
                continue
            self.by_cost.setdefault(cost, []).append((name, sid))
            self.names[sid] = str(row.get("serviceName"))
        for cands in self.by_cost.values():
            cands.sort(key=lambda c: -len(c[0]))
        self.doctor_costs = doctor_costs or set()

    @classmethod
    def from_world(cls, apidata: Path, region: int = 3) -> "PriceCatalog":
        rows = _read_jsonl(_latest(apidata / "price_by_region", f"price_region_{region}_*.jsonl"))
        doctor_costs: set[int] = set()
        doctor_file = _latest(apidata / "doctor_prices", "doctor_prices_*.jsonl")
        if doctor_file is not None:
            for row in _read_jsonl(doctor_file):
                for key in ("cost", "price"):
                    try:
                        doctor_costs.add(int(round(float(row.get(key)))))
                    except (TypeError, ValueError):
                        pass
        return cls(rows, doctor_costs)

    def pairs(self, text: str) -> list[PricePair]:
        out: list[PricePair] = []
        for line in str(text or "").splitlines():
            norm_line = _norm(line)
            for match in _AMOUNT_RE.finditer(line):
                amount = _amount(match.group(1))
                found = next(((name, sid) for name, sid in self.by_cost.get(amount, ()) if name in norm_line), None)
                if found is not None:
                    out.append(PricePair(amount, found[1], self.names.get(found[1]), "matched"))
                elif amount in self.by_cost or amount in self.doctor_costs:
                    out.append(PricePair(amount, None, None, "unattributed"))
                else:
                    out.append(PricePair(amount, None, None, "unsourced"))
        return out


def _latest(folder: Path, pattern: str) -> Path | None:
    files = sorted(p for p in folder.glob(pattern) if p.stat().st_size > 0)
    return files[-1] if files else None


def _read_jsonl(path: Path | None) -> list[dict]:
    if path is None:
        return []
    rows = []
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def price_verdict(pairs: list[PricePair], label: dict | None) -> str | None:
    """Исход ценового вопроса по разметке (`labels/price_allowed.jsonl`).

    Разметка: `kind` (single | family | multi | absent | not_price | context_dependent),
    `core` и `ok` — допустимые serviceId, `ambiguous` — одной строки не выбрать.

    - `correct` — одна цена, допустимая, вопрос однозначный;
    - `single_for_ambiguous` — одна допустимая цена там, где правильнее список (решение
      22.07: при неоднозначности — список). Не чужая, но и не верная: считается отдельно;
    - `foreign` — одна цена вне допустимых, или любая цена услуги, которой нет в прайсе
      (главная метрика: чужая цена);
    - `list_hit` — несколько цен, есть допустимая и нет чужих сверх половины;
    - `list_miss` — несколько цен, допустимых нет (считается как чужая);
    - `no_price` — цен в ответе нет;
    - None — разметки нет или вопрос не о цене / без контекста не понять.
    """

    if not label or label.get("kind") in ("not_price", "context_dependent"):
        return None
    allowed = {int(x) for x in (label.get("core") or [])} | {int(x) for x in (label.get("ok") or [])}
    shown = {p.service_id for p in pairs if p.status == "matched"}
    if not shown:
        return "no_price"
    if label.get("kind") == "absent" or not allowed:
        return "foreign"
    if len(shown) == 1:
        if not shown <= allowed:
            return "foreign"
        return "single_for_ambiguous" if label.get("ambiguous") else "correct"
    return "list_hit" if shown & allowed else "list_miss"
