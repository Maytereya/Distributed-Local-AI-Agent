"""Где именно теряется корзина мульти-расчёта: атрибуция промахов по слоям.

Зачем: «корзина не работает» — не диагноз. Разбор 16.09 показал, что реплика из
нескольких анализов гибнет в ЧЕТЫРЁХ разных местах, и починка одного слоя не
меняет того, что видит пациент. Скрипт отвечает на вопрос «сколько промахов даёт
каждый слой», чтобы чинить в порядке ущерба, а не в порядке обнаружения.

Слои (первый, на котором позиция потеряна):

    SPLIT   фрагмент выброшен сплиттером МОЛЧА, до резолва. «т3» — буква с
            цифрой, а `_MULTI_PRICE_SERVICE_HINT_RE` требует три подряд буквы.
            Пациент не получает даже «не распознал».
    RESOLVE фрагмент дошёл до резолвера и честно не распознан.
    GUARD   корзина распознана полностью, но `_match_drops_unsatisfiable_qualifier`
            вернул None (BUG-2026-09-07-GUARD-OVER-BLOCK).
    ROUTE   только с `--prod`: грундер схлопнул список в один `service_name`,
            либо планировщик увёл в `service_bundle_info`, где корзины нет.
    OK      корзина выдана.

Почему корпус порождается, а не выдуман: свип по 48 корзинам из ДОСЛОВНЫХ
названий каталога дал 0 гашений (журнал багов, BUG-2026-09-07-GUARD-OVER-BLOCK).
Гард бьёт по пациентскому наименованию, поэтому корпус собирается из пациентской
лексики — синонимов МИС, коротких обозначений и реальных формулировок из
переписки. Придумывать синонимы услуг за клинику нельзя (CLAUDE.md).

Запуск:
    # без VPN, на живом каталоге и словаре МИС
    ./venv/bin/python scripts/audit_cart_misses.py

    # по выгрузке реплик из БД агрегатора (одна реплика в строке)
    ./venv/bin/python scripts/audit_cart_misses.py --texts replies.txt

    # по логу бот-воркера
    docker logs --since 720h support-messenger-aggregator-telegram_bot-1 > day.log
    ./venv/bin/python scripts/audit_cart_misses.py --log day.log

    # с атрибуцией маршрута (прод обслуживает живых пациентов — осознанно!)
    ./venv/bin/python scripts/audit_cart_misses.py --prod http://ХОСТ/api/messenger-generate-once --max-probes 20

ПДн: лог и выгрузка содержат имена и телефоны пациентов. В репозитории их не
хранить; выходной CSV смотреть глазами перед отправкой кому-либо.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_logic_2.nayka_api import api_price  # noqa: E402
from messengers_router.services import _biomaterial  # noqa: E402

# Диагностика слоёв по определению лезет внутрь конвейера: публичного API,
# который сказал бы «на каком шаге потеряна позиция», не существует.
from messengers_router.services._prices_helpers import (  # noqa: E402
    _PRICE_SHORT_TOKEN_WHITELIST,
    SAMARA_PRICE_REGION_ID,
    _build_multi_price_payload,
    _catalog_token_vocabulary,
    _distinctive_service_tokens,
    _normalise_input,
    _resolve_multi_price_items,
    _split_price_query_items,
    _token_covered_by,
    resolve_price_service_name_from_catalog,
)
from agent_logic_2.nayka_api import api_service_info  # noqa: E402
from scripts.analyze_daily_dialogs import parse_log  # noqa: E402

# Корзина — это ДВЕ и более позиции. Одна позиция мульти-расчётом не является.
_MIN_CART_ITEMS = 2


def pin_latest_service_info() -> str:
    """Прикрепляет словарь МИС к САМОМУ СВЕЖЕМУ срезу, а не к «сегодняшнему».

    Зачем: срез датирован по Самаре (UTC+4). После самарской полуночи файла «на
    сегодня» ещё нет, `load_service_info` уходит скачивать его у клиники, без
    VPN получает отказ — и `_vocabularies()` по своей конструкции возвращает
    ПУСТОЙ словарь, молча. Свип в этот момент меряет бота без 872 синонимов и
    несравним с прогоном часом раньше: 16.09 это превратило 74% промахов гарда
    в 82% «не распознано» и выглядело как успех починки.

    Диагностика обязана быть воспроизводимой в любой час, поэтому берём
    новейший СУЩЕСТВУЮЩИЙ срез и говорим вслух, какой именно.

    :return: строка с описанием использованного среза
    """

    today = api_service_info.service_info_path()
    if today.exists():
        return f"{today.name} (срез на сегодня)"
    snapshots = sorted(today.parent.glob("service_info_*.jsonl"))
    if not snapshots:
        return "срезов МИС нет — словарь синонимов будет пуст"
    latest = snapshots[-1]
    api_service_info.service_info_path = lambda *a, **k: latest  # type: ignore[assignment]
    _biomaterial._VOCAB_CACHE.clear()
    return f"{latest.name} (на сегодня среза нет, взят последний)"

# Известные живые промахи: две формулировки из хэндоффа 16.09 и то, что
# воспроизведено на проде 16.09. Держим отдельно от порождённого корпуса —
# это не выборка, а зафиксированные жалобы.
KNOWN_LIVE_CARTS = (
    "ферритин, глюкоза, холестерин",
    "Кровь: АЛТ, АСТ, ГГТП, цистатин С",
    "Общий анализ крови с лейкоформулой, временем свертывания, тромбоциты и глюкоза",
    "ОАК ОАМ глюкоза ферритин",
    "т3, т4, ттг",
    "ТТГ, Т4 свободный, ферритин",
)

# Реплики, на которых гард ОБЯЗАН срабатывать: уточнение, которого клиника не
# оказывает. Свип без них показал бы только одну сторону (CLAUDE.md: целевой
# тест гарда молчит о том, что тот глушит лишнее).
ANTI_UNDER_BLOCK = (
    "приём терапевта по ОМС",
    "онлайн консультация",
    "приём кардиолога удалённо",
)


def _cart_like(text: str) -> bool:
    """Похожа ли реплика на корзину: есть разделитель перечисления.

    :param text: реплика пациента
    :return: True, если реплику имеет смысл разбирать как мульти-расчёт
    """

    lowered = str(text or "").lower()
    if len(lowered) > 300:
        return False
    return any(sep in lowered for sep in (",", ";", " и ", "+", "/", "\n"))


# Обрывки химических формул, которыми `serviceSynonyms` засорён из-за запятых
# ВНУТРИ названий: «11», «13-диметил-7-(1», «4-α-d-glucanohydralase». Пациент
# такого не пишет, и корзины из этого мусора завышают долю гарда. Сам засор —
# отдельный дефект, см. BUG-2026-09-16-NUMERIC-SYNONYM-SHARD.
_FORMULA_CHARS = frozenset("()[]<>{}αβγ→>≥≤")


def _patient_plausible(term: str) -> bool:
    """Может ли пациент это НАПИСАТЬ.

    Фильтр нужен не для красоты: корпус из обрывков формул мерил бы поведение
    на входах, которых в жизни нет.

    :param term: кандидат в пациентское обозначение
    :return: True, если обозначение похоже на человеческий ввод
    """

    value = str(term or "").strip()
    if len(value) < 2 or len(value) > 60:
        return False
    if _FORMULA_CHARS & set(value):
        return False
    if not any(ch.isalpha() for ch in value):
        return False
    # Обрывок формулы почти всегда начинается с цифры и несёт дефис: «25-hydroxy»,
    # «13-диметил», «2пропса». Настоящие пациентские коды («б12», «оак») — нет.
    if value[0].isdigit() and "-" in value:
        return False
    return True


def patient_vocabulary(rows: list[dict[str, Any]]) -> list[str]:
    """Пациентские названия услуг из живых данных.

    Источники — только данные, ничего не придумано: однозначные синонимы МИС
    (клиника сама сопоставила слово услуге), короткие обозначения из белого
    списка резолвера и реальные формулировки пациентов из CSV по переписке.

    :param rows: строки прайса региона
    :return: список пациентских обозначений, каждое резолвится по отдельности
    """

    vocab = _biomaterial._vocabularies()
    terms: list[str] = []
    seen: set[str] = set()

    def _add(term: str) -> None:
        key = _normalise_input(term)
        if not key or key in seen or not _patient_plausible(term):
            return
        seen.add(key)
        terms.append(term)

    # Синонимы МИС: пациентское слово, которое клиника САМА сопоставила услуге.
    for synonym in sorted(vocab.synonym_to_service):
        _add(synonym)
    # Короткие обозначения, которые single-путь уже умеет.
    for short in sorted(_PRICE_SHORT_TOKEN_WHITELIST):
        _add(short)
    # Как пациенты писали на самом деле — из разбора семи месяцев переписки.
    for phrase in _csv_patient_phrases():
        _add(phrase)

    # Оставляем только то, что резолвится по отдельности: корзина из позиций,
    # которые и поодиночке не находятся, ничего не скажет про потерю в списке.
    resolvable = [t for t in terms if resolve_price_service_name_from_catalog(t, rows=rows)]
    return resolvable


def _csv_patient_phrases() -> list[str]:
    """Читает колонку «как_написал_пациент» из рабочих CSV для клиники.

    :return: реальные формулировки пациентов; пустой список, если файлов нет
    """

    docs = Path(__file__).resolve().parent.parent / "docs"
    phrases: list[str] = []
    for path in sorted(docs.glob("synonyms_to_fill*.csv")):
        try:
            with path.open(encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    value = str(row.get("как_написал_пациент") or "").strip()
                    if value:
                        phrases.append(value)
        except OSError:
            continue
    return phrases


def build_corpus(rows: list[dict[str, Any]], *, size: int, seed: int) -> list[str]:
    """Собирает корзины из пациентской лексики детерминированно.

    Детерминированность обязательна: свип должен воспроизводиться между
    прогонами, иначе цифры не сравнить.

    :param rows: строки прайса региона
    :param size: сколько порождённых корзин вернуть
    :param seed: сдвиг выборки; одинаковый seed даёт одинаковый корпус
    :return: список реплик-корзин
    """

    terms = patient_vocabulary(rows)
    if len(terms) < _MIN_CART_ITEMS:
        return []

    # Шаг взаимно прост с длиной словаря, поэтому выборка обходит ВЕСЬ словарь,
    # а не его алфавитное начало. Без этого корпус состоял бы из одних латинских
    # синонимов на «a», и свип мерил бы не тот язык, на котором пишут пациенты.
    stride = _coprime_stride(len(terms))
    carts: list[str] = []
    cursor = seed % len(terms)
    for i in range(size):
        width = _MIN_CART_ITEMS + (i % 2)
        picked = [terms[(cursor + k * stride) % len(terms)] for k in range(width)]
        carts.append(", ".join(picked))
        cursor = (cursor + width * stride) % len(terms)
    return carts


def _coprime_stride(length: int) -> int:
    """Шаг обхода, взаимно простой с длиной, около золотого сечения.

    :param length: размер словаря
    :return: шаг, при котором обход покрывает весь словарь
    """

    from math import gcd

    candidate = max(1, int(length * 0.618))
    while candidate > 1 and gcd(candidate, length) != 1:
        candidate -= 1
    return candidate or 1


def guard_verdict(
    query_text: str,
    offered_names: str,
    rows: list[dict[str, Any]],
) -> list[tuple[str, bool, bool]]:
    """Повторяет решение гарда потокенно, чтобы было видно, что именно его уронило.

    :param query_text: исходная реплика
    :param offered_names: названия услуг, которые корзина собиралась предложить
    :param rows: строки прайса региона
    :return: список (токен, покрыт_предложенным, есть_в_каталоге)
    """

    import re

    canon_tokens = set(re.findall(r"[a-zа-яё0-9]+", _normalise_input(offered_names)))
    vocab = _catalog_token_vocabulary(rows)
    out: list[tuple[str, bool, bool]] = []
    for token in _distinctive_service_tokens(query_text):
        out.append((token, _token_covered_by(token, canon_tokens), token in vocab))
    return out


def attribute(query_text: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Определяет ПЕРВЫЙ слой, на котором корзина потеряна.

    :param query_text: реплика пациента
    :param rows: строки прайса региона
    :return: запись с вердиктом и подробностями по слоям
    """

    fragments = _split_price_query_items(query_text)
    # Что сплиттер выбросил молча: делим реплику сами по тем же разделителям и
    # смотрим, какие куски не дожили до списка фрагментов.
    import re

    raw_chunks = [c.strip(" ?!.,;:-–—") for c in re.split(r"[,;+/\n]| и ", query_text)]
    raw_chunks = [c for c in raw_chunks if c]
    kept = {_normalise_input(f) for f in fragments}
    dropped = [c for c in raw_chunks if _normalise_input(c) and _normalise_input(c) not in kept]
    # Выброшенное считаем потерей, только если оно резолвится само по себе —
    # иначе это шум списка («Сколько будет стоить?»), и молчание тут правильно.
    dropped_resolvable = [c for c in dropped if resolve_price_service_name_from_catalog(c, rows=rows)]

    record: dict[str, Any] = {
        "реплика": query_text,
        "фрагментов": len(fragments),
        "выброшено_молча": "; ".join(dropped_resolvable),
        "распознано": "",
        "не_распознано": "",
        "гард_уронил": "",
        "слой": "",
    }

    # Потеря на сплиттере видна до разбора корзины и стоит доли миллисекунды,
    # тогда как сборка корзины на живом каталоге — единицы секунд. Дешёвая
    # проверка идёт первой.
    if dropped_resolvable:
        record["слой"] = "SPLIT"
        return record

    # Вердикт берём у боевой функции, а не повторяем её условия: харнесс,
    # разошедшийся с продом, врёт убедительнее, чем отсутствие харнесса.
    payload = _build_multi_price_payload(query_text, rows)
    if payload is not None:
        record["слой"] = "OK"
        record["распознано"] = str(payload.get("service_name") or "")
        return record

    # Подробности нужны только когда корзина не вышла: здесь второй разбор
    # оправдан, в ветке OK он был бы чистой потерей секунд.
    items, unrecognized = _resolve_multi_price_items(query_text, rows)
    record["распознано"] = "; ".join(str(i.get("service_name") or "") for i in items)
    record["не_распознано"] = "; ".join(str(u) for u in unrecognized)
    if len(items) >= _MIN_CART_ITEMS:
        # Корзина собралась, payload пуст ⇒ это гард.
        offered = " ".join(
            [str(i.get("service_name") or "") for i in items] + [str(u) for u in unrecognized]
        )
        killers = [tok for tok, covered, known in guard_verdict(query_text, offered, rows) if not covered and not known]
        record["гард_уронил"] = "; ".join(killers)
        record["слой"] = "GUARD"
        return record
    record["слой"] = "RESOLVE"
    return record


def probe_prod(url: str, text: str, timeout: int) -> dict[str, Any]:
    """Спрашивает прод с `debug=True` и достаёт решение и план.

    :param url: эндпоинт messenger-generate-once
    :param text: реплика
    :param timeout: таймаут ответа в секундах
    :return: словарь с label, entities и выбранным инструментом
    """

    payload = {"session_id": f"cart_audit_{int(time.time() * 1000)}", "text": text, "debug": True}
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    debug = (data.get("state_update") or {}).get("debug") or {}
    decision = debug.get("decision") or {}
    steps = (debug.get("plan") or {}).get("steps") or []
    return {
        "label": decision.get("label") or "",
        "entities": decision.get("entities") or {},
        "tool": (steps[0].get("tool") if steps else "") or "",
        "answer": str(data.get("text") or ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--log", help="лог бот-воркера агрегатора")
    source.add_argument("--texts", help="выгрузка реплик из БД агрегатора, одна в строке")
    source.add_argument("--corpus", help="файл с репликами, по одной в строке")
    parser.add_argument("-o", "--out", default="cart_misses.csv")
    # Разбор одной корзины на живом каталоге — единицы секунд, поэтому по
    # умолчанию берём объём, укладывающийся в несколько минут. Больше — через
    # --size, осознанно.
    parser.add_argument("--size", type=int, default=40, help="размер порождённого корпуса")
    parser.add_argument("--seed", type=int, default=0, help="сдвиг выборки порождённого корпуса")
    # Порождённый корпус зависит от длины словаря МИС, а клиника дополняет его
    # ежедневно: словарь вырос на ОДИН термин — и выборка сместилась целиком.
    # Поэтому «до/после» сравнивают, только закрепив корпус: выгрузить здесь,
    # затем подать обоим прогонам через --corpus.
    parser.add_argument("--dump-corpus", help="записать использованный корпус в файл")
    parser.add_argument("--prod", help="эндпоинт для атрибуции маршрута (ВЫКЛ по умолчанию)")
    parser.add_argument("--max-probes", type=int, default=10, help="предел запросов к проду")
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    print(f"срез МИС: {pin_latest_service_info()}")
    synonyms = len(_biomaterial._vocabularies().synonym_to_service)
    print(f"синонимов МИС в словаре: {synonyms}")
    if not synonyms:
        # Пустой словарь — не «ноль промахов», а другой бот. Молчать нельзя:
        # прогон без синонимов несравним с прогоном с ними.
        print("  ВНИМАНИЕ: словарь синонимов ПУСТ — цифры несравнимы с прогоном, где он есть")

    rows = [r for r in api_price.load_price_by_region(SAMARA_PRICE_REGION_ID) if isinstance(r, dict)]
    print(f"прайс региона {SAMARA_PRICE_REGION_ID}: {len(rows)} строк")

    if args.log:
        lines = Path(args.log).read_text(encoding="utf-8", errors="ignore").splitlines()
        dialogs = parse_log(lines)
        texts = [t for d in dialogs.values() for who, t in d.events if who == "user"]
        print(f"диалогов: {len(dialogs)}, реплик пациентов: {len(texts)}")
        queries = [t for t in texts if _cart_like(t)]
        print(f"похожих на корзину: {len(queries)}")
    elif args.texts or args.corpus:
        path = Path(args.texts or args.corpus)
        texts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        queries = [t for t in texts if _cart_like(t)] if args.texts else texts
        print(f"реплик: {len(texts)}, к разбору: {len(queries)}")
    else:
        queries = list(KNOWN_LIVE_CARTS) + build_corpus(rows, size=args.size, seed=args.seed)
        print(f"корпус из живых данных: {len(queries)} корзин (из них известных живых: {len(KNOWN_LIVE_CARTS)})")

    if args.dump_corpus:
        Path(args.dump_corpus).write_text("\n".join(queries) + "\n", encoding="utf-8")
        print(f"корпус выгружен: {args.dump_corpus}")

    # Сборка корзины на живом каталоге стоит единицы секунд на реплику, поэтому
    # прогресс печатаем с принудительным сбросом: без него вывод буферизуется и
    # свип десять минут выглядит зависшим.
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for number, query in enumerate(queries, 1):
        records.append(attribute(query, rows))
        if number % 10 == 0 or number == len(queries):
            elapsed = time.perf_counter() - started
            print(f"  разобрано {number}/{len(queries)}  ({elapsed:.0f}с)", flush=True)

    if args.prod:
        print(f"\nатрибуция маршрута по проду, не более {args.max_probes} запросов")
        probed = 0
        # OK-строки зондируем ТОЖЕ. «Офлайн корзина собирается» и «пациент получил
        # ответ» — разные утверждения: «Кровь: АЛТ, АСТ, ГГТП, цистатин С» офлайн
        # даёт payload, а прод уводит реплику в `test_assist` и отвечает «АЛТ не
        # найден в базе». Свип, пропускающий OK, эту потерю не увидит.
        for record in records:
            if probed >= args.max_probes:
                break
            try:
                live = probe_prod(args.prod, str(record["реплика"]), args.timeout)
            except Exception as exc:  # noqa: BLE001 — диагностика, сбой зонда не должен ронять свип
                record["прод"] = f"ошибка: {exc}"
                probed += 1
                continue
            probed += 1
            entities = live["entities"]
            names = [v for k, v in entities.items() if k in ("service_name", "test_name") and v]
            record["прод_label"] = live["label"]
            record["прод_инструмент"] = live["tool"]
            record["прод_сущности"] = "; ".join(str(n) for n in names)
            # Маршрут признаётся виновным раньше гарда: если офлайн корзина
            # собиралась, а прод увёл в одноуслуговый инструмент — пациент до
            # корзины не доходит независимо от гарда.
            if live["tool"] == "service_bundle_info":
                record["слой"] = "ROUTE"
            elif record["слой"] == "OK" and live["tool"] != "price_info":
                # Корзина собралась бы, но реплику ведут мимо неё другим путём.
                record["слой"] = "ROUTE"
        print(f"зондов отправлено: {probed}")

    print("\n=== анти-under-block: гард обязан сработать ===")
    for text in ANTI_UNDER_BLOCK:
        payload = _build_multi_price_payload(text, rows)
        verdict = "гасит ✓" if payload is None else "ПРОПУСТИЛ ✗"
        print(f"  {text!r:36} {verdict}")

    counts = Counter(str(r["слой"]) for r in records)
    print("\n=== первый теряющий слой ===")
    total = len(records) or 1
    for layer in ("SPLIT", "RESOLVE", "GUARD", "ROUTE", "OK"):
        n = counts.get(layer, 0)
        print(f"  {layer:8} {n:4}  {100 * n / total:5.1f}%")

    fieldnames = [
        "слой",
        "реплика",
        "фрагментов",
        "выброшено_молча",
        "распознано",
        "не_распознано",
        "гард_уронил",
        "прод_label",
        "прод_инструмент",
        "прод_сущности",
        "прод",
    ]
    out = Path(args.out)
    with out.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for record in sorted(records, key=lambda r: str(r["слой"])):
            writer.writerow(record)
    print(f"\nподробности: {out}")


if __name__ == "__main__":
    main()
