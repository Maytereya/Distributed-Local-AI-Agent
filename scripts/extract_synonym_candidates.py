"""Кандидаты в `serviceSynonyms`: что пациенты называют, а каталог не находит.

Зачем: клиника заполняет поле `serviceSynonyms` в МИС вручную и просит список
«самых частых синонимов исходя из запросов пациентов». Придумывать синонимы за
клинику нельзя (мазок 290-510 руб. и посев 1120 руб. — разные исследования), но
мы можем показать, ЧТО пациенты пишут и на чём поиск спотыкается. Решение
«какой услуге это соответствует» остаётся за клиникой.

Как работает:
    1. разбирает лог бот-воркера агрегатора (тот же формат, что у
       `analyze_daily_dialogs.py`, парсер переиспользован);
    2. берёт ТОЛЬКО реплики пациентов;
    3. вытаскивает фрагменты, похожие на название услуги (по речевым шаблонам
       «сдать …», «сколько стоит …», «анализ на …», плюс короткие реплики,
       которые целиком выглядят как название);
    4. прогоняет каждый фрагмент через боевой резолвер каталога;
    5. ранжирует по частоте то, что НЕ нашлось, и пишет рабочий CSV.

Запуск:
    docker logs --since 720h support-messenger-aggregator-telegram_bot-1 > day.log
    ./venv/bin/python scripts/extract_synonym_candidates.py day.log -o candidates.csv

ПДн: лог содержит имена и телефоны пациентов. Скрипт отбрасывает фрагменты,
похожие на ФИО, телефон, дату и адрес, но выходной файл всё равно смотреть
глазами перед отправкой в клинику. Логи в репозитории не хранить.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_logic_2.nayka_api import api_price  # noqa: E402
from messengers_router.services._prices_helpers import (  # noqa: E402
    SAMARA_PRICE_REGION_ID,
    resolve_price_service_name_from_catalog,
)
from messengers_router.city import match_city  # noqa: E402
from scripts.analyze_daily_dialogs import parse_log  # noqa: E402

# Речевые шаблоны, после которых пациент называет услугу.
_TRIGGER_RE = re.compile(
    r"\b(?:сдать|сдаю|сдам|сдача|сдавать|"
    r"анализы?\s+на|кровь\s+на|моч[уа]\s+на|мазок\s+на|"
    r"сколько\s+стоит|стоимость|цена|цену|"
    r"записаться\s+на|сделать|пройти|нужен|нужна|нужно)\b",
    re.I,
)
# Служебные слова: сами по себе услугой не являются.
_STOP_RE = re.compile(
    r"^(?:и|в|во|на|по|с|со|за|у|к|о|об|от|до|из|же|ли|бы|это|мне|мы|вы|я|там|тут|"
    r"как|что|где|когда|вам|нам|его|её|ее|их|то|так|уже|ещё|еще|был|была|быть|есть|"
    r"нет|да|можно|нужно|хочу|пожалуйста|здравствуйте|добрый|доброе|день|утро|вечер|"
    r"спасибо|благодарю|хорошо|отлично|понятно|поняла|понял|ясно|ага|ок|окей|давайте|"
    r"завтра|сегодня|вчера|утром|вечером|сейчас|именно|очень|просто|ещё|тоже|"
    r"скажите|подскажите|уточнить|узнать|записаться|запишите|запись|приём|прием|"
    r"анализ|анализы|анализа|анализов|услуга|услуги|услуг|сдать|сдача|сделать)$",
    re.I,
)
# Разговорная реплика целиком — не название услуги.
_CHATTER_RE = re.compile(
    r"^(?:хорошо|спасибо|благодарю|поняла|понял|понятно|ясно|отлично|да|нет|ага|ок|"
    r"окей|давайте|здравствуйте|добрый день|доброе утро|добрый вечер|до свидания)\b",
    re.I,
)
# ПДн и мусор: телефон, дата, время, длинные числа, ФИО-подобное.
_PII_RE = re.compile(
    r"\d{2}[.:/]\d{2}|\+?\d{7,}|\b\d{4}\b|"
    r"\b[А-ЯЁ][а-яё]+(?:ов|ев|ин|ына|ова|ева|ская|ский)\b"
)
_MAX_WORDS = 4
_MIN_LEN = 4


def _looks_like_pii(phrase: str) -> bool:
    return bool(_PII_RE.search(phrase))


def _unquote_csv(line: str) -> str:
    """Снимает кавычки и удвоение из однополевой CSV-строки, отданной COPY."""

    value = line.strip()
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        value = value[1:-1].replace('""', '"')
    return value.replace("\\n", " ").strip()


def extract_candidates(texts: list[str]) -> Counter:
    """Собирает частотник фрагментов, похожих на название услуги.

    :param texts: реплики пациентов
    :return: счётчик «фрагмент → сколько раз встретился»
    """

    found: Counter = Counter()
    for raw in texts:
        text = str(raw or "").strip()
        if not text or _looks_like_pii(text):
            continue
        spans: list[str] = []
        for match in _TRIGGER_RE.finditer(text):
            spans.append(re.split(r"[.,;!?()\n]|\bи\b", text[match.end() :])[0])
        # Короткая реплика без триггера часто и есть голое название услуги —
        # но только если это не вежливая обвязка и не название города.
        if (
            not spans
            and len(text.split()) <= _MAX_WORDS
            and not _CHATTER_RE.match(text)
            and not match_city(text)
        ):
            spans.append(text)
        for span in spans:
            tokens = [
                tok
                for tok in re.findall(r"[а-яёa-z0-9]+", span.lower())
                if not _STOP_RE.match(tok)
            ]
            # Только самый длинный осмысленный фрагмент плюс его первое слово:
            # перебор ВСЕХ префиксов плодит мусор («сколько будет стоить» →
            # «будет»), а одно слово нужно для случаев «холестерин натощак».
            if not tokens:
                continue
            variants = {" ".join(tokens[:_MAX_WORDS])}
            if len(tokens[0]) >= 5:
                variants.add(tokens[0])
            for phrase in variants:
                if len(phrase) >= _MIN_LEN and not match_city(phrase):
                    found[phrase] += 1
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logfile", help="лог бот-воркера агрегатора либо CSV/текст с репликами")
    parser.add_argument(
        "--format",
        choices=("log", "texts"),
        default="log",
        help=(
            "log — вывод `docker logs` бот-воркера (по умолчанию); "
            "texts — по одной реплике пациента в строке, как отдаёт выгрузка из "
            "базы агрегатора: COPY (SELECT text FROM core_message WHERE sender='user')"
        ),
    )
    parser.add_argument("-o", "--out", default="synonym_candidates.csv")
    parser.add_argument("--min-count", type=int, default=2, help="порог частоты")
    args = parser.parse_args()

    lines = Path(args.logfile).read_text(encoding="utf-8", errors="ignore").splitlines()
    if args.format == "texts":
        # Выгрузка из БД агрегатора: одна реплика в строке. Источник лучше логов —
        # docker logs ротируются и хранят дни, а в `core_message` вся история.
        texts = [_unquote_csv(line) for line in lines if line.strip()]
        print(f"реплик пациентов: {len(texts)}")
    else:
        dialogs = parse_log(lines)
        texts = [
            text
            for dialog in dialogs.values()
            for who, text in dialog.events
            if who == "user"
        ]
        print(f"диалогов: {len(dialogs)}, реплик пациентов: {len(texts)}")

    candidates = extract_candidates(texts)
    frequent = [(p, c) for p, c in candidates.most_common() if c >= args.min_count]
    print(f"фрагментов с частотой >= {args.min_count}: {len(frequent)}")

    rows = [r for r in api_price.load_price_by_region(SAMARA_PRICE_REGION_ID) if isinstance(r, dict)]
    misses: list[dict[str, object]] = []
    for phrase, count in frequent:
        if resolve_price_service_name_from_catalog(phrase, rows=rows):
            continue
        misses.append(
            {
                "сколько_раз_спросили": count,
                "как_написал_пациент": phrase,
                "какая_это_услуга": "",
                "синонимы_вписать_в_МИС": "",
            }
        )

    out = Path(args.out)
    with out.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "сколько_раз_спросили",
                "как_написал_пациент",
                "какая_это_услуга",
                "синонимы_вписать_в_МИС",
            ],
        )
        writer.writeheader()
        writer.writerows(misses)
    print(f"не находится каталогом: {len(misses)} → {out}")


if __name__ == "__main__":
    main()
