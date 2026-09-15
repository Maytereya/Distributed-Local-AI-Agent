"""Конвейер синонимов услуг из МИС (`serviceSynonyms`) — от выгрузки до резолвера.

Механика подключена коммитом `203c025` (слой 1 в
`resolve_price_service_name_from_catalog`, высшее доверие). Живое поле клиника
ещё не заполнила, поэтому поведение проверяется на подставленном справочнике —
но проверяется ВЕСЬ путь, а не только словарь: заполненный синоним обязан
пережить загрузчик, разбор значения и склейку с прайсом.

Мотив (проверено на живом API 09.09): `serviceInfoAll` отдаёт 8472 услуги, до
бота доходит 1622 — остальные отбрасывает фильтр `_has_useful_text`, который
смотрит только на `description`/`indication`/`preparation`. Услуга, у которой
клиника заполнит ТОЛЬКО синонимы, до бота не дойдёт вовсе.
"""

from __future__ import annotations

from agent_logic_2.nayka_api import api_price, api_service_info
from messengers_router.services import _biomaterial as bio
from messengers_router.services._prices_helpers import SAMARA_PRICE_REGION_ID

ROWS = [r for r in api_price.load_price_by_region(SAMARA_PRICE_REGION_ID) if isinstance(r, dict)]


def _row(**over) -> dict:
    base = {
        "serviceId": 900001,
        "serviceName": "Липидограмма",
        "serviceSynonyms": None,
        "biomatNames": None,
        "description": "",
        "indication": "",
        "preparation": "",
    }
    base.update(over)
    return base


# --- Загрузчик: синоним — это полезное содержимое --------------------------

def test_loader_keeps_service_whose_only_content_is_synonyms():
    """Услуга, у которой заполнены ТОЛЬКО синонимы, обязана дойти до бота.

    Класс `mis_field_dropped_by_loader`: фильтр выгрузки решает, что запись
    «бесполезна», по списку полей, в котором синонимов нет. Клиника заполняет
    поле — работа молча теряется, и это выглядит как «бот не понимает синонимы».
    """
    row = _row(serviceSynonyms="холестерин, липидный профиль")
    assert api_service_info._has_useful_text(row) is True


def test_loader_still_drops_row_without_any_useful_field():
    """Анти-регресс: пустая запись по-прежнему отбрасывается."""
    assert api_service_info._has_useful_text(_row()) is False


# --- Разбор значения: формат задаёт клиника, не мы -------------------------

def test_vocabulary_parses_separators_case_and_spacing():
    """Синонимы приходят как текст, набранный руками: запятые, точки с запятой,
    произвольный регистр и лишние пробелы обязаны работать одинаково."""
    vocab = bio._build_vocabularies([
        _row(serviceName="Липидограмма", serviceSynonyms="  Холестерин ; липидный профиль,КРОВЬ НА ХОЛЕСТЕРИН "),
    ])
    for probe in ("холестерин", "Холестерин", "  ЛИПИДНЫЙ ПРОФИЛЬ  ", "кровь на холестерин"):
        assert vocab.synonym_to_service.get(bio._normalise_input(probe)) == "Липидограмма", probe


def test_vocabulary_accepts_list_value():
    """МИС может отдать поле списком, а не строкой — оба формата равноправны."""
    vocab = bio._build_vocabularies([
        _row(serviceName="Общий анализ мочи", serviceSynonyms=["ОАМ", "общий мочи"]),
    ])
    assert vocab.synonym_to_service.get("оам") == "Общий анализ мочи"
    assert vocab.synonym_to_service.get("общий мочи") == "Общий анализ мочи"


def test_empty_and_whitespace_synonyms_do_not_create_entries():
    """Пустые фрагменты («а,,б») не должны порождать пустой ключ-ловушку."""
    vocab = bio._build_vocabularies([
        _row(serviceName="Глюкоза", serviceSynonyms="сахар,, ; ,кровь на сахар"),
    ])
    assert "" not in vocab.synonym_to_service
    assert vocab.synonym_to_service.get("сахар") == "Глюкоза"


# --- Коллизии: 100+ строк заполняются руками -------------------------------

def test_same_synonym_on_two_services_is_not_resolved_silently():
    """Один синоним у нескольких услуг — не повод выбрать первую.

    Раньше здесь фиксировалось «побеждает первая по порядку выгрузки». Решение
    отменено данными: в живом словаре клиники неоднозначны 72 синонима из 843
    (9%), и среди них «рак» → 4 разных онкомаркера, «холестерин» → 4 анализа,
    «вэб» → 9 услуг. Молчаливый выбор одной из них — тот самый класс
    «дезинформация ценой». Однозначного ответа тут нет, и слой обязан это
    признать.
    """
    rows = [
        _row(serviceId=1, serviceName="Липидограмма", serviceSynonyms="холестерин"),
        _row(serviceId=2, serviceName="Холестерол - ЛПВП", serviceSynonyms="холестерин"),
    ]
    vocab = bio._build_vocabularies(rows)
    assert "холестерин" not in vocab.synonym_to_service
    assert bio.mis_synonym_service("холестерин", vocab=vocab) is None


def test_candidate_order_is_stable_across_rebuilds():
    """Порядок кандидатов не должен плавать между перезапусками — иначе развилка
    показывает пациенту варианты в разном порядке на один и тот же вопрос."""
    rows = [
        _row(serviceId=1, serviceName="Липидограмма", serviceSynonyms="холестерин"),
        _row(serviceId=2, serviceName="Холестерол - ЛПВП", serviceSynonyms="холестерин"),
    ]
    first = bio._build_vocabularies(rows).synonym_candidates["холестерин"]
    again = bio._build_vocabularies(rows).synonym_candidates["холестерин"]
    assert first == again == ("Липидограмма", "Холестерол - ЛПВП")


# --- Сквозной путь: словарь → резолвер → прайс -----------------------------

def test_synonym_from_mis_wins_over_lexical_match_end_to_end(monkeypatch):
    """Главный сценарий заказчика: пациентское слово приземляется на нужную услугу.

    Собираем словарь НАСТОЯЩИМ сборщиком из строк вида МИС (не подсовываем
    готовый dict), затем гоняем через боевой резолвер по живому прайсу.
    Без синонима «кровь на сахар» уходит на «Свекла сахарная (F227)» —
    аллергопанель (BUG-2026-09-07-PREFIX-COLLISION).
    """
    from messengers_router.services import _prices_helpers as ph

    assert ph.resolve_price_service_name_from_catalog("кровь на сахар", rows=ROWS) != "Глюкоза"

    vocab = bio._build_vocabularies([
        _row(serviceName="Глюкоза", serviceSynonyms="сахар, кровь на сахар, сахар крови"),
    ])
    monkeypatch.setattr(bio, "_vocabularies", lambda: vocab)
    monkeypatch.setattr(ph, "mis_synonym_service", bio.mis_synonym_service)

    assert ph.resolve_price_service_name_from_catalog("кровь на сахар", rows=ROWS) == "Глюкоза"


def test_synonym_pointing_outside_region_price_is_ignored(monkeypatch):
    """Синоним на услугу, которой нет в прайсе региона, не подставляется."""
    from messengers_router.services import _prices_helpers as ph

    vocab = bio._build_vocabularies([
        _row(serviceName="Услуги которой нет в прайсе", serviceSynonyms="выдуманный анализ"),
    ])
    monkeypatch.setattr(bio, "_vocabularies", lambda: vocab)
    monkeypatch.setattr(ph, "mis_synonym_service", bio.mis_synonym_service)

    assert ph.resolve_price_service_name_from_catalog("выдуманный анализ", rows=ROWS) is None


# --- Разделитель «||» из живых данных МИС ---------------------------------

def test_vocabulary_parses_pipe_separator():
    """В выгрузке клиники встречается «|| Хламидия трахоматис ||» наравне с запятыми."""
    vocab = bio._build_vocabularies([
        _row(serviceName="Chlamydia trachomatis IgG", serviceSynonyms="|| Хламидия трахоматис ||"),
    ])
    assert vocab.synonym_to_service.get("хламидия трахоматис") == "Chlamydia trachomatis IgG"


# --- Поиск синонима ВНУТРИ реплики ----------------------------------------

def test_synonym_found_inside_a_longer_patient_message():
    """Пациент пишет с обвязкой: «кровь с лейкоформулой сдать хочу».

    Сверка реплики ЦЕЛИКОМ с ключом словаря делает синонимы бесполезными —
    живые формулировки почти всегда несут вежливость и глаголы вокруг названия.
    """
    vocab = bio._build_vocabularies([
        _row(serviceName="Общий анализ крови (полный)", serviceSynonyms="оак, кровь с лейкоформулой"),
    ])
    for phrase in (
        "кровь с лейкоформулой сдать хочу",
        "хочу сдать кровь с лейкоформулой",
        "сколько стоит кровь с лейкоформулой?",
    ):
        assert bio.mis_synonym_service(phrase, vocab=vocab) == "Общий анализ крови (полный)", phrase


def test_longest_synonym_wins_when_several_match():
    """Из нескольких подходящих синонимов побеждает самый длинный — он точнее."""
    vocab = bio._build_vocabularies([
        _row(serviceId=1, serviceName="Общий анализ мочи", serviceSynonyms="моча"),
        _row(serviceId=2, serviceName="Глюкоза в моче (разовая порция)", serviceSynonyms="глюкоза в моче"),
    ])
    assert bio.mis_synonym_service("сдать глюкоза в моче", vocab=vocab) == "Глюкоза в моче (разовая порция)"


def test_synonym_does_not_match_across_word_boundary():
    """Синоним «оак» не должен находиться внутри «троакарная» или «психоактивные»."""
    vocab = bio._build_vocabularies([
        _row(serviceName="Общий анализ крови (полный)", serviceSynonyms="оак"),
    ])
    assert bio.mis_synonym_service("эпицистостомия троакарная", vocab=vocab) is None


# --- Неоднозначные синонимы: 9% живого словаря ----------------------------

def test_ambiguous_synonym_never_returns_a_single_service():
    """«вэб» в данных клиники указывает на 9 услуг, «рак» — на 4 разных онкомаркера.

    Отдать одну из них молча — это ровно класс «дезинформация ценой». Одиночный
    резолв обязан отказаться; выбор делает развилка уровнем выше.
    """
    vocab = bio._build_vocabularies([
        _row(serviceId=1, serviceName="CA 15 - 3 (молочная железа)", serviceSynonyms="рак"),
        _row(serviceId=2, serviceName="CA 19 - 9 (карцинома поджелудочной)", serviceSynonyms="рак"),
    ])
    assert bio.mis_synonym_service("рак", vocab=vocab) is None


def test_ambiguous_synonym_exposes_all_candidates():
    """Полный список кандидатов доступен вызывающему — из него строится развилка."""
    vocab = bio._build_vocabularies([
        _row(serviceId=1, serviceName="CA 15 - 3 (молочная железа)", serviceSynonyms="рак"),
        _row(serviceId=2, serviceName="CA 19 - 9 (карцинома поджелудочной)", serviceSynonyms="рак"),
    ])
    assert bio.mis_synonym_candidates("рак", vocab=vocab) == (
        "CA 15 - 3 (молочная железа)",
        "CA 19 - 9 (карцинома поджелудочной)",
    )


def test_unambiguous_synonym_returns_one_candidate():
    vocab = bio._build_vocabularies([
        _row(serviceName="Общий анализ мочи", serviceSynonyms="оам"),
    ])
    assert bio.mis_synonym_candidates("цена оам", vocab=vocab) == ("Общий анализ мочи",)


# --- Анти-перехват: синоним НАЗЫВАЕТ услугу, а не встречается в ней --------

def test_short_synonym_does_not_hijack_a_different_service_name():
    """Свип по живому каталогу (09.09) поймал 182 перехвата из 1200 названий.

    Худший: «CA 15 - 3 (молочная железа)» уходил в «Кальций ионизированный»,
    потому что «ca» — синоним кальция. Онкомаркер молочной железы подменялся
    анализом на кальций. Целевые тесты синонимов этого не видели.

    Инвариант: синоним засчитывается, только если он НАЗЫВАЕТ услугу — то есть
    всё, что осталось в реплике сверх него, это речевая обвязка. Если в остатке
    есть содержательные слова, синоним встретился случайно.
    """
    vocab = bio._build_vocabularies([
        _row(serviceId=1, serviceName="Кальций ионизированный", serviceSynonyms="ca"),
    ])
    assert bio.mis_synonym_candidates("CA 15 - 3 (молочная железа)", vocab=vocab) == ()
    assert bio.mis_synonym_candidates("Candida albicans (M5)", vocab=vocab) == ()


def test_synonym_survives_ordinary_speech_wrapping():
    """Обвязку («сколько стоит», «хочу сдать») синоним пережить обязан."""
    vocab = bio._build_vocabularies([
        _row(serviceId=1, serviceName="сТ4 (FT4) Свободный тироксин", serviceSynonyms="тироксин"),
        _row(serviceId=2, serviceName="Общий анализ крови (полный)", serviceSynonyms="кровь с лейкоформулой"),
    ])
    assert bio.mis_synonym_candidates("сколько стоит тироксин", vocab=vocab) == ("сТ4 (FT4) Свободный тироксин",)
    assert bio.mis_synonym_candidates("кровь с лейкоформулой сдать хочу", vocab=vocab) == ("Общий анализ крови (полный)",)
    assert bio.mis_synonym_candidates("тироксин", vocab=vocab) == ("сТ4 (FT4) Свободный тироксин",)


# --- Развилка на неоднозначном синониме ------------------------------------

def _vocab_for(pairs: list[tuple[str, str]]) -> "bio._MisVocabularies":
    return bio._build_vocabularies(
        [_row(serviceId=i, serviceName=name, serviceSynonyms=syn) for i, (name, syn) in enumerate(pairs, 1)]
    )


def test_ambiguous_synonym_builds_a_fork_with_every_candidate(monkeypatch):
    """«оак» в живом словаре клиники повешен на ТРИ услуги, «рак» — на пять.

    Решение владельца: показываем полный список вариантов, выбирает пациент.
    Молча взять один — это класс «дезинформация ценой».
    """
    from messengers_router.services import _prices_helpers as ph

    names = ["CA 15 - 3 (молочная железа)", "CA 19 - 9 (карцинома поджелудочной железы)"]
    for n in names:
        assert any(str(r.get("serviceName")) == n for r in ROWS), f"нет в каталоге: {n}"

    vocab = _vocab_for([(n, "рак") for n in names])
    monkeypatch.setattr(bio, "_vocabularies", lambda: vocab)

    payload = ph._build_synonym_fork_payload("сколько стоит рак", ROWS)

    assert payload is not None
    assert payload["service_kind"] == "family_query"
    shown = {str(v.get("serviceName")) for v in payload["family_variants"]}
    assert set(names) <= shown, sorted(shown)
    assert all(str(v.get("care_setting_label") or "") for v in payload["family_variants"])


def test_unambiguous_synonym_does_not_build_a_fork(monkeypatch):
    """Однозначный синоним — обычный одиночный путь, развилка не нужна."""
    from messengers_router.services import _prices_helpers as ph

    vocab = _vocab_for([("Ферритин", "депо железа")])
    monkeypatch.setattr(bio, "_vocabularies", lambda: vocab)
    assert ph._build_synonym_fork_payload("депо железа", ROWS) is None


def test_fork_drops_candidates_absent_from_region_price(monkeypatch):
    """Кандидат вне прайса региона в развилку не попадает; если остаётся один —
    развилки нет вовсе."""
    from messengers_router.services import _prices_helpers as ph

    vocab = _vocab_for([("Ферритин", "оба"), ("Услуги которой нет в прайсе", "оба")])
    monkeypatch.setattr(bio, "_vocabularies", lambda: vocab)
    assert ph._build_synonym_fork_payload("оба", ROWS) is None


def test_no_synonym_means_no_fork(monkeypatch):
    from messengers_router.services import _prices_helpers as ph

    vocab = _vocab_for([("Ферритин", "депо железа")])
    monkeypatch.setattr(bio, "_vocabularies", lambda: vocab)
    assert ph._build_synonym_fork_payload("узи щитовидной железы", ROWS) is None


def test_price_info_returns_the_fork_for_an_ambiguous_synonym(monkeypatch):
    """Проводка: развилка доходит до payload пациента, а не остаётся в хелпере.

    Стоит ПОСЛЕ мульти-пути (корзина «оак, ферритин» считается корзиной) и ДО
    семейного — словарь клиники доверенней лексической догадки.
    """
    import asyncio

    from messengers_router.services import Services

    names = ["CA 15 - 3 (молочная железа)", "CA 19 - 9 (карцинома поджелудочной железы)"]
    vocab = _vocab_for([(n, "рак") for n in names])
    monkeypatch.setattr(bio, "_vocabularies", lambda: vocab)

    payload = asyncio.run(Services().price_info("сколько стоит рак", {}))

    assert payload.get("note") == "price_synonym_ambiguous"
    shown = {str(v.get("serviceName")) for v in payload.get("family_variants") or []}
    assert set(names) <= shown, sorted(shown)


def test_basket_still_wins_over_synonym_fork(monkeypatch):
    """Корзина из нескольких услуг не должна перехватываться развилкой синонима."""
    import asyncio

    from messengers_router.services import Services

    vocab = _vocab_for([
        ("CA 15 - 3 (молочная железа)", "рак"),
        ("CA 19 - 9 (карцинома поджелудочной железы)", "рак"),
    ])
    monkeypatch.setattr(bio, "_vocabularies", lambda: vocab)

    payload = asyncio.run(Services().price_info("ферритин, глюкоза", {}))
    assert payload.get("note") != "price_synonym_ambiguous"


def test_fork_orders_candidates_by_price_ascending(monkeypatch):
    """Развилка показывает варианты от дешёвого к дорогому.

    До этого порядок брался из выгрузки МИС, то есть был произвольным с точки
    зрения пациента. Для списка, между которым человек выбирает, порядок — часть
    ответа: дешёвое первым, дороже ниже.
    """
    from messengers_router.services import _prices_helpers as ph

    names = ["CA 19 - 9 (карцинома поджелудочной железы)", "HE4", "CA 15 - 3 (молочная железа)"]
    vocab = _vocab_for([(n, "рак") for n in names])
    monkeypatch.setattr(bio, "_vocabularies", lambda: vocab)

    payload = ph._build_synonym_fork_payload("сколько стоит рак", ROWS)
    assert payload is not None

    costs = [float(v.get("cost") or 0) for v in payload["family_variants"]]
    assert costs == sorted(costs), costs


def test_synonym_survives_a_dropped_preposition():
    """Потерянный предлог не должен ломать синоним.

    Прод 15.09: «сколько стоит кровь С лейкоформулой» доходило до матчера как
    «сколько стоит кровь лейкоформулой» — предлог срезается выше по пути. Синоним
    в словаре записан с предлогом, матч шёл непрерывной цепочкой токенов и не
    складывался. Пациент получал «Похоже, вы имели в виду "Лейкоцитарная
    формула"?» — это ДРУГОЕ исследование, 210 руб. против 550.

    Инвариант: предлоги не значащая часть названия услуги ни в словаре клиники,
    ни в реплике пациента. Сверяем последовательности без них.
    """
    vocab = _vocab_for([("Общий анализ крови (полный)", "кровь с лейкоформулой")])
    for phrase in (
        "сколько стоит кровь с лейкоформулой",
        "сколько стоит кровь лейкоформулой",
        "кровь лейкоформулой",
    ):
        assert bio.mis_synonym_candidates(phrase, vocab=vocab) == ("Общий анализ крови (полный)",), phrase


def test_preposition_tolerance_does_not_open_the_door_to_hijacking():
    """Послабление не должно вернуть перехват чужих названий."""
    vocab = _vocab_for([("Кальций ионизированный", "ca")])
    assert bio.mis_synonym_candidates("CA 15 - 3 (молочная железа)", vocab=vocab) == ()
    assert bio.mis_synonym_candidates("Candida albicans (M5)", vocab=vocab) == ()
