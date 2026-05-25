from messengers_router.services._prices_helpers import _rank_price_rows


def _names(rows):
    return [str(r.get("serviceName") or "") for r in rows]


# Прайс-каталог пишет «Вирус кори Ig M/Ig G», пациент — «антитела на корь».
CATALOG = [
    {"serviceName": "Вирус кори Ig M", "serviceHomecode": "338", "cost": 500},
    {"serviceName": "Вирус кори Ig G", "serviceHomecode": "339", "cost": 500},
    {"serviceName": "Иммуноглобулин E общий", "serviceHomecode": "111", "cost": 700},
    {"serviceName": "Вирус краснухи Ig G", "serviceHomecode": "222", "cost": 600},
    {"serviceName": "Общий анализ крови", "serviceHomecode": "1", "cost": 300},
]


def test_measles_antibodies_query_matches_virus_kori_rows():
    """Регрессия: «антитела на корь» должен находить «Вирус кори Ig M/Ig G»."""
    for query in ("антитела на корь", "анализ антитела на корь", "корь антитела"):
        found = _names(_rank_price_rows(CATALOG, query))
        assert "Вирус кори Ig M" in found, query
        assert "Вирус кори Ig G" in found, query
        # не подмешиваем нерелевантные «антитела»/другие вирусы
        assert "Иммуноглобулин E общий" not in found, query
        assert "Вирус краснухи Ig G" not in found, query


def test_unrelated_lab_queries_unaffected():
    assert _names(_rank_price_rows(CATALOG, "общий анализ крови")) == ["Общий анализ крови"]
    # запрос про другой вирус не должен возвращать корь
    found = _names(_rank_price_rows(CATALOG, "антитела на краснуху"))
    assert "Вирус кори Ig M" not in found
    assert "Вирус кори Ig G" not in found
