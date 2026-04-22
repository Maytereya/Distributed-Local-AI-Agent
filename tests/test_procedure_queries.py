import pytest

from agent_logic_2 import llama_func_call as lf


PROCEDURE_VARIANTS = {
    "узи печени": [
        "УЗИ печени",
        "Ультразвуковое исследование печени",
        "Сканирование печени",
    ],
    "узи брюшной полости": [
        "УЗИ брюшной полости",
        "Ультразвуковое исследование брюшной полости",
        "УЗИ органов брюшной полости",
    ],
    "узи щитовидной железы": [
        "УЗИ щитовидной железы",
        "Ультразвуковое исследование щитовидной железы",
        "УЗИ щитовидки",
    ],
    "торакоцентез": [
        "торакоцентез",
    ],
}

UZI_KEYS = {
    "узи печени",
    "узи брюшной полости",
    "узи щитовидной железы",
}

FIXTURE_DOCTORS = [
    {
        "id": 1,
        "fio": "Иванов Иван Иванович",
        "specialization": (
            "• УЗИ брюшной полости:\n"
            "- печени\n"
            "- желчного пузыря\n"
            "• УЗИ щитовидной железы"
        ),
    },
    {
        "id": 2,
        "fio": "Петрова Анна Сергеевна",
        "specialization": "• УЗИ щитовидной железы",
    },
    {
        "id": 3,
        "fio": "Сидорова Мария Петровна",
        "specialization": "• УЗИ печени",
    },
    {
        "id": 4,
        "fio": "Кузнецов Олег Викторович",
        "specialization": "• УЗИ органов малого таза",
    },
    {
        "id": 5,
        "fio": "Тюмин Игорь Александрович",
        "specialization": "• Торакоцентез",
    },
    {
        "id": 6,
        "fio": "Шипунов Иван Дмитриевич",
        "specialization": "• Консультация невролога",
    },
]

EXPECTED_BY_QUERY = {
    "узи печени": {
        "Иванов Иван Иванович",
        "Сидорова Мария Петровна",
    },
    "узи брюшной полости": {
        "Иванов Иван Иванович",
    },
    "узи щитовидной железы": {
        "Иванов Иван Иванович",
        "Петрова Анна Сергеевна",
    },
    "торакоцентез": {
        "Тюмин Игорь Александрович",
    },
}

@pytest.fixture(autouse=True)
def _stub_doctors_repo(monkeypatch):
    monkeypatch.setattr(lf.repo, "read_all", lambda: list(FIXTURE_DOCTORS))


def _find_fios(variants: list[str], uzi_only: bool) -> list[str]:
    docs = lf._find_docs_by_specialization_variants(variants, uzi_only=uzi_only)
    return [d.get("fio") for d in docs if d.get("fio")]


@pytest.mark.parametrize("query_key", sorted(PROCEDURE_VARIANTS.keys()))
def test_procedure_queries_fio_list(query_key: str):
    variants = PROCEDURE_VARIANTS[query_key]
    actual = _find_fios(variants, uzi_only=query_key in UZI_KEYS)
    expected_set = EXPECTED_BY_QUERY[query_key]
    actual_set = set(actual)

    missing = sorted(expected_set - actual_set)
    extra = sorted(actual_set - expected_set)

    assert not missing and not extra, (
        f"Query '{query_key}' mismatch.\n"
        f"Missing: {missing}\n"
        f"Extra: {extra}"
    )
