import asyncio
import json
from pathlib import Path

import pytest

from agent_logic_2 import llama_func_call as lf


EXPECTED_PATH = Path(__file__).with_name("expected_procedures.json")

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


def _ensure_doctors_cache():
    docs = lf.repo.read_all()
    if docs:
        return
    ok = asyncio.run(lf.repo.update(lf.get_all_doctors))
    if not ok:
        pytest.skip("Doctors cache is empty and update failed.")


def _load_expected() -> dict:
    if not EXPECTED_PATH.exists():
        pytest.skip(f"Expected list file missing: {EXPECTED_PATH}")
    return json.loads(EXPECTED_PATH.read_text(encoding="utf-8"))


def _find_fios(variants: list[str], uzi_only: bool) -> list[str]:
    docs = lf._find_docs_by_specialization_variants(variants, uzi_only=uzi_only)
    return [d.get("fio") for d in docs if d.get("fio")]


@pytest.mark.parametrize("query_key", sorted(PROCEDURE_VARIANTS.keys()))
def test_procedure_queries_fio_list(query_key: str):
    _ensure_doctors_cache()
    expected_map = _load_expected()
    expected = expected_map.get(query_key)
    if expected is None:
        pytest.skip(f"No expected list for query: {query_key}")

    variants = PROCEDURE_VARIANTS[query_key]
    actual = _find_fios(variants, uzi_only=query_key in UZI_KEYS)

    expected_set = set(expected)
    actual_set = set(actual)

    missing = sorted(expected_set - actual_set)
    extra = sorted(actual_set - expected_set)

    assert not missing and not extra, (
        f"Query '{query_key}' mismatch.\n"
        f"Missing: {missing}\n"
        f"Extra: {extra}"
    )
