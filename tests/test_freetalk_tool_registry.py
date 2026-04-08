import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from localragagent.freetalk.tool_registry import (
    is_about_agent_query,
    is_medical_query,
    select_tool_plan,
    should_use_web_search,
)


def test_therapist_query_is_medical_and_routes_to_doctors_tools():
    text = "Кто из терапевтов принимает в клинике?"
    assert is_medical_query(text) is True
    plan = select_tool_plan(text, include_meili_tools=False)
    assert plan[:2] == ["doctors_info", "doctors_schedule_week"]


def test_freeform_find_request_prefers_web_search():
    text = "Поищи стоимость альскирена в РФ"
    assert is_medical_query(text) is False
    assert should_use_web_search(text) is True


def test_medical_query_does_not_trigger_web_search_by_default():
    text = "Поищи в интернете, кто из терапевтов принимает в клинике"
    assert is_medical_query(text) is True
    assert should_use_web_search(text) is False


def test_typo_therapist_in_clinic_query_still_routes_to_doctors_tools():
    text = "Выведи пожалуйста теарапевтов клиники"
    assert is_medical_query(text) is True
    plan = select_tool_plan(text, include_meili_tools=False)
    assert plan[:2] == ["doctors_info", "doctors_schedule_week"]


def test_explicit_meili_query_routes_to_main_and_news_tools():
    text = "Поищи в meiisearch информацию по налоговому вычету"
    assert is_medical_query(text) is True
    assert should_use_web_search(text) is False
    plan = select_tool_plan(text, include_meili_tools=True)
    assert plan[:2] == ["main_index_info", "news_info"]


def test_clinic_news_query_routes_to_news_meili_tool():
    text = "Новости клиники Наука за этот месяц"
    assert is_medical_query(text) is True
    assert should_use_web_search(text) is False
    plan = select_tool_plan(text, include_meili_tools=True)
    assert plan[:2] == ["news_info", "main_index_info"]


def test_loaded_documents_command_routes_to_main_index_tool():
    text = "Поищи в загруженных документах клиники информацию о подготовке к МРТ"
    assert is_medical_query(text) is True
    assert should_use_web_search(text) is False
    plan = select_tool_plan(text, include_meili_tools=True)
    assert plan[:2] == ["main_index_info", "news_info"]


def test_schedule_short_query_routes_to_schedule_tools():
    text = "Расписание Дразнин"
    assert is_medical_query(text) is True
    plan = select_tool_plan(text, include_meili_tools=True)
    assert plan[:2] == ["doctors_schedule_week", "doctors_info"]


def test_slots_query_routes_to_schedule_tools():
    text = "Есть свободные слоты у Дразнина?"
    assert is_medical_query(text) is True
    plan = select_tool_plan(text, include_meili_tools=True)
    assert plan[:2] == ["doctors_schedule_week", "doctors_info"]


def test_about_agent_query_with_zanimaeshsya_is_detected():
    text = "Чем ты занимаешься?"
    assert is_about_agent_query(text) is True


def test_about_agent_query_does_not_steal_doctor_question():
    text = "Чем занимается врач Дразнин в клинике?"
    assert is_about_agent_query(text) is False
