from agent_logic_2.doctor_name_matching import resolve_schedule_surname, surname_variants


def test_surname_variants_include_common_male_case_forms():
    assert "Трубин" in surname_variants("Трубину")
    assert "Иванов" in surname_variants("Иванову")
    assert "Петров" in surname_variants("Петровым")


def test_surname_variants_include_common_female_case_forms():
    assert "Белохвостикова" in surname_variants("Белохвостиковой")
    assert "Султанова" in surname_variants("Султановой")


def test_resolve_schedule_surname_accepts_inflected_doctor_forms():
    doctors = [
        {"fio": "Трубин Андрей Сергеевич"},
        {"fio": "Белохвостикова Ирина Викторовна"},
        {"fio": "Султанова Лилия Равильевна"},
    ]

    assert resolve_schedule_surname("хочу к Трубину", doctors) == "Трубин"
    assert resolve_schedule_surname("цена у Белохвостиковой", doctors) == "Белохвостикова"
    assert resolve_schedule_surname("прием у Султановой", doctors) == "Султанова"
