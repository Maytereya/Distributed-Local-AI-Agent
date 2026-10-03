"""Памятку подготовки выбирает LLM: по предмету памятки, а не по упоминанию в ней (02.10).

## Что произошло

BUG-2026-10-01-PREPARE-MENTION-AS-SUBJECT: выбор по совпадению слов отдавал «подготовке к
анализу на сахар» памятку ФКС со слабительным (в её списке еды — «Желе, сахар, мед»),
колоноскопии — памятку анализа кала «Колонофлор-16», ФГДС — «профиль анализов перед
эндоскопией». Стенд 02.10 по 32 вопросам: верную памятку получали 14, чужую — 14.

## Как теперь

Правила только собирают кандидатов: памятки МИС и, пока клиника не перенесёт их в МИС,
памятки ПАЦИЕНТУ из базы знаний («ПАМЯТКА …»). Скрипты администраторов в подготовку не
попадают — решение владельца 02.10. Выбирает LLM, номером из списка: выдумать памятку
она не может; null — памятки нет, бот предлагает оператора.

## Инварианты

- `prepare_admin_script_leaks`: документ базы знаний не «ПАМЯТКА» кандидатом не становится;
- ответ LLM вне списка, сбой, strict, выключатель — политика без LLM (`_choose_without_llm`):
  только карточки МИС, названные предметом вопроса; лучшая по оценке правил, ничья — обе
  (решение владельца 03.10), больше двух — оффер оператора; «колоноскопия» не называет
  «Колонофлор-16»;
- синоним клиники ставит услугу в кандидаты первой, даже если слов пациента в её памятке
  нет («сахар» → «Глюкоза»);
- вариант запроса без предмета не вытесняет настоящий кандидат: «Сдаче крови» из
  извлечения для записи давал 0.40 всем памяткам крови против 0.38 у «Ферритина»;
- промпт выбора одинаков в обеих локациях;
- две памятки с LLM — только когда вторая карточка МИС названа словами пациента И LLM
  подтвердила, что подготовка практически та же («ТТГ (TSH)» и «Антитела к рецепторам
  ТТГ», решение владельца 03.10); синоним клиники двойника не даёт.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from messengers_router import router as router_mod
from messengers_router.memory import MemoryStore
from messengers_router.mess_types import Evidence, SessionState
from messengers_router import evidence_keys as ek
from messengers_router.llm_doesnt_work_fallback import build_prepare_fallback_answer, join_wrapped_lines
from messengers_router.response_builder import build_prepare_response
from messengers_router.services import Services
from messengers_router.services import _biomaterial as bio
from messengers_router.services import _common as _common_mod
from messengers_router.services import _prepare_select_llm as PS
from messengers_router.services import core as svc_mod
from messengers_router.services import prepare as prepare_mod

_REPO = Path(__file__).resolve().parents[1]

_FERRITIN = PS.PrepareMemo(PS.SOURCE_MIS, "Ферритин", "Кровь сдают утром натощак, через 8–12 часов после еды.")
_FKS = PS.PrepareMemo(
    PS.SOURCE_KB, "ПАМЯТКА ФКС с наркозом", "За 3 дня бесшлаковая диета: разрешены желе, сахар, мёд. Вечером — «Фортранс»."
)


def run(coro):
    return asyncio.run(coro)


def _llm_replies(monkeypatch, reply) -> list[tuple[str, dict]]:
    calls: list[tuple[str, dict]] = []

    async def fake(prompt, **kwargs):
        calls.append((prompt, kwargs))
        if isinstance(reply, BaseException):
            raise reply
        return reply

    monkeypatch.setattr(PS, "generate_text", fake)
    return calls


def _policy_without_llm(monkeypatch) -> list[list[PS.PrepareMemo]]:
    seen: list[list[PS.PrepareMemo]] = []
    monkeypatch.setattr(PS, "_choose_without_llm", lambda memos: seen.append(list(memos)) or ())
    return seen


# --- Выбор памятки ------------------------------------------------------------


def test_llm_number_picks_that_memo(monkeypatch):
    _llm_replies(monkeypatch, '{"match": 2}')
    assert run(PS.select_prepare_memos("подготовка к ФКС под наркозом", [_FERRITIN, _FKS])) == (_FKS,)


def test_llm_null_means_no_memo_about_the_subject(monkeypatch):
    _llm_replies(monkeypatch, '{"match": null}')
    seen = _policy_without_llm(monkeypatch)

    assert run(PS.select_prepare_memos("подготовка к анализу на сахар", [_FKS])) == ()
    assert seen == [], "null — ответ LLM, а не сбой: политика без LLM не нужна"


@pytest.mark.parametrize(
    "reply", ["не JSON", '{"match": 3}', '{"match": 0}', '{"answer": 1}', '[1]', TimeoutError("llm")]
)
def test_unusable_llm_reply_goes_to_the_policy_without_llm(monkeypatch, reply):
    _llm_replies(monkeypatch, reply)
    seen = _policy_without_llm(monkeypatch)

    run(PS.select_prepare_memos("подготовка к ферритину", [_FERRITIN, _FKS]))

    assert seen == [[_FERRITIN, _FKS]]


@pytest.mark.parametrize("mode, switch_on", [("strict", True), ("hybrid", False)])
def test_no_llm_call_in_strict_mode_or_with_the_switch_off(monkeypatch, mode, switch_on):
    async def must_not_call(*_args, **_kwargs):
        raise AssertionError("LLM не зовём")

    monkeypatch.setattr(PS, "generate_text", must_not_call)
    monkeypatch.setattr(
        _common_mod,
        "_runtime_bool",
        lambda name, default: switch_on if name == "MR_PREPARE_RELEVANCE_LLM_ENABLED" else default,
    )
    seen = _policy_without_llm(monkeypatch)

    run(PS.select_prepare_memos("подготовка к ферритину", [_FERRITIN], runtime_llm_mode=mode))

    assert seen == [[_FERRITIN]]


def test_prompt_numbers_memos_and_shows_only_their_beginning(monkeypatch):
    long_memo = PS.PrepareMemo(PS.SOURCE_MIS, "Глюкоза", "Натощак, воду пить можно. " * 60)
    calls = _llm_replies(monkeypatch, '{"match": 1}')

    run(PS.select_prepare_memos("подготовка к анализу на сахар", [long_memo, _FKS]))

    prompt, kwargs = calls[0]
    lines = prompt.splitlines()
    first = next(line for line in lines if line.startswith("1. "))
    assert first.startswith("1. Глюкоза — Натощак") and first.endswith("…")
    assert len(first) < len("1. Глюкоза — ") + 310
    assert any(line.startswith("2. ПАМЯТКА ФКС с наркозом — ") for line in lines)
    assert "«подготовка к анализу на сахар»" in prompt
    assert kwargs["fmt"] == "json" and kwargs["options"] == {"temperature": 0}


def test_selection_prompt_is_the_same_in_both_locations():
    router_copy = (_REPO / "messengers_router/prompts/prepare_document_select.txt").read_text(encoding="utf-8")
    prod_copy = (_REPO / "app_data/prompts/mr_prepare_document_select.txt").read_text(encoding="utf-8")
    assert router_copy == prod_copy


@pytest.mark.parametrize(
    "reply, expected",
    [
        ('{"match": [1, 2]}', (1, 2)),
        ('{"match": [2, 2]}', (2,)),
        ('{"match": [1, 2, 3]}', None),
        ('{"match": [true]}', None),
        ('{"match": []}', None),
    ],
)
def test_choice_may_be_a_list_of_at_most_two(reply, expected):
    assert PS._parse_choice(reply, 3) == expected


# --- Две практически одинаковые памятки (решение владельца 03.10) ----------------

_TSH = PS.PrepareMemo(PS.SOURCE_MIS, "ТТГ (TSH) тиреотропный гормон", "Кровь натощак через 8–14 часов.", 0.38, True)
_TRAB = PS.PrepareMemo(PS.SOURCE_MIS, "Антитела к рецепторам ТТГ", "Забор крови натощак через 8–14 часов.", 0.38, True)


def _llm_script(monkeypatch, replies: list) -> list[str]:
    prompts: list[str] = []

    async def fake(prompt, **kwargs):
        prompts.append(prompt)
        reply = replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply

    monkeypatch.setattr(PS, "generate_text", fake)
    return prompts


def test_practically_identical_memo_is_added(monkeypatch):
    prompts = _llm_script(monkeypatch, ['{"match": 1}', '{"same": true}'])

    assert run(PS.select_prepare_memos("подготовка к ттг", [_TSH, _TRAB])) == (_TSH, _TRAB)
    assert "«Антитела к рецепторам ТТГ»" in prompts[1] and "«ТТГ (TSH) тиреотропный гормон»" in prompts[1]


@pytest.mark.parametrize("second_reply", ['{"same": false}', "не JSON", TimeoutError("llm")])
def test_memo_with_different_or_unknown_preparation_is_not_added(monkeypatch, second_reply):
    _llm_script(monkeypatch, ['{"match": 1}', second_reply])
    assert run(PS.select_prepare_memos("подготовка к ттг", [_TSH, _TRAB])) == (_TSH,)


def test_no_comparison_without_a_twin_named_by_the_patient(monkeypatch):
    transferrin = PS.PrepareMemo(PS.SOURCE_MIS, "Трансферрин", "Кровь натощак.", 0.2, False)
    prompts = _llm_script(monkeypatch, ['{"match": 1}'])

    assert run(PS.select_prepare_memos("подготовка к ферритину", [_FERRITIN, transferrin])) == (_FERRITIN,)
    assert len(prompts) == 1


def test_clinic_synonym_alone_does_not_make_a_twin(monkeypatch):
    # Синоним клиники «холестерин» указывает и на «Триглицериды» — это не название.
    cholesterol = PS.PrepareMemo(PS.SOURCE_MIS, "Холестерол", "Кровь натощак.", 1.0, True)
    triglycerides = PS.PrepareMemo(PS.SOURCE_MIS, "Триглицериды", "Кровь натощак.", 1.0, True)
    prompts = _llm_script(monkeypatch, ['{"match": 1}'])

    chosen = run(PS.select_prepare_memos("подготовка к анализу на холестерин", [cholesterol, triglycerides]))

    assert chosen == (cholesterol,) and len(prompts) == 1


def test_knowledge_base_variants_are_never_twins(monkeypatch):
    # «С наркозом» и «без наркоза» — разная подготовка; база знаний в сравнение не идёт.
    without = PS.PrepareMemo(PS.SOURCE_KB, "ПАМЯТКА ФКС без наркоза", "Бесшлаковая диета 3 дня.")
    with_anesthesia = PS.PrepareMemo(PS.SOURCE_KB, "ПАМЯТКА ФКС с наркозом", "Бесшлаковая диета, не есть 6 часов.")
    prompts = _llm_script(monkeypatch, ['{"match": 1}'])

    assert run(PS.select_prepare_memos("подготовка к ФКС", [without, with_anesthesia])) == (without,)
    assert len(prompts) == 1


def test_same_memo_prompt_is_the_same_in_both_locations():
    router_copy = (_REPO / "messengers_router/prompts/prepare_same_memo.txt").read_text(encoding="utf-8")
    prod_copy = (_REPO / "app_data/prompts/mr_prepare_same_memo.txt").read_text(encoding="utf-8")
    assert router_copy == prod_copy


# --- Без LLM: только карточки МИС, названные предметом --------------------------


def _mis(title: str, score: float, *, named: bool = True) -> PS.PrepareMemo:
    return PS.PrepareMemo(PS.SOURCE_MIS, title, f"Памятка «{title}»: кровь сдают натощак.", score=score, named=named)


def test_without_llm_the_best_named_mis_memo_wins():
    best, other = _mis("Ферритин", 0.38), _mis("Трансферрин", 0.2)
    assert PS._choose_without_llm([other, best, _mis("Эритропоэтин", 0.72, named=False)]) == (best,)


def test_without_llm_a_tie_gives_both_memos():
    # Решение владельца 03.10: тестовый период, администраторы скажут, какую выдавать.
    tsh, antibodies = _mis("ТТГ (TSH) тиреотропный гормон", 0.38), _mis("Антитела к рецепторам ТТГ", 0.38)
    assert PS._choose_without_llm([tsh, antibodies]) == (tsh, antibodies)


def test_without_llm_more_than_two_tied_is_no_choice():
    lipids = [_mis(title, 1.0) for title in ("Триглицериды", "Холестерол", "Холестерол - ЛПНП")]
    assert PS._choose_without_llm(lipids) == ()


def test_without_llm_knowledge_base_memos_are_not_used():
    # Названия памяток базы знаний объединяют процедуры: без LLM на «ФГДС» ушла бы
    # «ФКС + ФГДС» со слабительным.
    combined = PS.PrepareMemo(PS.SOURCE_KB, "ПАМЯТКА ФКС + ФГДС с наркозом", "Фортранс", score=1.0, named=True)
    assert PS._choose_without_llm([combined]) == ()


@pytest.mark.parametrize(
    "question, title, named",
    [
        ("Как подготовиться к колоноскопии?", "Колонофлор-16 (биоценоз — нормофлора)", False),
        ("подготовка к анализу на глюкозу", "Глюкозотолерантный тест", False),
        ("как подготовиться к стрептатесту", "Стрептомицин (с66)", False),
        ("как подготовиться к спермограмме", "Спермограмма", True),
        ("подготовка к глюкозотолерантному тесту", "Глюкозотолерантный тест (глюкоза+ C-пептид)", True),
        ("подготовка к ттг", "Антитела к рецепторам ТТГ", True),
        ("как сдавать кал на яйца глист", "Кал на яйца глист", True),
    ],
)
def test_title_names_the_subject_only_as_the_same_word(question, title, named):
    # Общее начало разных слов — не название: «колоноскопия» не называет «Колонофлор-16».
    subject_roots = prepare_mod._prepare_term_roots(prepare_mod._prepare_subject_phrase(question))
    assert prepare_mod._prepare_names_subject(subject_roots, prepare_mod._prepare_term_roots(title)) is named


def test_tie_without_llm_answers_with_both_memos_each_under_its_title(monkeypatch):
    monkeypatch.setattr(
        svc_mod.api_service_info,
        "load_service_info",
        lambda: [
            {
                "serviceName": "ТТГ (TSH) тиреотропный гормон",
                "preparation": "Подготовка к исследованию\nКровь лучше сдавать утром натощак, после 8-14 часового\n"
                "перерыва в еде. Воду пить не запрещается.",
            },
            {
                "serviceName": "Антитела к рецепторам ТТГ",
                "preparation": "Забор крови натощак. При наблюдении за динамикой кровь сдавать в одной лаборатории.",
            },
        ],
    )
    monkeypatch.setattr(prepare_mod, "kb_patient_memos", lambda: ())

    async def must_not_call(*_args, **_kwargs):
        raise AssertionError("в strict LLM не зовём — ни для выбора, ни для обёртки")

    monkeypatch.setattr(PS, "generate_text", must_not_call)
    monkeypatch.setattr(prepare_mod.llm_runtime_mod, "generate_text", must_not_call)

    res = run(Services().test_prepare("подготовка к ттг", {"__runtime_llm_mode": "strict"}))
    text = res["prepare"]

    assert res["note"] == "prepare: several memos"
    tsh_at, antibodies_at = text.index("«ТТГ (TSH) тиреотропный гормон»:"), text.index("«Антитела к рецепторам ТТГ»:")
    assert "после 8-14 часового перерыва в еде." in text[tsh_at:antibodies_at], "перенос строки склеен"
    assert "в одной лаборатории" in text[antibodies_at:] and "в одной лаборатории" not in text[:antibodies_at]
    assert "Подготовка к исследованию" not in text


def test_wrapped_lines_are_joined_but_list_items_stay_separate():
    # Памятки МИС приходят из HTML с переносами посреди фразы; построчная нарезка запасного
    # ответа отдавала пациенту обрывки («- рентгеноконтрастных исследований.»).
    text = (
        "Важно! Проба крови отбирается только до проведения\n      рентгеноконтрастных исследований.\n\n"
        "Общие рекомендации:\n  оптимальное время с 8 до 11 часов;\n  не курить за час до\n  тестирования;\n"
        "Важно! Покой не менее\n30 минут.\nПорядок:\n1. Прийти натощак\n2. Взять паспорт"
    )
    assert join_wrapped_lines(text) == [
        "Важно! Проба крови отбирается только до проведения рентгеноконтрастных исследований.",
        "Общие рекомендации:",
        "оптимальное время с 8 до 11 часов;",
        "не курить за час до тестирования;",
        "Важно! Покой не менее 30 минут.",
        "Порядок:",
        "1. Прийти натощак",
        "2. Взять паспорт",
    ]


def test_fallback_answer_keeps_sentences_whole():
    memo = (
        "Кровь лучше сдавать утром натощак, после 8-14 часового\n      перерыва в еде.\n"
        "Важно! Проба крови отбирается только до проведения\n      рентгеноконтрастных исследований."
    )
    answer = build_prepare_fallback_answer("подготовка к ттг", memo)
    assert "после 8-14 часового перерыва в еде." in answer
    assert "\n- рентгеноконтрастных" not in answer and "\n- перерыва" not in answer


# --- Памятки базы знаний: только пациенту --------------------------------------


def _fake_knowledge_base(monkeypatch, docs, *, fail: bool = False) -> None:
    class _Index:
        def get_documents(self, params):
            if fail:
                raise ConnectionError("база знаний недоступна")
            return type("Results", (), {"results": [dict(doc) for doc in docs]})()

    class _Client:
        def index(self, name):
            assert name == "main_index"
            return _Index()

    monkeypatch.setattr(PS.meilisearch, "get_meilisearch_client", lambda: _Client())
    monkeypatch.setattr(PS.meilisearch, "ensure_default_indexes_once", lambda: None)
    monkeypatch.setattr(PS, "_kb_cache", {"at": 0.0, "memos": ()})


def test_admin_scripts_never_become_preparation_candidates(monkeypatch):
    _fake_knowledge_base(
        monkeypatch,
        [
            {"title": "ПАМЯТКА ФКС с наркозом", "content": "<p>За 3 дня — бесшлаковая диета.</p>"},
            {"title": "Скрипт по кольпоскопии", "content": "Записывать к гинекологу на 7–14 день цикла."},
            {"title": "Консультация по введению и удалению внутриматочной спирали", "content": "Сказать пациентке…"},
            {"title": "Инструкция для администраторов", "content": "Куда записывать пациента."},
        ],
    )

    memos = PS.kb_patient_memos()

    assert [memo.title for memo in memos] == ["ПАМЯТКА ФКС с наркозом"]
    assert memos[0].source == PS.SOURCE_KB and memos[0].text == "За 3 дня — бесшлаковая диета."


def test_knowledge_base_failure_leaves_preparation_to_mis(monkeypatch):
    _fake_knowledge_base(monkeypatch, [], fail=True)
    assert PS.kb_patient_memos() == ()


# --- Кандидаты МИС: правила собирают, не выбирают ------------------------------


def _clinic_synonyms(monkeypatch, synonyms: dict[str, str]) -> None:
    fake = bio._MisVocabularies()
    fake.synonym_to_service = dict(synonyms)
    monkeypatch.setattr(bio, "_vocabularies", lambda: fake)


def test_clinic_synonym_puts_the_service_first_without_patient_words_in_its_memo(monkeypatch):
    # В памятке «Глюкозы» слова «сахар» нет, а у «Эритропоэтина» есть — по словам
    # первым был бы он; синоним клиники называет услугу прямо.
    monkeypatch.setattr(
        svc_mod.api_service_info,
        "load_service_info",
        lambda: [
            {"serviceName": "Эритропоэтин", "preparation": "За сутки исключить сахар и сладкое. Кровь сдают утром натощак."},
            {"serviceName": "Глюкоза", "preparation": "Кровь сдают утром натощак, через 8–12 часов после еды."},
        ],
    )
    _clinic_synonyms(monkeypatch, {"сахар": "Глюкоза"})

    candidates = run(Services()._prepare_candidates_from_analysis_api_cache("подготовка к анализу на сахар", {}))

    assert [c.service_title for c in candidates][:1] == ["Глюкоза"]


def test_variant_without_subject_does_not_crowd_out_the_named_service(monkeypatch):
    # «Сдаче крови» кладёт в service_name извлечение для записи. Без предмета такой
    # вариант давал 0.40 любой памятке с конкретными указаниями — больше, чем 0.38 у
    # «Ферритина», и он выпадал из кандидатов (стенд 02.10).
    blood_memos = [
        {"serviceName": f"Гормон {i}", "preparation": "Кровь сдают утром натощак, за сутки исключить алкоголь."}
        for i in range(20)
    ]
    monkeypatch.setattr(
        svc_mod.api_service_info,
        "load_service_info",
        lambda: blood_memos + [{"serviceName": "Ферритин", "preparation": "Кровь сдают утром натощак."}],
    )

    candidates = run(
        Services()._prepare_candidates_from_analysis_api_cache(
            "как подготовиться к сдаче крови на ферритин", {"service_name": "Сдаче крови"}
        )
    )

    assert [c.service_title for c in candidates] == ["Ферритин"]


@pytest.mark.parametrize(
    "question, entities, expected_first",
    [
        ("как подготовиться к сдаче крови на ферритин", {"service_name": "Сдаче крови"}, "ферритин"),
        ("подготовка к анализу на холестерин", {"service_name": "Анализу"}, "холест"),
        ("как сдавать кал на яйца глист", {}, "яйца глист"),
        ("подготовка к ттг", {}, "ттг"),
    ],
)
def test_named_service_reaches_the_llm_on_live_catalogue(question, entities, expected_first):
    # Известные случаи стенда 02.10 на последнем срезе справочника МИС.
    candidates = run(Services()._prepare_candidates_from_analysis_api_cache(question, entities))
    titles = [c.service_title for c in candidates[: prepare_mod._PREPARE_MIS_CANDIDATES]]
    assert any(expected_first in title.lower() for title in titles), titles


@pytest.mark.parametrize(
    "preparation",
    ["", "<h1>Подготовка к исследованию</h1>"],
    ids=["пустая", "один заголовок"],
)
def test_memo_without_instructions_is_not_offered(monkeypatch, preparation):
    monkeypatch.setattr(
        svc_mod.api_service_info,
        "load_service_info",
        lambda: [{"serviceName": "Пайпель-биопсия", "preparation": preparation}],
    )
    candidates = run(Services()._prepare_candidates_from_analysis_api_cache("подготовка к пайпель-биопсии", {}))
    assert candidates == []


# --- Памятки нет: оффер оператора вопросом -------------------------------------


def test_no_memo_offers_the_operator_and_yes_hands_off(monkeypatch):
    monkeypatch.setattr(svc_mod.api_service_info, "load_service_info", lambda: [])
    monkeypatch.setattr(prepare_mod, "kb_patient_memos", lambda: (_FKS,))
    _llm_replies(monkeypatch, '{"match": null}')

    payload = run(Services().test_prepare("Как подготовиться к кольпоскопии?", {}))

    assert payload["operator_offer"] is True
    assert "кольпоскопии" in payload["prepare"] and payload["prepare"].endswith("Перевести на оператора?")
    assert "Фортранс" not in payload["prepare"]

    state, memory = SessionState(session_id="prepare-no-memo"), MemoryStore()
    envelope = build_prepare_response("PREPARE", Evidence(items={ek.PREPARE: payload}), state, memory)
    assert envelope is not None and envelope.text == payload["prepare"] and envelope.handoff is False

    consumed = router_mod._handle_operator_offer_pending("да", state, memory)
    assert consumed is not None and consumed[2].get(ek.OPERATOR_OFFER_RESPONSE)["handoff"] is True


# --- Живая LLM: известные случаи стенда 02.10 ----------------------------------


@pytest.mark.live
@pytest.mark.parametrize(
    "question, expected_title",
    [
        ("Как подготовиться к колоноскопии?", "фкс"),
        ("Как подготовиться к ФГДС?", "гастроскопия"),
        ("подготовка к анализу на сахар", None),  # без синонима МИС: главное — не памятка ФКС
        ("подготовка к кольпоскопии", None),
    ],
)
def test_live_llm_picks_the_memo_about_the_subject(question, expected_title):
    memos = list(PS.kb_patient_memos())
    assert memos, "нужна живая база знаний"
    chosen = run(PS.select_prepare_memos(question, memos))
    if expected_title is None:
        assert chosen == (), chosen
    else:
        assert len(chosen) == 1 and expected_title in chosen[0].title.lower(), chosen
