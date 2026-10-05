"""Мысли о самоубийстве, самоповреждение, передозировка — срочное, а не медвопрос (05.10).

Ревью DLA (находка SYM-1, BUG-2026-10-05-SELF-HARM-AS-ADVICE): этих признаков не знали
ни регулярка URGENT, ни промпт совета по жалобе. «чем лечить, если не хочу жить» по
правилу становилось медвопросом, и совет мог ответить «обычно обращаются к специалисту
нашей клиники: терапевт» без оператора (handoff=false).

Инварианты:
- явная фраза о самоубийстве, самоповреждении или передозировке — срочная по регулярке,
  независимо от LLM; её двойники («убить время», «не хочу жить в этом районе») — нет;
- такая фраза получает шаблон скорой и оператора, что бы ни ответили LLM-классификатор
  (медвопрос, «прочее») и LLM совета (номер врача, «СРОЧНО», «НЕТ», сбой);
- срочность мимо регулярки (иные слова) решают LLM-классификатор и LLM совета: критерий
  есть в обеих копиях обоих промптов; пищевое отравление в критерий совета не входит;
- вторая линия: если предпроверку в роутере уберут, каскад правил пайплайна (early_guards)
  всё равно ставит URGENT раньше медвопроса.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from messengers_router import renderer
from messengers_router import router as router_mod
from messengers_router.memory import MemoryStore
from messengers_router.mess_types import Evidence, RouteDecision, SessionState
from messengers_router.nlu_pipeline import NLUResult
from messengers_router.orchestrator import OrchestratorContext, early_guards, run_pipeline
from messengers_router.policies import detect_urgent
from messengers_router.services import Services
from messengers_router.services import _symptom_specialists as S
from messengers_router.services._regions import _normalize_region_text

_REAL_ADVISE = S.advise  # до подмены фикстурой conftest: совет идёт настоящим путём
_ROOT = Path(__file__).resolve().parents[1]
_SAMARA_BRANCH = "пр.Ленина, 5"

# --- класс фраз -----------------------------------------------------------------------

SELF_HARM_PHRASES = [
    "не хочу жить",
    "Не хочу больше жить",
    "не хочется жить, что делать",
    "жить не хочется совсем",
    "не хочу жить в этом мире",
    "чем лечить, если не хочу жить",
    "что со мной? не хочу больше так жить",
    "думаю покончить с собой",
    "хочу покончить с жизнью",
    "у меня суицидальные мысли",
    "мысли о самоубийстве",
    "хочу умереть",
    "хочу убить себя",
    "порезала себя",
    "режу себя, что со мной",
    "хочу вскрыть вены",
    "порезал себе вены",
    "выпил много таблеток",
    "выпила пачку снотворного",
    "приняла слишком много таблеток от давления",
    "наглотался таблеток что делать",
    "ребенок наглотался таблеток",
    "передозировка феназепама",
    "кажется, у меня передоз",
    "отравилась таблетками",
    "какие таблетки выпить чтобы уснуть и не проснуться",
    "хочу свести счёты с жизнью",
    "нет смысла жить",
    "хочу повеситься",
    "думаю спрыгнуть с крыши",
    # частые формы (05.10, ревью): слитно, с опечаткой, со вставными словами
    "нехочу жить",
    "не хочю жить",
    "не хочу уже жить",
    "я вообще не хочу жить",
    "не хочу я жить",
    "жить уже не хочется",
    "надоело жить",
    "устала жить",
    "устал жить, что делать",
    "незачем жить",
    "не вижу смысла жить",
    "хочу заснуть и не проснуться",
    # передозировка: количество или упаковка, название препарата любое
    "выпила 20 таблеток",
    "выпил 30 таблеток феназепама",
    "20 таблеток выпила, что будет",
    "ребенок съел 10 таблеток",
    "выпил упаковку парацетамола",
    "выпила всю упаковку",
    "приняла блистер антидепрессантов",
    "наглоталась полпачки",
    "отравление лекарствами",
]

# Похожи по словам, но не про угрозу жизни: ни одна не должна стать срочной.
LOOKALIKE_PHRASES = [
    "убить время до приёма",
    "умираю от скуки в очереди",
    "умираю со смеху",
    "не хочу жить в этом районе, есть филиал ближе?",
    "не хочу жить на съёмной квартире",
    "смертельно устал, запишите к терапевту",
    "покончить с долгами",
    "таблетки от давления какие пить",
    "какие таблетки нельзя пить перед анализом",
    "не хочу умереть от рака, какие анализы сдать",
    "выпил таблетку от головы, можно сдавать кровь?",
    "выпила все таблетки по курсу, когда сдавать анализ",
    "убедить себя сходить к врачу",
    "живот режет второй день",
    "порезался, нужна перевязка",
    "хочу умереть от смеха",
    "сколько стоит анализ на передозировку витамина D",
    "мне хоть вешайся с этой записью",
    "наглотался воды в бассейне",
    "отравилась едой, к какому врачу",
    # к частым формам — жильё, страх, бытовое
    "не хочу жить тут, далеко до клиники",
    "не хочу больше жить на две квартиры",
    "надоело жить в этом районе",
    "устала жить на съёмной квартире",
    "храплю по ночам, боюсь уснуть и не проснуться, к какому врачу",
    "мне страшно уснуть и не проснуться, это апноэ?",
    "боюсь передозировки, сколько можно пить",
    "выпил 2 таблетки от головы, можно сдавать кровь?",
    "съел пачку чипсов, болит живот",
    "принимаю 10 таблеток в день, можно ли сдавать анализы",
]

# Принятые компромиссы (05.10): двойники, которые остаются срочными, — исключение для них
# было бы хрупким или требовало бы перечня веществ. Ошибка в безопасную сторону: шаблон
# скорой и оператор. Тест фиксирует сторону ошибки: сужать — осознанно, со встречным свипом.
ACCEPTED_FALSE_ALARMS = [
    "какие симптомы передозировки витамина D",
    "анализ после передозировки железа",
    "выпила упаковку йогурта",
    "выпила много таблеток от головы за неделю, можно сдать анализ?",
    "не хочу жить с родителями",
    "суицидальный риск у подростков, лекция",
    "убил себя на тренировке, болят мышцы",
]


@pytest.mark.parametrize("text", SELF_HARM_PHRASES)
def test_self_harm_and_overdose_are_urgent_by_rule(text):
    assert detect_urgent(text) is True


@pytest.mark.parametrize("text", LOOKALIKE_PHRASES)
def test_lookalikes_of_self_harm_are_not_urgent(text):
    assert detect_urgent(text) is False


@pytest.mark.parametrize("text", ACCEPTED_FALSE_ALARMS)
def test_accepted_false_alarms_err_on_the_safe_side(text):
    assert detect_urgent(text) is True


def test_advice_prompt_names_self_harm_as_urgent():
    for path in (
        _ROOT / "messengers_router" / "prompts" / "symptom_specialists.txt",
        _ROOT / "app_data" / "prompts" / "mr_symptom_specialists.txt",
    ):
        prompt = path.read_text(encoding="utf-8")
        urgent_rule = prompt[prompt.index("Признаки неотложного") : prompt.index("→ ответь: СРОЧНО")]
        urgent_rule = " ".join(urgent_rule.split())
        assert "самоубийств" in urgent_rule, path
        assert "самоповрежд" in urgent_rule, path
        assert "приём большого количества лекарств" in urgent_rule, path
        # пищевое отравление — к врачу, не в скорую: в критерии только лекарства и химия
        assert "отравление лекарствами или химическими веществами" in urgent_rule, path
        assert "или отравление)" not in urgent_rule, path


@pytest.mark.parametrize(
    "path", ["messengers_router/prompts/classifier_patient.txt", "app_data/prompts/mr_classifier_patient.txt"]
)
def test_classifier_prompt_names_self_harm_as_urgent(path):
    # LLM-first: формулировки мимо регулярки («жизнь потеряла смысл») решает классификатор.
    prompt = (_ROOT / path).read_text(encoding="utf-8")
    rule = next(line for line in prompt.splitlines() if line.startswith("- URGENT — и угроза жизни от себя"))
    for sign in ("мысли о самоубийстве", "самоповреждение", "приём большого количества лекарств", "отравление лекарствами"):
        assert sign in rule, (path, sign)
    assert "Текст: жизнь потеряла смысл, хочу исчезнуть\nLabel: URGENT" in prompt
    assert "Текст: устала от всего, не хочу просыпаться\nLabel: URGENT" in prompt
    assert "Текст: умираю от скуки в очереди\nLabel: OTHER" in prompt
    # примеры — именно для LLM: регулярка их не узнаёт
    assert not detect_urgent("жизнь потеряла смысл, хочу исчезнуть")
    assert not detect_urgent("устала от всего, не хочу просыпаться")


# --- вторая линия: каскад правил пайплайна -----------------------------------------


@pytest.mark.parametrize(
    "text",
    ["чем лечить, если не хочу жить", "что со мной? надоело жить", "какие таблетки выпить чтобы уснуть и не проснуться"],
)
def test_pipeline_rules_put_urgent_before_medical_advice(text):
    # Без предпроверки роутера фразу ловит early_guards: URGENT стоит в каскаде раньше
    # правила медвопроса («чем лечить», «что со мной», «какие таблетки»).
    ctx = asyncio.run(early_guards(OrchestratorContext(text=text, state=SessionState(session_id="cascade"))))

    assert ctx.short_circuit is True
    assert ctx.decision is not None and ctx.decision.label == "URGENT"


def test_pipeline_rule_control_medical_advice_without_self_harm():
    # Контроль: та же конструкция без признаков угрозы жизни — медвопрос по правилу.
    ctx = asyncio.run(
        early_guards(OrchestratorContext(text="чем лечить головную боль", state=SessionState(session_id="cascade")))
    )

    assert ctx.decision is not None and ctx.decision.label == "MEDICAL_ADVICE"


def test_pipeline_renders_urgent_template_for_self_harm(monkeypatch):
    async def must_not_advise(*_args, **_kwargs):
        raise AssertionError("срочная фраза не идёт в совет по жалобе")

    monkeypatch.setattr(S, "advise", must_not_advise)
    state, memory = SessionState(session_id="cascade-render"), MemoryStore()

    ctx = asyncio.run(run_pipeline("чем лечить, если не хочу жить", state, Services(), memory))

    assert ctx.decision.label == "URGENT"
    assert ctx.response is not None
    assert ctx.response.text == renderer.render_urgent().text
    assert ctx.response.handoff is True


# --- сквозной путь: правило или LLM × любой ответ совета ------------------------------


class _FakeServices:
    async def _ensure_doctors_cache_loaded(self):
        return [{"regions": [_SAMARA_BRANCH], "unit_links": [{"company_unit_name": "Врач терапевт"}]}]

    async def _samara_region_tokens(self):
        return {_normalize_region_text(_SAMARA_BRANCH)}


def _wire(monkeypatch, *, llm_label: str, advice_answer) -> list[str]:
    """NLU отдаёт `llm_label`; совет — настоящий, на одном самарском терапевте; его LLM
    отвечает `advice_answer`. Возвращает реплики, с которыми звали совет."""

    async def fake_analyze_with_candidates(_text, _state, runtime_options=None):
        return NLUResult(
            decision=RouteDecision(label=llm_label, confidence=0.9, entities={}, flags=set(), source="llm_primary"),
            candidates=[],
            merged_from="llm",
        )

    async def fake_execute_plan(_plan, _state, _services):
        return Evidence()

    async def fake_generate_text(_prompt, **_kwargs):
        if isinstance(advice_answer, Exception):
            raise advice_answer
        return advice_answer

    advised: list[str] = []

    async def real_advice_on_fake_catalog(said, _services):
        advised.append(said)
        return await _REAL_ADVISE(said, _FakeServices())

    async def free_stream(*_args, **_kwargs):
        yield "По техническим вопросам обратитесь к администратору клиники."

    monkeypatch.setattr(router_mod, "analyze_with_candidates", fake_analyze_with_candidates)
    monkeypatch.setattr(router_mod, "execute_plan", fake_execute_plan)
    monkeypatch.setattr(
        router_mod,
        "_env_flag",
        lambda name, default: True if name == "MR_ROUTER_V2_ENABLE" else (False if name == "MR_ROUTER_V2_SHADOW" else default),
    )
    monkeypatch.setattr(S, "generate_text", fake_generate_text)
    monkeypatch.setattr(S, "advise", real_advice_on_fake_catalog)
    monkeypatch.setattr(renderer, "render_stream", free_stream)
    return advised


def _say(text: str) -> tuple[str, bool]:
    async def go():
        state, memory = SessionState(session_id="urgent-self-harm"), MemoryStore()
        out = [env async for env in router_mod.patient_routing_stream(text, state, Services(), memory)]
        return "".join(env.text for env in out if env.text), any(env.handoff for env in out)

    return asyncio.run(go())


def test_harness_reaches_the_advice(monkeypatch):
    # Контроль стенда: обычная жалоба доходит до настоящего совета и получает терапевта.
    advised = _wire(monkeypatch, llm_label="MEDICAL_ADVICE", advice_answer="1")

    text, handoff = _say("у меня болит голова 3 дня")

    assert advised == ["у меня болит голова 3 дня"]
    assert "терапевт" in text
    assert handoff is False


@pytest.mark.parametrize(
    "advice_answer",
    ["1", "СРОЧНО", "НЕТ", TimeoutError()],
    ids=["advice_picks_doctor", "advice_urgent", "advice_no", "advice_llm_failure"],
)
@pytest.mark.parametrize(
    "text",
    [
        # медвопрос по правилу («чем лечить», «что со мной», «какие таблетки») — короткий путь
        "чем лечить, если не хочу жить",
        "что со мной? не хочу больше так жить",
        "какие таблетки выпить чтобы уснуть и не проснуться",
        # без правила медвопроса — метку ставила бы LLM
        "думаю покончить с собой",
        "у меня суицидальные мысли",
        "выпила пачку снотворного",
        "наглотался таблеток что делать",
        "хочу вскрыть вены",
    ],
)
@pytest.mark.parametrize("llm_label", ["MEDICAL_ADVICE", "OTHER"])
def test_self_harm_gets_urgent_template_whatever_llms_answer(monkeypatch, text, llm_label, advice_answer):
    advised = _wire(monkeypatch, llm_label=llm_label, advice_answer=advice_answer)

    answer, handoff = _say(text)

    assert answer == renderer.render_urgent().text
    assert handoff is True
    assert advised == []  # до совета по жалобе срочная фраза не доходит
