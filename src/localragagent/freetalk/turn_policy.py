"""Top-level user turn classification for FreeTalk."""

from __future__ import annotations

from dataclasses import dataclass
import re

from .contracts import DialogState


TURN_KIND_EMPTY = "empty"
TURN_KIND_HARD_RESET = "hard_reset"
TURN_KIND_HANDOFF_OPERATOR = "handoff_operator"
TURN_KIND_NEGATIVE_FEEDBACK = "negative_feedback"
TURN_KIND_CONTACT = "contact"
TURN_KIND_WEB = "web"
TURN_KIND_GENERAL = "general"

CONTROL_ACTION_NONE = ""
CONTROL_ACTION_RESET_SESSION = "reset_session"
CONTROL_ACTION_HANDOFF_OPERATOR = "handoff_operator"

SESSION_RESET_TEXT = "Диалог очищен. Начинаем заново."
NEGATIVE_FEEDBACK_RESET_TEXT = "Понял. Сбрасываю текущий диалог, чтобы начать заново."
OPERATOR_HANDOFF_TEXT = "Передаю диалог оператору."


_OPERATOR_RE = re.compile(
    r"(?:"
    r"\bоператор\w*"
    r"|\bменеджер\w*"
    r"|\bжив(?:ой|ого|ым|ому|ом)\s+человек\w*"
    r"|\bне\s+бот\w*"
    r"|\bне\s+робот\w*"
    r"|\b(?:дай|дайте|нужен|нужна|позов\w+|перевед\w+|переключ\w+|соедин\w+|свяж\w+)\b"
    r".{0,24}\b(?:оператор\w*|человек\w*|менеджер\w*)"
    r")",
    re.I,
)
_RESET_RE = re.compile(
    r"\b("
    r"hard\s+reset|reset|"
    r"стоп|"
    r"сброс(?:ь|ить)?|"
    r"очист(?:и|ить)\s+(?:диалог|память)|"
    r"забудь\s+вс[её]|"
    r"нов(?:ый|ую)\s+диалог|"
    r"начн[её]м\s+(?:заново|сначала|нов(?:ый|ую)\s+диалог)"
    r")\b",
    re.I,
)
_NEGATIVE_FEEDBACK_RE = re.compile(
    r"\b("
    r"бред\w*|"
    r"что\s+ты\s+нес(?:е|ё)ш\w*|"
    r"ты\s+нес(?:е|ё)ш\w*|"
    r"ты\s+не\s+то\s+пиш\w*|"
    r"неправильн\w*\s+ответ"
    r")\b",
    re.I,
)
_CONTACT_RE = re.compile(
    r"\b("
    r"телефон\w*|"
    r"номер\s+телефон\w*|"
    r"контакт\w*|"
    r"как\s+связаться|"
    r"связаться\s+с\s+клиник\w*|"
    r"регистратур\w*"
    r")\b",
    re.I,
)
_CLINIC_RE = re.compile(r"\b(клиник\w*|наук\w*|мед[\s\-]?центр\w*|филиал\w*)\b", re.I)
_BRANCH_OR_REGISTRY_RE = re.compile(
    r"\b(регистратур\w*|ул\.?|улиц\w*|пр\.?|проспект|ленина|гагарина|ново-садов\w*)\b",
    re.I,
)
_WEB_RE = re.compile(
    r"\b("
    r"в\s+интернет(?:е)?|"
    r"в\s+сети|"
    r"погугли|загугли|"
    r"актуальн\w*|"
    r"свеж\w*|"
    r"последн\w*|"
    r"latest|today|search|lookup"
    r")\b",
    re.I,
)


@dataclass(frozen=True, slots=True)
class TurnDecision:
    kind: str
    confidence: float = 1.0
    control_action: str = CONTROL_ACTION_NONE
    source_mode: str = ""
    flow_relation: str = "none"
    reply_text: str = ""
    reason: str = ""


def is_operator_request(text: str) -> bool:
    return bool(_OPERATOR_RE.search(str(text or "")))


def is_reset_request(text: str) -> bool:
    return bool(_RESET_RE.search(str(text or "")))


def is_negative_feedback(text: str) -> bool:
    return bool(_NEGATIVE_FEEDBACK_RE.search(str(text or "")))


def is_clinic_contact_request(text: str) -> bool:
    probe = str(text or "").strip()
    if not probe:
        return False
    if not _CONTACT_RE.search(probe):
        return False
    return bool(_CLINIC_RE.search(probe) or _BRANCH_OR_REGISTRY_RE.search(probe))


def classify_turn(user_message: str, *, dialog_state: DialogState | None = None) -> TurnDecision:
    """Classify the user's turn before slot extraction or source routing."""

    probe = str(user_message or "").strip()
    if not probe:
        return TurnDecision(kind=TURN_KIND_EMPTY, confidence=1.0, reason="empty")

    active = bool(
        dialog_state
        and (
            dialog_state.flow_active
            or str(dialog_state.flow_kind or "").strip()
            or dialog_state.expected_slots
            or dialog_state.missing_slots
        )
    )

    if is_operator_request(probe):
        return TurnDecision(
            kind=TURN_KIND_HANDOFF_OPERATOR,
            confidence=1.0,
            control_action=CONTROL_ACTION_HANDOFF_OPERATOR,
            flow_relation="interrupt" if active else "none",
            reply_text=OPERATOR_HANDOFF_TEXT,
            reason="operator_request",
        )

    if is_reset_request(probe):
        return TurnDecision(
            kind=TURN_KIND_HARD_RESET,
            confidence=1.0,
            control_action=CONTROL_ACTION_RESET_SESSION,
            flow_relation="interrupt" if active else "none",
            reply_text=SESSION_RESET_TEXT,
            reason="reset_request",
        )

    if is_negative_feedback(probe):
        return TurnDecision(
            kind=TURN_KIND_NEGATIVE_FEEDBACK,
            confidence=0.95,
            control_action=CONTROL_ACTION_RESET_SESSION,
            flow_relation="interrupt" if active else "none",
            reply_text=NEGATIVE_FEEDBACK_RESET_TEXT,
            reason="negative_feedback",
        )

    if is_clinic_contact_request(probe):
        return TurnDecision(
            kind=TURN_KIND_CONTACT,
            confidence=0.95,
            source_mode="contact",
            flow_relation="switch" if active else "none",
            reason="clinic_contact",
        )

    if _WEB_RE.search(probe):
        return TurnDecision(
            kind=TURN_KIND_WEB,
            confidence=0.85,
            source_mode="web",
            flow_relation="switch" if active else "none",
            reason="web_signal",
        )

    return TurnDecision(kind=TURN_KIND_GENERAL, confidence=0.5, reason="default")
