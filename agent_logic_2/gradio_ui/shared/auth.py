from __future__ import annotations

from functools import lru_cache
from typing import Literal

from agent_logic_2 import config as c


def _norm(value: str | None) -> str:
    return str(value or "").strip().lower()


@lru_cache(maxsize=1)
def _basic_creds() -> tuple[str, str] | None:
    username = str(getattr(c, "AUTH_NAME", "") or "").strip()
    password = str(getattr(c, "AUTH_PASS", "") or "").strip()
    if username and password:
        return username, password
    return None


@lru_cache(maxsize=1)
def _admin_creds() -> tuple[str, str] | None:
    # Optional pair in config.ini:
    # [MEILI.<ENV>] auth_admin_name=... / auth_admin_pass=...
    try:
        admin_name = c._get_str("MEILI", "auth_admin_name", default="", legacy_key="AUTH_ADMIN_NAME")
        admin_pass = c._get_str("MEILI", "auth_admin_pass", default="", legacy_key="AUTH_ADMIN_PASS")
    except Exception:
        admin_name, admin_pass = "", ""

    admin_name = str(admin_name or "").strip()
    admin_pass = str(admin_pass or "").strip()
    if admin_name and admin_pass:
        return admin_name, admin_pass
    return None


def check_auth(username, password):
    basic = _basic_creds()
    admin = _admin_creds()

    username_norm = _norm(username)
    password_str = str(password or "")

    if basic:
        basic_name, basic_pass = basic
        if username_norm == _norm(basic_name) and password_str == basic_pass:
            return True

    if admin:
        admin_name, admin_pass = admin
        if password_str == admin_pass and username_norm in {_norm(admin_name), "admin", "root"}:
            return True

    return False


def get_user_role(username: str | None) -> Literal["admin", "basic"]:
    user_norm = _norm(username)
    admin = _admin_creds()

    if admin:
        admin_name, _ = admin
        return "admin" if user_norm in {_norm(admin_name), "admin", "root"} else "basic"

    # Legacy single-user mode: if admin-учетка явно не задана,
    # существующий логин получает полный доступ.
    basic = _basic_creds()
    if basic and user_norm == _norm(basic[0]):
        return "admin"

    return "basic"
