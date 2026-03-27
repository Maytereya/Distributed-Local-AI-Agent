from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_logic_2 import config as c
from agent_logic_2.persist import ensure_dir

logger = logging.getLogger(__name__)

_USERS_DB_FILENAME = "users_db.json"
_USERS_DB_VERSION = 1
_PWD_ALG = "pbkdf2_sha256"
_PWD_ITERATIONS = 390_000

_LOCK = threading.Lock()
_USERS_DB_PATH: Path | None = None


def _norm(value: str | None) -> str:
    return str(value or "").strip().lower()


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _resolve_users_db_path() -> Path:
    global _USERS_DB_PATH
    if _USERS_DB_PATH is not None:
        return _USERS_DB_PATH

    settings_dir = ensure_dir((Path(c.APP_DATA_DIR).expanduser().resolve() / "settings").resolve(), "settings")
    _USERS_DB_PATH = (settings_dir / _USERS_DB_FILENAME).resolve()
    return _USERS_DB_PATH


def _set_users_db_path_for_tests(path: Path | None) -> None:
    global _USERS_DB_PATH
    _USERS_DB_PATH = path.resolve() if path is not None else None


def _empty_db() -> dict[str, Any]:
    return {
        "version": _USERS_DB_VERSION,
        "users": [],
    }


def _set_private_permissions(path: Path) -> None:
    try:
        path.chmod(0o600)
    except Exception:
        # На некоторых файловых системах chmod может быть ограничен.
        pass


def _read_db_unlocked() -> dict[str, Any]:
    db_path = _resolve_users_db_path()
    if not db_path.exists():
        return _empty_db()

    try:
        raw = json.loads(db_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Файл пользователей поврежден: {db_path}") from e

    if not isinstance(raw, dict):
        raise RuntimeError("Формат users_db.json некорректен: ожидается JSON-объект")

    users = raw.get("users", [])
    if not isinstance(users, list):
        raise RuntimeError("Формат users_db.json некорректен: поле 'users' должно быть списком")

    return {
        "version": int(raw.get("version") or _USERS_DB_VERSION),
        "users": [item for item in users if isinstance(item, dict)],
    }


def _write_db_unlocked(data: dict[str, Any]) -> None:
    db_path = _resolve_users_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = db_path.with_suffix(f"{db_path.suffix}.tmp")
    tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    _set_private_permissions(tmp_path)
    os.replace(tmp_path, db_path)
    _set_private_permissions(db_path)


def _hash_password(password: str, salt: bytes, iterations: int) -> str:
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        iterations,
    )
    return digest.hex()


def _build_password_record(password: str) -> dict[str, Any]:
    salt = secrets.token_bytes(16)
    return {
        "pwd_alg": _PWD_ALG,
        "pwd_iter": _PWD_ITERATIONS,
        "pwd_salt": salt.hex(),
        "pwd_hash": _hash_password(password, salt, _PWD_ITERATIONS),
    }


def _verify_password(password: str, user_record: dict[str, Any]) -> bool:
    try:
        alg = str(user_record.get("pwd_alg") or "")
        iterations = int(user_record.get("pwd_iter") or 0)
        salt_hex = str(user_record.get("pwd_salt") or "")
        saved_hash = str(user_record.get("pwd_hash") or "")
        salt = bytes.fromhex(salt_hex)
    except Exception:
        return False

    if alg != _PWD_ALG or iterations <= 0 or not saved_hash:
        return False

    calc_hash = _hash_password(password, salt, iterations)
    return hmac.compare_digest(calc_hash, saved_hash)


def list_users() -> list[dict[str, Any]]:
    with _LOCK:
        data = _read_db_unlocked()

    rows: list[dict[str, Any]] = []
    for user in data["users"]:
        login = str(user.get("login") or "").strip()
        if not login:
            continue
        rows.append(
            {
                "name": str(user.get("name") or "").strip(),
                "login": login,
                "active": bool(user.get("active", True)),
                "created_at": str(user.get("created_at") or ""),
            }
        )

    rows.sort(key=lambda row: _norm(str(row.get("login") or "")))
    return rows


def is_basic_user(login: str | None) -> bool:
    login_norm = _norm(login)
    if not login_norm:
        return False

    try:
        users = list_users()
    except Exception:
        logger.exception("Ошибка чтения users_db при проверке пользователя")
        return False

    return any(_norm(str(user.get("login") or "")) == login_norm and bool(user.get("active", True)) for user in users)


def verify_basic_user(login: str | None, password: str | None) -> bool:
    login_norm = _norm(login)
    password_raw = str(password or "")
    if not login_norm or not password_raw:
        return False

    try:
        with _LOCK:
            data = _read_db_unlocked()
    except Exception:
        logger.exception("Ошибка чтения users_db при проверке пароля")
        return False

    for user in data["users"]:
        if _norm(str(user.get("login") or "")) != login_norm:
            continue
        if not bool(user.get("active", True)):
            return False
        return _verify_password(password_raw, user)

    return False


def create_basic_user(
    name: str,
    login: str,
    password: str,
    *,
    deny_logins: set[str] | None = None,
) -> dict[str, Any]:
    name_clean = str(name or "").strip()
    login_clean = str(login or "").strip()
    password_raw = str(password or "")
    login_norm = _norm(login_clean)
    reserved = {_norm(item) for item in (deny_logins or set()) if _norm(item)}

    if not name_clean:
        raise ValueError("Введите имя пользователя.")
    if len(name_clean) > 120:
        raise ValueError("Имя пользователя слишком длинное (максимум 120 символов).")

    if not login_clean:
        raise ValueError("Введите логин.")
    if len(login_clean) > 64:
        raise ValueError("Логин слишком длинный (максимум 64 символа).")
    if not re.fullmatch(r"[A-Za-z0-9_.@-]+", login_clean):
        raise ValueError("Логин может содержать только латиницу, цифры и символы . _ @ -")
    if login_norm in {"admin", "root"} or login_norm in reserved:
        raise ValueError("Этот логин зарезервирован и не может быть создан как базовый пользователь.")

    if len(password_raw) < 4:
        raise ValueError("Пароль должен содержать не менее 4 символов.")

    with _LOCK:
        data = _read_db_unlocked()
        if any(_norm(str(user.get("login") or "")) == login_norm for user in data["users"]):
            raise ValueError("Пользователь с таким логином уже существует.")

        password_record = _build_password_record(password_raw)
        new_user = {
            "id": secrets.token_hex(12),
            "name": name_clean,
            "login": login_clean,
            "active": True,
            "created_at": _now_utc_iso(),
            **password_record,
        }
        data["users"].append(new_user)
        _write_db_unlocked(data)

    return {
        "name": new_user["name"],
        "login": new_user["login"],
        "active": new_user["active"],
        "created_at": new_user["created_at"],
    }
