import pytest

from agent_logic_2.gradio_ui.shared import auth as auth_mod
from agent_logic_2.gradio_ui.shared import user_store


@pytest.fixture()
def isolated_users_db(tmp_path):
    db_path = tmp_path / "users_db.json"
    user_store._set_users_db_path_for_tests(db_path)
    yield db_path
    user_store._set_users_db_path_for_tests(None)


def test_user_store_create_and_verify_password(isolated_users_db):
    created = user_store.create_basic_user(
        name="Иван Петров",
        login="ivan.petrov",
        password="secret123",
    )

    assert created["login"] == "ivan.petrov"
    assert user_store.verify_basic_user("ivan.petrov", "secret123") is True
    assert user_store.verify_basic_user("ivan.petrov", "wrong") is False

    rows = user_store.list_users()
    assert len(rows) == 1
    assert rows[0]["name"] == "Иван Петров"
    assert rows[0]["active"] is True


def test_user_store_blocks_reserved_and_duplicate_logins(isolated_users_db):
    with pytest.raises(ValueError):
        user_store.create_basic_user(name="Root User", login="root", password="12345")

    user_store.create_basic_user(name="Petya", login="petya", password="12345")

    with pytest.raises(ValueError):
        user_store.create_basic_user(name="Petya 2", login="petya", password="12345")

    with pytest.raises(ValueError):
        user_store.create_basic_user(
            name="Reserved",
            login="service",
            password="12345",
            deny_logins={"service"},
        )


def test_auth_accepts_dynamic_basic_user(monkeypatch, isolated_users_db):
    monkeypatch.setattr(auth_mod, "_basic_creds", lambda: None)
    monkeypatch.setattr(auth_mod, "_admin_creds", lambda: None)

    user_store.create_basic_user(name="User", login="basic.user", password="pass-123")

    assert auth_mod.check_auth("basic.user", "pass-123") is True
    assert auth_mod.check_auth("basic.user", "wrong") is False
    assert auth_mod.get_user_role("basic.user") == "basic"


def test_auth_role_admin_and_legacy_paths(monkeypatch, isolated_users_db):
    monkeypatch.setattr(auth_mod, "_admin_creds", lambda: ("SuperAdmin", "adm-pass"))
    monkeypatch.setattr(auth_mod, "_basic_creds", lambda: ("legacy", "legacy-pass"))

    assert auth_mod.get_user_role("superadmin") == "admin"
    assert auth_mod.get_user_role("root") == "admin"
    assert auth_mod.get_user_role("legacy") == "basic"

    monkeypatch.setattr(auth_mod, "_admin_creds", lambda: None)
    assert auth_mod.get_user_role("legacy") == "admin"


def test_reserved_logins_collects_configured_credentials(monkeypatch, isolated_users_db):
    monkeypatch.setattr(auth_mod, "_admin_creds", lambda: ("Boss", "x"))
    monkeypatch.setattr(auth_mod, "_basic_creds", lambda: ("Viewer", "y"))

    reserved = auth_mod.get_reserved_logins()

    assert "admin" in reserved
    assert "root" in reserved
    assert "boss" in reserved
    assert "viewer" in reserved
