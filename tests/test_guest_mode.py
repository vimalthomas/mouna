"""Guest mode (settings.allow_guest): the demo deployment skips sign-in.
Off by default, so a normal install still requires a login."""
import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from config import settings
from db.models import Base
from auth.dependencies import GUEST_EMAIL, get_current_user


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine)()
    yield s
    s.close()


def test_login_required_when_guest_mode_is_off(db, monkeypatch):
    monkeypatch.setattr(settings, "allow_guest", False)
    with pytest.raises(HTTPException) as e:
        get_current_user(token=None, db=db)
    assert e.value.status_code == 401
    with pytest.raises(HTTPException):
        get_current_user(token="not-a-real-token", db=db)


def test_guest_mode_serves_one_shared_guest(db, monkeypatch):
    monkeypatch.setattr(settings, "allow_guest", True)
    a = get_current_user(token=None, db=db)
    b = get_current_user(token="expired-or-junk", db=db)
    assert a.email == b.email == GUEST_EMAIL and a.id == b.id and a.name == "Guest"
