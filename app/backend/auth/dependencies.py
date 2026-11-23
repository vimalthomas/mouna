from fastapi import Depends, HTTPException, WebSocket, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

import secrets

from auth.security import decode_token, hash_password
from config import settings
from db.models import User
from db.session import get_db

_oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)


def _load_user(db: Session, token: str, expected_type: str = "access") -> User | None:
    try:
        payload = decode_token(token, expected_type)
    except ValueError:
        return None
    user_id = payload.get("sub")
    if user_id is None:
        return None
    return db.query(User).filter(User.id == int(user_id)).first()


GUEST_EMAIL = "guest@mouna.demo"


def guest_user(db: Session) -> User:
    """The one shared account used in guest mode (settings.allow_guest).
    Created on first use with a random password nobody knows, so it can't be
    signed into -- it only exists so attempts and progress have an owner."""
    user = db.query(User).filter(User.email == GUEST_EMAIL).first()
    if user is None:
        user = User(email=GUEST_EMAIL, name="Guest", hashed_password=hash_password(secrets.token_urlsafe(32)))
        db.add(user)
        db.commit()
        db.refresh(user)
    return user


def get_current_user(
    token: str | None = Depends(_oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    user = _load_user(db, token) if token else None
    if user is not None:
        return user
    if settings.allow_guest:
        return guest_user(db)
    if token is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")


async def get_current_user_ws(websocket: WebSocket, db: Session) -> User | None:
    token = websocket.query_params.get("token")
    user = _load_user(db, token) if token and token not in ("null", "undefined") else None
    if user is None and settings.allow_guest:
        return guest_user(db)
    return user
