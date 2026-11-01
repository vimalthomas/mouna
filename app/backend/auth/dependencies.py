from fastapi import Depends, HTTPException, WebSocket, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from auth.security import decode_token
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


def get_current_user(
    token: str | None = Depends(_oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    if token is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    user = _load_user(db, token)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")
    return user


async def get_current_user_ws(websocket: WebSocket, db: Session) -> User | None:
    token = websocket.query_params.get("token")
    if not token:
        return None
    return _load_user(db, token)
