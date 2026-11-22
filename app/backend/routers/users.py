from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from auth.dependencies import get_current_user
from auth.schemas import UserPublic, UserUpdateRequest
from db.models import Sign, SignProgress, User
from db.session import get_db
from services.stats_service import get_review_queue, get_user_stats

router = APIRouter(prefix="/users", tags=["users"])


@router.get("/me", response_model=UserPublic)
def get_me(user: User = Depends(get_current_user)) -> UserPublic:
    return UserPublic.model_validate(user)


@router.patch("/me", response_model=UserPublic)
def update_me(
    payload: UserUpdateRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> UserPublic:
    if payload.name is not None:
        user.name = payload.name
    if payload.avatar_url is not None:
        user.avatar_url = payload.avatar_url
    db.commit()
    db.refresh(user)
    return UserPublic.model_validate(user)


@router.get("/me/stats")
def get_my_stats(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    return get_user_stats(db, user.id)


@router.get("/me/review-queue")
def get_my_review_queue(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    return {"due": get_review_queue(db, user.id)}


@router.get("/me/signs/progress")
def get_my_sign_progress(
    gloss: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    progress = (
        db.query(SignProgress)
        .join(Sign, Sign.id == SignProgress.sign_id)
        .filter(SignProgress.user_id == user.id, Sign.gloss == gloss)
        .first()
    )
    if progress is None:
        return {"total_attempts": 0, "correct_attempts": 0, "accuracy": 0, "completed": False}
    accuracy = (
        round((progress.correct_attempts / progress.total_attempts) * 100)
        if progress.total_attempts
        else 0
    )
    return {
        "total_attempts": progress.total_attempts,
        "correct_attempts": progress.correct_attempts,
        "accuracy": accuracy,
        "completed": progress.completed,
    }
