from datetime import datetime, timezone

from sqlalchemy.orm import Session

from config import settings
from db.models import Attempt, SignProgress
from db.signs_repo import get_or_create_sign


def record_attempt(
    db: Session,
    user_id: int,
    gloss: str,
    correct: bool,
    confidence: float,
    session_id: str,
) -> tuple[Attempt, SignProgress, bool]:
    sign = get_or_create_sign(db, gloss)

    attempt = Attempt(
        user_id=user_id,
        sign_id=sign.id,
        session_id=session_id,
        correct=correct,
        confidence=confidence,
    )
    db.add(attempt)

    progress = (
        db.query(SignProgress)
        .filter(SignProgress.user_id == user_id, SignProgress.sign_id == sign.id)
        .first()
    )
    if progress is None:
        progress = SignProgress(
            user_id=user_id,
            sign_id=sign.id,
            total_attempts=0,
            correct_attempts=0,
            completed=False,
        )
        db.add(progress)

    progress.total_attempts += 1
    if correct:
        progress.correct_attempts += 1

    newly_completed = False
    if not progress.completed:
        accuracy = progress.correct_attempts / progress.total_attempts
        if (
            progress.total_attempts >= settings.min_attempts_for_completion
            and accuracy >= settings.completion_accuracy_threshold
        ):
            progress.completed = True
            progress.completed_at = datetime.now(timezone.utc)
            newly_completed = True

    db.commit()
    db.refresh(attempt)
    db.refresh(progress)
    return attempt, progress, newly_completed
