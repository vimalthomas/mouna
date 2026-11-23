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

    db.flush()  # make this attempt visible to the recent-window query below
    newly_completed = False
    if not progress.completed and is_learned(db, user_id, sign.id):
        progress.completed = True
        progress.completed_at = datetime.now(timezone.utc)
        newly_completed = True

    db.commit()
    db.refresh(attempt)
    db.refresh(progress)
    return attempt, progress, newly_completed


# A sign counts as learned when the learner can do it NOW: at least
# LEARNED_MIN_CORRECT of their last LEARNED_WINDOW attempts were correct.
# (The old rule -- lifetime accuracy >= 70% over >= 5 attempts -- meant early
# misses could block a sign forever, e.g. 6/11 stays below 70% no matter how
# many recent attempts succeed.)
LEARNED_WINDOW = 5
LEARNED_MIN_CORRECT = 3


def is_learned(db: Session, user_id: int, sign_id: int) -> bool:
    recent = (
        db.query(Attempt.correct)
        .filter(Attempt.user_id == user_id, Attempt.sign_id == sign_id)
        .order_by(Attempt.created_at.desc(), Attempt.id.desc())
        .limit(LEARNED_WINDOW)
        .all()
    )
    return sum(1 for (c,) in recent if c) >= LEARNED_MIN_CORRECT
