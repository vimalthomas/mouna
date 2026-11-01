from datetime import datetime, timedelta, timezone

from sqlalchemy import Integer, cast, func
from sqlalchemy.orm import Session

from db.models import Attempt, Sign, SignProgress

MIN_ATTEMPTS_FOR_WEAK_SIGN = 3
WEAK_SIGN_ACCURACY_THRESHOLD = 0.70
MAX_WEAK_SIGNS = 5


def get_user_stats(db: Session, user_id: int) -> dict:
    total_sessions = (
        db.query(func.count(func.distinct(Attempt.session_id)))
        .filter(Attempt.user_id == user_id)
        .scalar()
        or 0
    )
    signs_learned = (
        db.query(func.count(SignProgress.id))
        .filter(SignProgress.user_id == user_id, SignProgress.completed.is_(True))
        .scalar()
        or 0
    )

    total_attempts, correct_attempts = (
        db.query(func.count(Attempt.id), func.coalesce(func.sum(cast(Attempt.correct, Integer)), 0))
        .filter(Attempt.user_id == user_id)
        .one()
    )
    accuracy = round((correct_attempts / total_attempts) * 100) if total_attempts else 0

    streak = _compute_streak(db, user_id)
    weak_signs = _get_weak_signs(db, user_id)
    signs = _get_all_sign_progress(db, user_id)

    return {
        "total_sessions": total_sessions,
        "signs_learned": signs_learned,
        "accuracy": accuracy,
        "streak": streak,
        "weak_signs": weak_signs,
        "signs": signs,
    }


def _compute_streak(db: Session, user_id: int) -> int:
    rows = (
        db.query(func.distinct(func.date(Attempt.created_at)))
        .filter(Attempt.user_id == user_id)
        .all()
    )
    practiced_days = {row[0] for row in rows}
    if not practiced_days:
        return 0

    today = datetime.now(timezone.utc).date()
    streak = 0
    cursor = today
    if cursor not in practiced_days:
        cursor -= timedelta(days=1)
        if cursor not in practiced_days:
            return 0
    while cursor in practiced_days:
        streak += 1
        cursor -= timedelta(days=1)
    return streak


def _get_weak_signs(db: Session, user_id: int) -> list[dict]:
    rows = (
        db.query(
            Sign.gloss,
            func.count(Attempt.id).label("total"),
            func.coalesce(func.sum(cast(Attempt.correct, Integer)), 0).label("correct"),
        )
        .join(Attempt, Attempt.sign_id == Sign.id)
        .filter(Attempt.user_id == user_id)
        .group_by(Sign.gloss)
        .having(func.count(Attempt.id) >= MIN_ATTEMPTS_FOR_WEAK_SIGN)
        .all()
    )

    weak = []
    for gloss, total, correct in rows:
        accuracy = correct / total
        if accuracy < WEAK_SIGN_ACCURACY_THRESHOLD:
            weak.append({"sign": gloss, "accuracy": round(accuracy * 100)})

    weak.sort(key=lambda w: w["accuracy"])
    return weak[:MAX_WEAK_SIGNS]


def _get_all_sign_progress(db: Session, user_id: int) -> list[dict]:
    rows = (
        db.query(Sign.gloss, SignProgress)
        .join(Sign, Sign.id == SignProgress.sign_id)
        .filter(SignProgress.user_id == user_id)
        .order_by(SignProgress.updated_at.desc())
        .all()
    )

    signs = []
    for gloss, progress in rows:
        accuracy = (
            round((progress.correct_attempts / progress.total_attempts) * 100)
            if progress.total_attempts
            else 0
        )
        signs.append({
            "sign": gloss,
            "total_attempts": progress.total_attempts,
            "correct_attempts": progress.correct_attempts,
            "accuracy": accuracy,
            "completed": progress.completed,
            "completed_at": progress.completed_at.isoformat() if progress.completed_at else None,
        })
    return signs
