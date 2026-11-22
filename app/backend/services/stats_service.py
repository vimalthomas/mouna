from datetime import datetime, timedelta, timezone

from sqlalchemy import Integer, cast, func
from sqlalchemy.orm import Session

from db.models import Attempt, Sign, SignProgress

MIN_ATTEMPTS_FOR_WEAK_SIGN = 3
WEAK_SIGN_ACCURACY_THRESHOLD = 0.70
MAX_WEAK_SIGNS = 5

# Review-queue spacing: a first, legible pass at spaced repetition (P10 in
# the late-signers paper: "going back over what you learned last week, what
# you learned a month ago"), NOT a real SM-2/Leitner implementation -- no
# ease factors, no per-review history, just a linear interval from current
# accuracy. Disclosed as a simple heuristic on purpose rather than presented
# as more rigorous than it is, same as the NMS thresholds' honesty pattern.
REVIEW_INTERVAL_MIN_DAYS = 1.0   # accuracy 0% -> due again after 1 day
REVIEW_INTERVAL_MAX_DAYS = 10.0  # accuracy 100% -> due again after 10 days
MAX_REVIEW_QUEUE = 10


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


def get_review_queue(db: Session, user_id: int) -> list[dict]:
    """Signs the user has already attempted at least once, ranked by how
    overdue they are for review. `updated_at` on SignProgress already
    tracks "last touched" (see db/models.py), so this needs no new schema
    -- just a priority computed from data that already exists.

    interval_days scales linearly with accuracy: a sign the user gets right
    often resurfaces less often, one they struggle with resurfaces sooner.
    A sign is "due" once days since last practiced >= its interval."""
    rows = (
        db.query(Sign.gloss, SignProgress)
        .join(Sign, Sign.id == SignProgress.sign_id)
        .filter(SignProgress.user_id == user_id, SignProgress.total_attempts > 0)
        .all()
    )

    now = datetime.now(timezone.utc)
    queue = []
    for gloss, progress in rows:
        accuracy = progress.correct_attempts / progress.total_attempts
        updated_at = progress.updated_at
        if updated_at.tzinfo is None:
            updated_at = updated_at.replace(tzinfo=timezone.utc)
        days_since = (now - updated_at).total_seconds() / 86400
        interval_days = REVIEW_INTERVAL_MIN_DAYS + accuracy * (REVIEW_INTERVAL_MAX_DAYS - REVIEW_INTERVAL_MIN_DAYS)
        overdue_days = days_since - interval_days
        if overdue_days < 0:
            continue
        queue.append({
            "sign": gloss,
            "accuracy": round(accuracy * 100),
            "days_since_practiced": round(days_since, 1),
            "overdue_days": round(overdue_days, 1),
        })

    queue.sort(key=lambda q: -q["overdue_days"])
    return queue[:MAX_REVIEW_QUEUE]


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
