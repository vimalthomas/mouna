from datetime import datetime, timedelta, timezone

from sqlalchemy import Integer, cast, func
from sqlalchemy.orm import Session

from db.models import Attempt, Sign, SignProgress
from services.attempts_service import LEARNED_MIN_CORRECT, LEARNED_WINDOW, is_learned

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
    # Derived from recent attempts (see attempts_service.is_learned), so it
    # can't go stale and older rows recorded under the previous rule count.
    sign_ids = [sid for (sid,) in db.query(SignProgress.sign_id).filter(SignProgress.user_id == user_id).all()]
    signs_learned = sum(1 for sid in sign_ids if is_learned(db, user_id, sid))

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
    """Consecutive days (server-local calendar) with at least one attempt,
    ending today -- or yesterday, so the streak survives until you practice
    today. Dates are derived in Python: SQLite's date() returns TEXT, which
    never compared equal to datetime.date and made the streak always 0."""
    stamps = [ts for (ts,) in db.query(Attempt.created_at).filter(Attempt.user_id == user_id).all() if ts]
    if not stamps:
        return 0
    local_tz = datetime.now().astimezone().tzinfo

    def local_day(ts: datetime):
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)  # stored as UTC
        return ts.astimezone(local_tz).date()

    practiced_days = {local_day(ts) for ts in stamps}
    cursor = datetime.now(local_tz).date()
    if cursor not in practiced_days:
        cursor -= timedelta(days=1)
        if cursor not in practiced_days:
            return 0
    streak = 0
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
        recent = [
            bool(c) for (c,) in db.query(Attempt.correct)
            .filter(Attempt.user_id == user_id, Attempt.sign_id == progress.sign_id)
            .order_by(Attempt.created_at.desc(), Attempt.id.desc())
            .limit(LEARNED_WINDOW)
            .all()
        ][::-1]  # oldest -> newest, for display
        signs.append({
            "sign": gloss,
            "total_attempts": progress.total_attempts,
            "correct_attempts": progress.correct_attempts,
            "accuracy": accuracy,
            # Same rule as signs_learned (see attempts_service.is_learned).
            "completed": sum(recent) >= LEARNED_MIN_CORRECT,
            "completed_at": progress.completed_at.isoformat() if progress.completed_at else None,
            "recent": recent,
        })
    return signs
