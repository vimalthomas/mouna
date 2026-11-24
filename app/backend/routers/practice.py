"""v5 Practice tab: stories from the learner's learned signs, the avatar signing
them, sign-by-sign grading, and the Words / Sentences history for the dashboard.

  GET  /practice/pool              learned signs usable in stories (+ practising ones), unlock state
  POST /practice/story             a new checked story from the learned signs (saved)
  GET  /practice/sentence?signs=   one stitched 3D clip of the signs in order
  POST /practice/score             grade one recorded sign (same scoring as Learn; doesn't change Learn progress)
  POST /practice/sentence-attempts save a graded sentence
  GET  /practice/summary           Words (signed in sentences) and Sentences history
"""
import asyncio
import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from auth.dependencies import GUEST_EMAIL, get_current_user
from config import settings
from db.models import PracticeStory, SentenceAttempt, Sign, SignProgress, User
from db.session import get_db
from services import coach_writer, example_videos, story_service
from services.keypoint_service import keypoint_service
from services.model_service import model_service
from services.signs_service import signs_service
from mouna.skin import sentence as sentence_skin  # noqa: E402 -- src/ is on the path after model_service

router = APIRouter(prefix="/practice", tags=["practice"])

MIN_SIGNS = 5          # learned signs needed before stories unlock


def _story_signs() -> set[str]:
    """Signs a story can use: a practice sign the avatar has 3D data for."""
    return {w for w in sentence_skin.available() if example_videos.practice_ready(w)}


def _progress(db: Session, user_id: int) -> dict[str, SignProgress]:
    rows = (db.query(Sign.gloss, SignProgress).join(SignProgress, SignProgress.sign_id == Sign.id)
            .filter(SignProgress.user_id == user_id).all())
    return {g.lower(): p for g, p in rows}


def _is_guest(user: User) -> bool:
    """The public demo's shared guest: its progress is everyone's, so stories use every practice sign."""
    return bool(settings.allow_guest) and user.email == GUEST_EMAIL


def _sign_card(w: str, p: SignProgress | None) -> dict:
    info = signs_service.signs_by_word.get(w)
    ex = example_videos.get(w) or {}
    return {"word": w, "description": info.description if info else "",
            "video_url": f"/static/videos/{ex['video_id']}.mp4" if ex.get("video_id") else None,
            "learned": bool(p and p.completed), "tries": p.total_attempts if p else 0}


@router.get("/pool")
def pool(user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    prog, usable = _progress(db, user.id), _story_signs()
    if _is_guest(user):
        return {"learned": [_sign_card(w, prog.get(w)) for w in sorted(usable)], "practising": [],
                "min_signs": MIN_SIGNS, "unlocked": True, "ai": story_service.enabled(), "demo": True}
    learned = sorted(w for w in usable if prog.get(w) and prog[w].completed)
    practising = sorted(w for w in usable if prog.get(w) and not prog[w].completed and prog[w].total_attempts)
    return {"learned": [_sign_card(w, prog.get(w)) for w in learned],
            "practising": [_sign_card(w, prog.get(w)) for w in practising],
            "min_signs": MIN_SIGNS, "unlocked": len(learned) >= MIN_SIGNS, "ai": story_service.enabled()}


class StoryRequest(BaseModel):
    stretch: bool = False      # include one sign still being practised


@router.post("/story")
async def new_story(req: StoryRequest, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    if not story_service.enabled():
        raise HTTPException(503, "Stories need an AI key (GEMINI_API_KEY) on the server.")
    prog, usable = _progress(db, user.id), _story_signs()
    learned = list(usable) if _is_guest(user) else [w for w in usable if prog.get(w) and prog[w].completed]
    if len(learned) < MIN_SIGNS:
        raise HTTPException(400, f"Learn {MIN_SIGNS - len(learned)} more sign(s) to unlock stories.")
    practising = [w for w in usable if prog.get(w) and not prog[w].completed and prog[w].total_attempts]
    stretch = None
    if req.stretch and practising:
        import random
        stretch = random.choice(practising)
    recent = db.query(PracticeStory).filter(PracticeStory.user_id == user.id).order_by(PracticeStory.created_at.desc()).limit(6).all()
    avoid = [r.title + ": " + " / ".join(" ".join(s["gloss"]) for s in json.loads(r.content)["sentences"]) for r in recent]
    try:
        story = await story_service.write_story(learned, stretch, avoid)
    except RuntimeError as e:
        raise HTTPException(502, f"Couldn't write a story right now ({e}). Try again.")
    row = PracticeStory(user_id=user.id, title=story["title"],
                        content=json.dumps({"sentences": story["sentences"], "structure": story["structure"]}))
    db.add(row)
    db.commit()
    words = [g.lower() for s in story["sentences"] for g in s["gloss"]]
    return {"id": row.id, **story, "stretch": stretch,
            "signs": [_sign_card(w, prog.get(w)) for w in words]}


@router.get("/sentence")
def sentence(signs: str, user: User = Depends(get_current_user)) -> JSONResponse:
    words = [w.strip().lower() for w in signs.split(",") if w.strip()][:12]
    return JSONResponse(sentence_skin.sentence(words))


class ScoreRequest(BaseModel):
    sign: str
    frames: list[str]          # base64 JPEG, 640x480, ~10 fps


@router.post("/score")
async def score(req: ScoreRequest, user: User = Depends(get_current_user)) -> dict:
    sign = req.sign.lower()
    if len(req.frames) < 5:
        raise HTTPException(400, "Need at least 5 frames")
    if len(req.frames) > 80:
        raise HTTPException(400, "Too many frames")

    def run():
        kps = [keypoint_service.extract_from_base64(b.split(",", 1)[-1]) for b in req.frames]
        size = getattr(keypoint_service, "last_frame_size", None) or (640, 480)
        return kps, model_service.score_against_target(kps, sign, size)
    kps, sc = await asyncio.to_thread(run)
    info = signs_service.signs_by_word.get(sign)
    ps = sc.get("parameter_scores") or {}
    return {"sign": sign, "verdict": sc.get("overall_verdict"), "similarity": sc.get("embedding_similarity"),
            "graded": sc.get("embedding_similarity") is not None,
            "hands_seen": sum(any(v != 0 for v in k[99:225]) for k in kps), "frames": len(kps),
            "parts": {k: {"label": v.get("label"), "level": v.get("level"), "score": v.get("score"), "reliable": v.get("reliable")}
                      for k, v in ps.items()},
            "feedback": coach_writer.local_feedback(sign, info.description if info else "", sc)}


class SignResult(BaseModel):
    sign: str
    verdict: str | None


class SentenceAttemptIn(BaseModel):
    story_id: int | None = None
    english: str
    results: list[SignResult]


@router.post("/sentence-attempts")
def save_attempt(a: SentenceAttemptIn, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    if not a.results:
        raise HTTPException(400, "No results")
    if a.story_id is not None and not db.query(PracticeStory).filter_by(id=a.story_id, user_id=user.id).first():
        a.story_id = None
    good = sum(r.verdict == "good" for r in a.results)
    row = SentenceAttempt(user_id=user.id, story_id=a.story_id, english=a.english[:2000],
                          gloss=" ".join(r.sign.upper() for r in a.results)[:500],
                          results=json.dumps([r.model_dump() for r in a.results]),
                          signs_good=good, signs_total=len(a.results), all_good=good == len(a.results))
    db.add(row)
    db.commit()
    return {"id": row.id, "signs_good": good, "signs_total": len(a.results), "all_good": row.all_good}


@router.get("/summary")
def summary(user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    rows = db.query(SentenceAttempt).filter(SentenceAttempt.user_id == user.id).order_by(SentenceAttempt.created_at.desc()).all()
    words: dict[str, dict] = {}
    for r in rows:
        for x in json.loads(r.results):
            w = words.setdefault(x["sign"].lower(), {"in_sentences": 0, "good": 0, "last": None})
            w["in_sentences"] += 1
            w["good"] += x.get("verdict") == "good"
            w["last"] = w["last"] or r.created_at.isoformat()
    return {"sentences": [{"id": r.id, "english": r.english, "gloss": r.gloss, "signs_good": r.signs_good,
                           "signs_total": r.signs_total, "all_good": r.all_good, "at": r.created_at.isoformat()} for r in rows[:50]],
            "sentences_total": len(rows), "sentences_all_good": sum(r.all_good for r in rows),
            "words": [{"word": w, **v} for w, v in sorted(words.items())]}
