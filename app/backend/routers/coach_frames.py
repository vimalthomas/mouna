import json
import logging
import re
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from auth.dependencies import get_current_user_ws
from db.session import SessionLocal
from services.attempts_service import record_attempt
from services.coaching_session import CoachingSession
from services.gemini_live_service import gemini_manager
from services.keypoint_service import keypoint_service
from services.model_service import model_service
from services.signs_service import signs_service

logger = logging.getLogger(__name__)

router = APIRouter()


FACE_POSE_INDICES = range(0, 11)  # MediaPipe Pose: nose, eyes, ears, mouth corners (0-10)


def _bbox_from_points(points: list[tuple[float, float]], margin: float) -> dict | None:
    valid = [p for p in points if p != (0.0, 0.0)]
    if not valid:
        return None
    xs, ys = [p[0] for p in valid], [p[1] for p in valid]
    return {
        "x_min": max(0.0, min(xs) - margin), "x_max": min(1.0, max(xs) + margin),
        "y_min": max(0.0, min(ys) - margin), "y_max": min(1.0, max(ys) + margin),
    }


def _face_bbox(kp: list[float]) -> dict | None:
    # Approximate from already-extracted pose landmarks (normalized [0,1]
    # image coords) -- no extra model call, just a different crop of data
    # the frame handler already computed for the keypoint buffer.
    points = [(kp[i * 3], kp[i * 3 + 1]) for i in FACE_POSE_INDICES]
    return _bbox_from_points(points, margin=0.08)


def _hand_bbox(kp: list[float]) -> dict | None:
    left = [(kp[99 + i * 3], kp[99 + i * 3 + 1]) for i in range(21)]
    right = [(kp[162 + i * 3], kp[162 + i * 3 + 1]) for i in range(21)]
    both = [p for p in left + right if p != (0.0, 0.0)]
    return _bbox_from_points(both, margin=0.04)


def _is_supported(sign: str) -> bool:
    # Computed fresh at call time, not at module-import time -- WLASL_100_LABELS
    # is populated inside model_service.load_model() during app startup, which
    # runs AFTER this module is imported, so a module-level snapshot here would
    # always see the pre-startup empty list (Python's `from x import y` copies
    # the reference once, it doesn't track later reassignment in x).
    return sign.lower() in {w.lower() for w in model_service.idx_to_label.values()}


@router.websocket("/ws/coach-frames")
async def coach_from_frames(websocket: WebSocket) -> None:
    with SessionLocal() as auth_db:
        user = await get_current_user_ws(websocket, auth_db)
    if user is None:
        await websocket.close(code=4401)
        return

    await websocket.accept()
    session_id = str(uuid.uuid4())
    coaching = CoachingSession(session_id=session_id, user_id=user.id)
    gemini_session = None
    keypoint_buffer: list[list[float]] = []
    face_buffer: list[dict | None] = []

    logger.info("Coach-frames WebSocket connected: %s (user %s)", session_id, user.id)

    try:
        gemini_session = await gemini_manager.create_session(session_id)

        await websocket.send_json({
            "type": "session_start",
            "session_id": session_id,
            "message": "Coaching session started! Select a sign to practice.",
        })

        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)
            msg_type = data.get("type", "")

            if msg_type == "start_session":
                coaching.target_sign = data.get("target_sign", "")
                supported = not coaching.target_sign or _is_supported(coaching.target_sign)
                message = (
                    f"Great! Show me your '{coaching.target_sign}' sign!"
                    if supported
                    else f"Heads up: '{coaching.target_sign}' isn't in the model's practice vocabulary yet, "
                    "so attempts won't be scored accurately."
                )
                await websocket.send_json({
                    "type": "session_start",
                    "session_id": session_id,
                    "message": message,
                    "supported": supported,
                })

            elif msg_type == "frame":
                b64 = data.get("frame", "")
                if not b64:
                    continue

                kp = keypoint_service.extract_from_base64(b64)
                keypoint_buffer.append(kp)
                # Always extracted alongside pose/hands -- simpler than a
                # separate "which step is active" protocol message, at the
                # cost of one extra model pass per frame. Revisit if that
                # cost matters once this is under real load.
                face_buffer.append(keypoint_service.extract_face_from_base64(b64))

                await websocket.send_json({
                    "type": "keypoint_status",
                    "frame_count": len(keypoint_buffer),
                    "pose_detected": any(v != 0 for v in kp[:99]),
                    "left_hand_detected": any(v != 0 for v in kp[99:162]),
                    "right_hand_detected": any(v != 0 for v in kp[162:225]),
                    "face_detected": face_buffer[-1] is not None,
                    "face_bbox": _face_bbox(kp),
                    "hand_bbox": _hand_bbox(kp),
                })

            elif msg_type == "check_step":
                target_sign = data.get("target_sign", coaching.target_sign)
                step = data.get("step", "")

                sign_info = signs_service.signs_by_word.get(target_sign.lower())
                if sign_info is None:
                    await websocket.send_json({"type": "error", "message": f"Unknown sign '{target_sign}'."})
                    continue

                if step == "facial":
                    if not sign_info.nms_required:
                        await websocket.send_json({
                            "type": "step_result", "step": "facial",
                            "applicable": False,
                            "message": f"'{target_sign}' has no non-manual marker requirement -- nothing to check.",
                        })
                    elif len(face_buffer) < 5:
                        await websocket.send_json({
                            "type": "error", "message": f"Need more frames ({len(face_buffer)}/5 minimum)",
                        })
                        continue
                    else:
                        result = model_service.check_facial_step(face_buffer, target_sign, sign_info.nms_type)
                        await websocket.send_json({
                            "type": "step_result", "step": "facial", "applicable": True,
                            **result,
                            "notes": [sign_info.nms_notes] if sign_info.nms_notes else [],
                        })
                    face_buffer = []

                elif step in ("handshape", "location", "orientation", "hand_count"):
                    if step not in sign_info.manual_checks:
                        await websocket.send_json({
                            "type": "step_result", "step": step, "applicable": False,
                            "message": f"'{step}' is not a required check for '{target_sign}'.",
                        })
                    elif len(keypoint_buffer) < 5:
                        await websocket.send_json({
                            "type": "error", "message": f"Need more frames ({len(keypoint_buffer)}/5 minimum)",
                        })
                        continue
                    else:
                        score = model_service.score_against_target(keypoint_buffer, target_sign)
                        if step == "hand_count":
                            step_result = {"match": score["hand_count_match"]}
                        else:
                            step_result = (score.get("parameters") or {}).get(step, {"target": None, "yours": None, "match": None})
                        await websocket.send_json({
                            "type": "step_result", "step": step, "applicable": True,
                            **step_result,
                        })
                    keypoint_buffer = []

                else:
                    await websocket.send_json({"type": "error", "message": f"Unknown step '{step}'."})

            elif msg_type == "predict":
                target_sign = data.get("target_sign", coaching.target_sign)
                coaching.target_sign = target_sign

                if len(keypoint_buffer) < 5:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Need more frames ({len(keypoint_buffer)}/5 minimum)",
                    })
                    continue

                # Teach/correct scoring: compare against the KNOWN target's
                # reference (embedding similarity + hand-count), not a 1-of-100
                # guess. See src/mouna/coach/compare.py for the full reasoning.
                score = model_service.score_against_target(keypoint_buffer, target_sign)

                # Skeleton comparison needs the raw attempt buffer -- compute it
                # BEFORE clearing, not after (this used to discard the buffer
                # immediately, leaving nothing to render a comparison from).
                attempt_skeletons = model_service.get_attempt_skeletons(keypoint_buffer)
                reference_skeleton = model_service.get_reference_skeleton(target_sign)
                divergent_snapshots = model_service.get_attempt_snapshots(keypoint_buffer, score.get("divergent_frames") or {})
                keypoint_buffer = []
                face_buffer = []

                if score["embedding_similarity"] is None and not score["notes"]:
                    await websocket.send_json({
                        "type": "retry",
                        "message": "Didn't get a clear read on that — make sure your hands are "
                        "visible and try the sign again.",
                    })
                    continue

                prediction_label = score["coarse_prediction"] or "unknown"
                prediction_confidence = score["coarse_confidence"] or 0.0
                is_correct = score["overall_verdict"] == "good"
                coaching.record_attempt(prediction_label, is_correct)

                with SessionLocal() as attempt_db:
                    _, sign_progress, newly_completed = record_attempt(
                        attempt_db,
                        user_id=coaching.user_id,
                        gloss=target_sign,
                        correct=is_correct,
                        confidence=prediction_confidence,
                        session_id=session_id,
                    )

                try:
                    response_text = await gemini_session.send_context(
                        target_sign=target_sign,
                        predicted_sign=prediction_label,
                        confidence=prediction_confidence,
                        is_correct=is_correct,
                        attempt_number=coaching.attempts,
                    )
                except Exception:
                    logger.warning("Gemini unavailable, using fallback feedback")
                    response_text = ""

                feedback_data = _parse_gemini_response(response_text)
                if not feedback_data.get("feedback"):
                    # Fallback feedback built directly from score_attempt()'s
                    # verdict/notes (hand-count mismatch, embedding similarity
                    # tier) instead of a generic message -- keeps the signal
                    # from the teach/correct engine even without Gemini.
                    if is_correct:
                        feedback_data["feedback"] = f"Good job with '{target_sign}'!"
                    elif score["overall_verdict"] == "needs_work":
                        feedback_data["feedback"] = f"Close, but not quite '{target_sign}' yet — keep practicing."
                    else:
                        feedback_data["feedback"] = f"That didn't look like '{target_sign}' — try again."
                if not feedback_data.get("tips"):
                    feedback_data["tips"] = score["notes"]

                await websocket.send_json({
                    "type": "coaching",
                    "prediction": prediction_label,
                    "confidence": prediction_confidence,
                    "correct": is_correct,
                    "verdict": score["overall_verdict"],
                    "embedding_similarity": score["embedding_similarity"],
                    "hand_count_match": score["hand_count_match"],
                    "parameters": score["parameters"],
                    "attempt_skeleton_raw": attempt_skeletons["raw"],
                    "attempt_skeleton_canonical": attempt_skeletons["canonical"],
                    "reference_skeleton": reference_skeleton,
                    "divergent_snapshots": divergent_snapshots,
                    "feedback": feedback_data.get("feedback", ""),
                    "tips": feedback_data.get("tips", []),
                    "attempt_number": coaching.attempts,
                    "session_accuracy": coaching.accuracy,
                    "sign_completed": newly_completed,
                    "sign_progress": {
                        "total_attempts": sign_progress.total_attempts,
                        "correct_attempts": sign_progress.correct_attempts,
                        "accuracy": round(
                            (sign_progress.correct_attempts / sign_progress.total_attempts) * 100
                        ) if sign_progress.total_attempts else 0,
                        "completed": sign_progress.completed,
                    },
                })

            elif msg_type == "chat":
                user_text = data.get("text", "")
                if not user_text.strip():
                    continue

                try:
                    reply = await gemini_session.send_chat(
                        user_text=user_text,
                        target_sign=coaching.target_sign,
                        attempts=coaching.attempts,
                        accuracy=coaching.accuracy,
                    )
                except Exception:
                    reply = ""
                if not reply:
                    reply = f"I'm having trouble connecting right now. For '{coaching.target_sign}', focus on clear hand shapes and movement."

                await websocket.send_json({
                    "type": "chat_reply",
                    "message": reply,
                })

            elif msg_type == "change_target":
                coaching.target_sign = data.get("target_sign", "")
                keypoint_buffer = []
                face_buffer = []
                supported = not coaching.target_sign or _is_supported(coaching.target_sign)
                message = (
                    f"Switched! Now show me '{coaching.target_sign}'."
                    if supported
                    else f"Heads up: '{coaching.target_sign}' isn't in the model's practice vocabulary yet, "
                    "so attempts won't be scored accurately."
                )
                await websocket.send_json({
                    "type": "session_start",
                    "message": message,
                    "supported": supported,
                })

            elif msg_type == "end_session":
                await websocket.send_json({
                    "type": "session_end",
                    "total_attempts": coaching.attempts,
                    "accuracy": coaching.accuracy,
                    "message": "Great practice session! Keep it up!",
                })
                break

    except WebSocketDisconnect:
        logger.info("Coach-frames WebSocket disconnected: %s", session_id)
    except Exception:
        logger.exception("Coach-frames WebSocket error: %s", session_id)
        await websocket.close(code=1011)
    finally:
        await gemini_manager.close_session(session_id)


def _parse_gemini_response(response_text: str) -> dict:
    text = response_text.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {"feedback": response_text.strip(), "tips": []}
