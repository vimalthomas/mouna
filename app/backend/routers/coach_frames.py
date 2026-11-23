import asyncio
import json
import logging
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from auth.dependencies import get_current_user_ws
from db.session import SessionLocal
from services import coach_writer
from services.attempts_service import record_attempt
from services.coaching_session import CoachingSession
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


async def _send_ai_wording(websocket: WebSocket, target: str, description: str, score: dict, attempt_number: int) -> None:
    """Follow-up to a "coaching" message: the AI's wording of the same,
    already-shown result. Silent if the AI fails or the learner has left."""
    try:
        upgraded = await coach_writer.ai_feedback(target, description, score)
        if upgraded:
            text, source = upgraded
            await websocket.send_json({"type": "coaching_update", "attempt_number": attempt_number,
                                       "feedback": text, "feedback_source": source})
    except Exception as e:  # noqa: BLE001 -- socket closed, etc.; the local message already stands
        logger.info("AI wording not delivered (%s)", type(e).__name__)


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
    keypoint_buffer: list[list[float]] = []
    face_buffer: list[dict | None] = []
    # Chat context: the last coaching message shown, and recent turns.
    last_feedback: str | None = None
    chat_history: list[dict] = []

    logger.info("Coach-frames WebSocket connected: %s (user %s)", session_id, user.id)

    try:
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

            elif msg_type == "reset_buffer":
                # Sent when a graded recording starts, so the scored attempt
                # covers exactly the recorded clip -- not the prep screen or
                # countdown frames that otherwise accumulate here.
                keypoint_buffer = []
                face_buffer = []

            elif msg_type == "check_group":
                # Several manual CP checks on ONE window of frames (the learner-
                # facing prep groups, e.g. "hand shape & palm"), so a group
                # doesn't need one hold per CP. Facial stays on check_step.
                target_sign = data.get("target_sign", coaching.target_sign)
                steps = [st for st in data.get("steps", []) if st in ("handshape", "location", "orientation", "hand_count")]
                if len(keypoint_buffer) < 5:
                    await websocket.send_json({"type": "group_result", "steps": steps, "results": {}, "message": "Need more frames"})
                    continue
                score = model_service.score_against_target(keypoint_buffer, target_sign)
                params = score.get("parameters") or {}
                results = {}
                for st in steps:
                    if st == "hand_count":
                        results[st] = {"applicable": True, "match": score["hand_count_match"]}
                    else:
                        results[st] = {"applicable": True, **params.get(st, {"target": None, "yours": None, "match": None})}
                await websocket.send_json({"type": "group_result", "steps": steps, "results": results})
                keypoint_buffer = []
                face_buffer = []

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
                # face_buffer was already being collected every frame
                # alongside keypoint_buffer (see the "frame" handler above) --
                # it just wasn't used here before the face-mesh skeleton.
                attempt_skeletons = model_service.get_attempt_skeletons(keypoint_buffer, face_buffer)
                reference_skeleton = model_service.get_reference_skeleton(target_sign)
                divergent_frames = score.get("divergent_frames") or {}
                divergent_snapshots = model_service.get_attempt_snapshots(keypoint_buffer, divergent_frames, face_buffer)
                # When the attempt's own bucket triple matched a DIFFERENT
                # known sign (see compare.py's looks_like logic), fetch that
                # sign's reference skeleton too -- so "this looks like X"
                # comes with an actual X to look at, not just the word.
                looks_like = score.get("looks_like")
                looks_like_skeleton = model_service.get_reference_skeleton(looks_like) if looks_like else None
                charts = model_service.get_attempt_charts(keypoint_buffer, face_buffer, target_sign)
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
                # No reference recording for this sign -> nothing was compared.
                # Don't count it as a failed attempt in the learner's progress.
                graded = score["embedding_similarity"] is not None
                is_correct = graded and score["overall_verdict"] == "good"
                sign_progress, newly_completed = None, False
                if graded:
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

                # Decided locally first: the verdict and checklist come from
                # the app's own checks and go out immediately. Only then is
                # the finished status sent to the AI (if a key is set) to
                # reword the summary; that arrives as "coaching_update".
                sign_info = signs_service.signs_by_word.get(target_sign.lower())
                description = sign_info.description if sign_info else ""
                feedback_text, feedback_source = coach_writer.local_feedback(target_sign, description, score), "template"
                feedback_data = {"feedback": feedback_text, "tips": []}
                last_feedback = feedback_text
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
                    "attempt_mirrored": attempt_skeletons["mirrored"],
                    "reference_skeleton": reference_skeleton,
                    "divergent_snapshots": divergent_snapshots,
                    # Raw frame index per mismatched param, into
                    # attempt_skeleton_raw/canonical -- lets the frontend
                    # replay a short looping window around the moment of
                    # max divergence instead of only showing one frozen
                    # frame (P1's "replay it" request).
                    "divergent_frames": divergent_frames,
                    "looks_like": looks_like,
                    "looks_like_skeleton": looks_like_skeleton,
                    "feedback": feedback_data.get("feedback", ""),
                    "feedback_source": feedback_source,
                    "tips": feedback_data.get("tips", []),
                    "attempt_number": coaching.attempts,
                    "session_accuracy": coaching.accuracy,
                    "graded": graded,
                    "charts": charts,
                    "sign_completed": newly_completed,
                    "sign_progress": {
                        "total_attempts": sign_progress.total_attempts,
                        "correct_attempts": sign_progress.correct_attempts,
                        "accuracy": round(
                            (sign_progress.correct_attempts / sign_progress.total_attempts) * 100
                        ) if sign_progress.total_attempts else 0,
                        "completed": sign_progress.completed,
                    } if sign_progress else None,
                })
                if coach_writer.ai_enabled():
                    asyncio.create_task(_send_ai_wording(websocket, target_sign, description, score, coaching.attempts))

            elif msg_type == "chat":
                user_text = data.get("text", "")
                if not user_text.strip():
                    continue

                sign_info = signs_service.signs_by_word.get((coaching.target_sign or "").lower())
                reply, _source = await coach_writer.answer_chat(
                    user_text, coaching.target_sign, sign_info.description if sign_info else "",
                    last_feedback, chat_history,
                )
                chat_history += [{"role": "user", "content": user_text}, {"role": "assistant", "content": reply}]
                chat_history = chat_history[-12:]

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

