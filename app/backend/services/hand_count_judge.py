"""Hand count, checked by an LLM that sees the measurements and key frames.

The judgement benchmark (scripts/llm_evidence_judge.py, arm "vision") found
this beats our own hand-count check on every measure (catches 76% vs 72%,
false alarms on correct signing 5% vs 23%), so it is used for hand count only.
It runs after the local result has been sent and never blocks it; if there
is no Gemini key, no reference rows for the sign, or the model is slow, the
Score tab keeps the local check (marked "estimate").
"""
import asyncio
import base64
import json
import logging
import os

import cv2
import numpy as np

from config import settings
from services import model_service as _ms  # noqa: F401 -- puts src/ on sys.path before the mouna import
from mouna.coach import llm_judge  # noqa: E402

logger = logging.getLogger(__name__)

PER_MODEL_TIMEOUT_S = 40     # image + long prompt: Flash took up to ~30 s in the benchmark run
TOTAL_BUDGET_S = 60
MAX_FRAMES = 600             # keep at most this many recorded frames per attempt (~20 s at 30 fps)
_client = None


def enabled() -> bool:
    return bool(settings.gemini_api_key) and os.environ.get("LLM_HAND_COUNT", "1") != "0"


def available_for(sign: str) -> bool:
    return enabled() and llm_judge.has_references(sign)


def _decode(b64: str) -> np.ndarray | None:
    try:
        if "," in b64[:100]:
            b64 = b64.split(",", 1)[1]
        arr = np.frombuffer(base64.b64decode(b64), np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:  # noqa: BLE001
        return None


def build_image(sign: str, frames_b64: list[str], keypoints: list[list[float]]) -> bytes | None:
    """ROW 1-2: fluent signers (pre-rendered); ROW 3: three moments of the attempt."""
    refs = llm_judge.reference_rows(sign)
    n = min(len(frames_b64), len(keypoints))
    if refs is None or n < 5:
        return None
    kp = np.asarray(keypoints[:n], dtype=np.float32).reshape(n, 75, 3)
    idx, box = llm_judge.pick_frames(kp, n)
    row = llm_judge.crop_row([_decode(frames_b64[i]) for i in idx], box)
    return llm_judge.stack_rows(refs + [row]) if row is not None else None


async def _ask(image: bytes, prompt: str) -> dict | None:
    global _client
    from google import genai
    from google.genai import types
    if _client is None:
        _client = genai.Client(api_key=settings.gemini_api_key,
                               http_options=types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=1)))
    loop = asyncio.get_running_loop()
    deadline = loop.time() + TOTAL_BUDGET_S
    # the models the benchmark used: Flash first, Lite if Flash is busy
    for model in ("gemini-flash-latest", "gemini-flash-lite-latest"):
        remaining = deadline - loop.time()
        if remaining <= 1:
            break
        try:
            r = await asyncio.wait_for(_client.aio.models.generate_content(
                model=model,
                contents=[types.Part.from_bytes(data=image, mime_type="image/jpeg"), prompt],
                config=types.GenerateContentConfig(response_mime_type="application/json", max_output_tokens=1500),
            ), timeout=min(remaining, PER_MODEL_TIMEOUT_S))
            return {**llm_judge.parse_answer(json.loads(r.text)), "model": model}
        except Exception as e:  # noqa: BLE001 -- busy / timeout / bad JSON: next model, then give up
            logger.warning("hand-count judge: %s unavailable (%s: %s)", model, type(e).__name__, str(e)[:120])
    return None


def _note(ans: dict) -> str:
    """The model's 'how' covers every part; keep it only if it talks about the hands."""
    note = (ans.get("note") or "").strip()
    return note if any(w in note.lower() for w in ("one hand", "two hand", "both hand", "single hand", "second hand", "other hand")) else ""


async def judge(sign: str, frames_b64: list[str], keypoints: list[list[float]], score: dict) -> dict:
    """{"status": "done", "verdict": same|different|unsure, "note", "model"} or {"status": "unavailable"}."""
    try:
        image = await asyncio.to_thread(build_image, sign, frames_b64, keypoints)
        if image is None:
            return {"status": "unavailable"}
        from mouna.coach.param_scores import _refs
        evidence = llm_judge.evidence_text(score.get("parameter_scores") or {}, score.get("embedding_similarity"),
                                           score.get("hand_count_match"), _refs()["calibration"])
        ans = await _ask(image, llm_judge.prompt(sign, evidence))
        if ans is None:
            return {"status": "unavailable"}
        verdict = ans["hands"] if ans["hands"] in ("same", "different") else "unsure"
        return {"status": "done", "verdict": verdict, "note": _note(ans) if verdict == "different" else "", "model": ans["model"]}
    except Exception:  # noqa: BLE001 -- a bonus check; never break the session
        logger.exception("hand-count judge failed")
        return {"status": "unavailable"}
