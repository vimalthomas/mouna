import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Response

from models.schemas import SignListResponse, VideoResponse
from services import example_videos
from services.model_service import model_service
from services.signs_service import signs_service

router = APIRouter()


@router.get("/signs/{word}/reference-skeleton")
async def get_reference_skeleton(word: str) -> list:
    """The target's skeleton frames (incl. hand-local 3D) for the prep
    screen's 3D "how the hand looks" view, before any attempt exists."""
    frames = model_service.get_reference_skeleton(word.lower())
    if frames is None:
        raise HTTPException(status_code=404, detail=f"No reference data for '{word}' yet.")
    return frames


_SKIN = Path(__file__).resolve().parents[3] / "data" / "references" / "skin"


@router.get("/signs/{word}/skin")
async def get_skin(word: str) -> Response:
    """3D skin data for the Learn screen (v3): per frame of the sign's example
    video, both hands (video position + 3D finger shape), upper body and the
    face mesh, from scripts/build_sign_skin.py. Stored gzipped; sent as-is."""
    path = _SKIN / f"{word.lower().replace('/', '')}.json.gz"
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"No 3D skin for '{word}' yet.")
    return Response(path.read_bytes(), media_type="application/json",
                    headers={"Content-Encoding": "gzip", "Cache-Control": "max-age=3600"})


@router.get("/signs/{word}/variations")
async def get_variations(word: str) -> list[dict]:
    """Other on-disk recordings of this sign (different signers; some use a
    different version of the sign). Informational -- grading uses the main
    example's reference."""
    key = word.lower().replace("_", " ")
    main = signs_service.get_video(word)
    out = []
    # The sign's other form (one hand vs two, see scripts/assign_variants.py)
    # comes first, labelled, so learners know it's also correct ASL.
    chosen = example_videos.get(word) or {}
    for v in chosen.get("variations", []):
        out.append({"video_url": f"/static/videos/{v['video_id']}.mp4", "frame_start": v.get("frame_start"),
                    "frame_end": v.get("frame_end"), "fps": v.get("fps"),
                    "label": "One-handed version" if v["n_hands"] == 1 else "Two-handed version"})
    for e in signs_service.video_index.get(key, []):
        url = f"/static/videos/{e['video_id']}.mp4"
        if (main and url == main.video_url) or any(o["video_url"] == url for o in out):
            continue
        out.append({"video_url": url, "frame_start": e.get("frame_start"), "frame_end": e.get("frame_end"), "fps": e.get("fps")})
        if len(out) >= 3:
            break
    return out


@router.get("/signs/{word}/reference-buckets")
async def get_reference_buckets(word: str) -> dict:
    buckets = model_service.get_reference_buckets(word.lower())
    if buckets is None:
        raise HTTPException(status_code=404, detail=f"No reference data for '{word}' yet.")
    chosen = example_videos.get(word)
    if chosen and chosen.get("hand_bbox"):
        # Crop/still timing from the example video actually shown, not from
        # the (different) reference recording.
        buckets = {**buckets, "hand_bbox": chosen["hand_bbox"], "hands_up_seconds": chosen.get("hands_up_seconds"),
                   "hand_paths": chosen.get("hand_paths") or []}
    return buckets


@router.get("/signs", response_model=SignListResponse)
async def get_signs() -> SignListResponse:
    return SignListResponse(signs=signs_service.get_all_signs())


_CATALOG = Path(__file__).resolve().parents[3] / "data" / "references" / "sign_catalog.json"


@router.get("/sign-catalog")
async def get_sign_catalog() -> dict:
    """ASL-CDI category, ASL-LEX frequency and "essential" flag per sign
    (built by scripts/build_sign_catalog.py from published sources)."""
    try:
        return json.loads(_CATALOG.read_text())
    except FileNotFoundError:
        return {}


@router.get("/signs-with-reference")
async def get_signs_with_reference() -> list[str]:
    """The practice set: signs the coach can fully support -- a reviewed
    reference recording with a readable CP breakdown, plus an example video."""
    words = [
        w for w in model_service.references
        if model_service.get_reference_buckets(w) is not None and signs_service.get_video(w) is not None
    ]
    # Once example videos have been checked against their references, keep
    # only signs whose shown example itself passes grading.
    if example_videos.all_choices():
        words = [w for w in words if example_videos.practice_ready(w)]
    return sorted(words)


@router.get("/signs-with-video")
async def get_signs_with_video() -> list[str]:
    return signs_service.get_signs_with_video()


@router.get("/video/{word}", response_model=VideoResponse)
async def get_video(word: str) -> VideoResponse:
    result = signs_service.get_video(word)
    if result is None:
        raise HTTPException(status_code=404, detail=f"No video found for '{word}'")
    return result
