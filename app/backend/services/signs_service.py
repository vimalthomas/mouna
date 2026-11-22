import json
import logging
from pathlib import Path

from config import settings
from models.schemas import SignInfo, VideoResponse

logger = logging.getLogger(__name__)

# WLASL's own per-instance frame_start/frame_end mark which sub-range of a
# raw source video is the actual isolated sign. frame_end == -1 means "the
# whole file from frame_start is the sign" (already a clean clip) -- but
# some entries are a tiny window inside a much longer, unrelated recording
# (e.g. "what" video_id 62944: frames 1748-1780 inside a 70+ second clip),
# and get_video() used to just take entries[0] with no regard for which
# kind it was, then the frontend played the raw file start-to-finish with
# no trimming -- showing a long stretch of irrelevant footage before ever
# reaching the sign. A real ASL sign is ~1-3 seconds; prefer any instance
# WLASL itself already marks as a short, known-duration sub-clip.
MAX_CLEAN_SIGN_SECONDS = 6.0


def _instance_duration_s(entry: dict) -> float | None:
    """Prefer WLASL's own sub-range annotation when present (frame_end not
    -1); fall back to the actual file duration probed once at startup (see
    _probe_duration) for frame_end == -1 entries, since "whole file is the
    sign" doesn't mean the whole file is SHORT -- one real case found in
    this deployment's downloaded files ("thanksgiving") is a genuine 8.1s
    frame_end=-1 file with no shorter sibling instance available."""
    frame_end = entry.get("frame_end")
    fps = entry.get("fps")
    if frame_end not in (None, -1) and fps:
        return (frame_end - entry.get("frame_start", 0)) / fps
    return entry.get("probed_duration_s")


def _pick_cleanest_instance(entries: list[dict]) -> dict:
    known = [(e, d) for e in entries if (d := _instance_duration_s(e)) is not None]
    short = [(e, d) for e, d in known if d <= MAX_CLEAN_SIGN_SECONDS]
    if short:
        short.sort(key=lambda pair: pair[1])
        return short[0][0]
    if known:
        # Nothing qualifies as genuinely short -- still pick the shortest
        # available rather than an arbitrary (possibly much longer) one.
        known.sort(key=lambda pair: pair[1])
        return known[0][0]
    return entries[0]


def _probe_duration(video_path: Path) -> float | None:
    """Real file duration via OpenCV metadata (reads the header, not every
    frame -- cheap). Called once per available video at startup
    (_load_video_index), not per-request, so get_video() stays O(1)."""
    try:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        if fps and frame_count:
            return frame_count / fps
    except Exception:
        logger.exception("Failed to probe duration for %s", video_path)
    return None


class SignsService:
    def __init__(self) -> None:
        self.signs: list[SignInfo] = []
        self.signs_by_word: dict[str, SignInfo] = {}
        self.video_index: dict[str, list[dict]] = {}
        self.available_videos: set[str] = set()

    def load(self) -> None:
        self._scan_video_files()
        self._load_signs()
        self._load_video_index()

    def _scan_video_files(self) -> None:
        videos_dir = Path(settings.videos_dir)
        if not videos_dir.exists():
            logger.warning("Videos directory not found at %s", videos_dir)
            return

        for f in videos_dir.glob("*.mp4"):
            self.available_videos.add(f.stem)

        logger.info("Found %d video files on disk", len(self.available_videos))

    def _load_signs(self) -> None:
        signs_path = Path(settings.signs_json_path)
        if not signs_path.exists():
            logger.warning("signs.json not found at %s", signs_path)
            return

        with open(signs_path, encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            sign = SignInfo(
                word=item["word"],
                category=item.get("category", "general"),
                difficulty=item.get("difficulty", "beginner"),
                description=item.get("description", ""),
                manual_checks=item.get("manual_checks", []),
                nms_required=item.get("nms_required", False),
                nms_type=item.get("nms_type"),
                nms_notes=item.get("nms_notes"),
            )
            self.signs.append(sign)
            self.signs_by_word[sign.word.lower()] = sign

        logger.info("Loaded %d signs", len(self.signs))

    def _load_video_index(self) -> None:
        wlasl_path = Path(settings.wlasl_json_path)
        if not wlasl_path.exists():
            logger.warning("WLASL JSON not found at %s", wlasl_path)
            return

        with open(wlasl_path, encoding="utf-8") as f:
            data = json.load(f)

        videos_dir = Path(settings.videos_dir)
        for entry in data:
            gloss = entry["gloss"].lower()
            available = []
            for inst in entry.get("instances", []):
                vid = inst.get("video_id", "")
                if vid in self.available_videos:
                    frame_end = inst.get("frame_end")
                    probed = None
                    # Only probe actual video files for words we actually
                    # serve (171 WLASL glosses have files on disk, but only
                    # the active 100-word vocabulary needs this) -- and only
                    # when WLASL's own frame_end doesn't already tell us the
                    # duration, so startup stays fast.
                    if (frame_end is None or frame_end == -1) and gloss in self.signs_by_word:
                        probed = _probe_duration(videos_dir / f"{vid}.mp4")
                    available.append({
                        "video_id": vid,
                        "frame_start": inst.get("frame_start"),
                        "frame_end": frame_end,
                        "fps": inst.get("fps"),
                        "probed_duration_s": probed,
                    })
            if available:
                self.video_index[gloss] = available

        logger.info(
            "Indexed %d glosses with available videos", len(self.video_index)
        )

    def get_all_signs(self) -> list[SignInfo]:
        return self.signs

    def get_signs_with_video(self) -> list[str]:
        return sorted(self.video_index.keys())

    def get_video(self, word: str) -> VideoResponse | None:
        key = word.lower().replace("_", " ")
        entries = self.video_index.get(key)
        if not entries:
            return None

        entry = _pick_cleanest_instance(entries)
        video_id = entry["video_id"]
        video_url = f"/static/videos/{video_id}.mp4"

        return VideoResponse(
            word=word,
            video_url=video_url,
            frame_start=entry.get("frame_start"),
            frame_end=entry.get("frame_end"),
            fps=entry.get("fps"),
        )


signs_service = SignsService()
