import json
import logging
from pathlib import Path

from config import settings
from models.schemas import SignInfo, VideoResponse

logger = logging.getLogger(__name__)


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

        for entry in data:
            gloss = entry["gloss"].lower()
            available = []
            for inst in entry.get("instances", []):
                vid = inst.get("video_id", "")
                if vid in self.available_videos:
                    available.append({
                        "video_id": vid,
                        "frame_start": inst.get("frame_start"),
                        "frame_end": inst.get("frame_end"),
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

        entry = entries[0]
        video_id = entry["video_id"]
        video_url = f"/static/videos/{video_id}.mp4"

        return VideoResponse(
            word=word,
            video_url=video_url,
            frame_start=entry.get("frame_start"),
            frame_end=entry.get("frame_end"),
        )


signs_service = SignsService()
