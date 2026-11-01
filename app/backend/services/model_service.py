"""Model serving for Mouna's backend -- PoseTransformer (v4canon), not TGCN.

Replaces the original TGCN-WLASL model with the statistically-validated
PoseTransformer from the core mouna ML pipeline (+4.50pp, p=0.020 vs. the
pre-canonicalization baseline -- see src/mouna/data/handedness.py). Two
capabilities are served:

  - predict(): open-set classification (same interface/shape as before, so
    anything calling model_service.predict() doesn't need to change) --
    useful for dictionary lookup / "what did I just sign" style use.
  - score_against_target(): the teach/correct coach scoring, via
    src/mouna/coach/compare.py's score_attempt() -- a closed-set comparison
    against a known target sign's reference, not a 100-way guess. This is
    what coach_frames.py's "predict" message now uses instead of the old
    crude `prediction in top3` check.
"""

import logging
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

from models.schemas import Prediction, PredictionResult

logger = logging.getLogger(__name__)

_BACKEND_DIR = Path(__file__).parent.parent
_PROJECT_ROOT = _BACKEND_DIR.parent.parent  # app/backend -> app -> project root
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from mouna.models.transformer import PoseTransformer  # noqa: E402
from mouna.data.handedness import canonicalize_handedness  # noqa: E402
from mouna.data.part_normalize import normalize_per_part  # noqa: E402
from mouna.coach.compare import score_attempt  # noqa: E402
from mouna.coach.facial import check_nms, from_live_frames  # noqa: E402
from mouna.coach.skeleton import extract_skeleton_sequence  # noqa: E402
from mouna.coach.parameters import (  # noqa: E402
    classify_handshape_bucket,
    classify_location_bucket,
    classify_orientation_bucket,
    extract_dominant_hand_signal,
)

DEFAULT_CHECKPOINT = _PROJECT_ROOT / "data" / "best_model_v4canon.pth"
DEFAULT_REFERENCES = _PROJECT_ROOT / "data" / "references" / "sign_references.pkl"
MAX_SEQ_LEN = 150
CONF_THRESH = 0.25

# Populated from the checkpoint's idx_to_label once load_model() runs. Kept
# under this name (not e.g. VOCAB_LABELS) because main.py / coach_frames.py
# import WLASL_100_LABELS directly -- renaming would mean touching every
# import site for no functional benefit.
WLASL_100_LABELS: list[str] = []


class ModelService:
    def __init__(self) -> None:
        self.model: PoseTransformer | None = None
        self.idx_to_label: dict[int, str] = {}
        self.references: dict = {}
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
        self.is_loaded = False

    def load_model(self) -> None:
        global WLASL_100_LABELS

        override = os.environ.get("MODEL_CHECKPOINT_PATH_ABS", "")
        checkpoint_path = Path(override) if override else DEFAULT_CHECKPOINT
        if not checkpoint_path.exists():
            checkpoint_path = DEFAULT_CHECKPOINT
        if not checkpoint_path.exists():
            logger.warning("No PoseTransformer checkpoint found at %s", checkpoint_path)
            return

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        self.model = PoseTransformer(
            **{k: v for k, v in cfg.items() if k != "num_classes"},
            num_classes=cfg["num_classes"],
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.idx_to_label = ckpt["idx_to_label"]
        WLASL_100_LABELS = [self.idx_to_label[i] for i in range(len(self.idx_to_label))]

        self.is_loaded = True
        logger.info(
            "PoseTransformer (v4canon) loaded from %s — %d classes, %d params, device=%s",
            checkpoint_path.name,
            cfg["num_classes"],
            sum(p.numel() for p in self.model.parameters()),
            self.device,
        )

        if DEFAULT_REFERENCES.exists():
            with open(DEFAULT_REFERENCES, "rb") as f:
                self.references = pickle.load(f)
            logger.info("Coach reference library loaded: %d/%d words covered",
                        len(self.references), cfg["num_classes"])
        else:
            logger.warning(
                "sign_references.pkl not found -- score_against_target will report "
                "no-reference-data for every word until scripts/build_sign_references.py runs."
            )

    def _preprocess(self, keypoints: list[list[float]]) -> tuple[torch.Tensor, torch.Tensor]:
        flat = np.array(keypoints, dtype=np.float32)
        flat = canonicalize_handedness(flat, target="right")
        flat = normalize_per_part(flat)
        T = min(len(flat), MAX_SEQ_LEN)
        padded = flat[:T]
        if T < MAX_SEQ_LEN:
            padded = np.vstack([padded, np.zeros((MAX_SEQ_LEN - T, padded.shape[1]), dtype=np.float32)])
        x = torch.from_numpy(padded).unsqueeze(0).float().to(self.device)
        lengths = torch.tensor([T], device=self.device)
        return x, lengths

    async def predict(self, keypoints: list[list[float]]) -> PredictionResult:
        """Open-set classification -- same signature/shape as the TGCN
        version this replaces, so existing callers don't need to change."""
        if not self.is_loaded or self.model is None:
            return PredictionResult(
                prediction="unknown", confidence=0.0,
                top3=[Prediction(sign="unknown", confidence=0.0)],
                uncertain=True, inference_time_ms=0.0,
            )

        start = time.perf_counter()
        x, lengths = self._preprocess(keypoints)
        with torch.no_grad():
            logits = self.model(x, lengths)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        elapsed = (time.perf_counter() - start) * 1000

        top3_idx = probs.argsort()[::-1][:3]
        top3 = [Prediction(sign=self.idx_to_label[int(i)], confidence=round(float(probs[i]), 4))
                for i in top3_idx]

        return PredictionResult(
            prediction=top3[0].sign if top3[0].confidence >= CONF_THRESH else "unknown",
            confidence=top3[0].confidence,
            top3=top3,
            uncertain=top3[0].confidence < CONF_THRESH,
            inference_time_ms=round(elapsed, 2),
        )

    def score_against_target(self, keypoints: list[list[float]], target_word: str) -> dict:
        """Teach/correct scoring: compare a live attempt against a known
        target sign's reference instead of guessing from scratch. This is
        the "coach angle" -- see src/mouna/coach/compare.py for the full
        reasoning (embedding similarity drives the verdict, hand-count is
        checked separately, the coarse classifier is diagnostic-only)."""
        if not self.is_loaded or self.model is None:
            return {
                "target": target_word, "coarse_prediction": None, "coarse_confidence": None,
                "embedding_similarity": None, "hand_count_match": None, "parameters": None, "divergent_frames": {},
                "overall_verdict": "needs_work", "notes": ["Model not loaded."],
            }
        raw = np.array(keypoints, dtype=np.float32)
        return score_attempt(raw, target_word, self.references, self.model, self.idx_to_label)

    def get_reference_buckets(self, target_word: str) -> dict | None:
        """Reference-only bucket classification (handshape/location/
        orientation) -- no live attempt needed, just the target's own
        reference. Used for the Preview screen's annotated diagram, before
        the camera is even open."""
        ref = self.references.get(target_word)
        if ref is None:
            return None
        raw_canon = ref.get("representative_sequence_raw_canon")
        if raw_canon is None:
            return None
        signal = extract_dominant_hand_signal(raw_canon)
        if signal is None:
            return None
        return {
            "handshape": classify_handshape_bucket(signal),
            "location": classify_location_bucket(signal),
            "orientation": classify_orientation_bucket(signal),
            "hand_bbox": self._reference_hand_bbox(raw_canon),
        }

    def _reference_hand_bbox(self, raw_canon: np.ndarray) -> dict | None:
        """Bounding box covering the dominant hand's range of motion across
        the WHOLE reference clip, normalized [0,1] image coords, same frame
        the reference video itself is rendered in. Used to crop a synced
        'hand-zoom' video on the Preview screen instead of a schematic icon
        -- the real clip is more honest than a drawn one.

        Percentile bounds (5th/95th), not raw min/max -- a single noisy
        misdetected frame (MediaPipe hands do misfire occasionally) would
        otherwise blow the crop out to most of the frame, which happened
        on a real test (apple's y_max hit a clipped 1.0)."""
        arr = raw_canon.reshape(-1, 75, 3)
        rh = arr[:, 54:75, :2]  # (T, 21, 2) -- dominant hand slot, x/y only
        valid = rh[np.any(rh != 0, axis=-1)]
        if len(valid) == 0:
            return None
        margin = 0.08
        min_size = 0.12  # keep it from degenerating into a near-zero-area box for a very still hand
        x_min, x_max = np.percentile(valid[:, 0], [5, 95])
        y_min, y_max = np.percentile(valid[:, 1], [5, 95])
        if x_max - x_min < min_size:
            cx = (x_min + x_max) / 2
            x_min, x_max = cx - min_size / 2, cx + min_size / 2
        if y_max - y_min < min_size:
            cy = (y_min + y_max) / 2
            y_min, y_max = cy - min_size / 2, cy + min_size / 2
        return {
            "x_min": float(max(0.0, x_min - margin)),
            "x_max": float(min(1.0, x_max + margin)),
            "y_min": float(max(0.0, y_min - margin)),
            "y_max": float(min(1.0, y_max + margin)),
        }

    def get_reference_skeleton(self, target_word: str) -> list | None:
        """Reference's normalized skeleton sequence (canonical space only
        -- there's no 'raw camera' ambiguity to show for a pre-recorded
        reference clip the way there is for a live attempt). Both hands,
        full pose. See mouna.coach.skeleton."""
        ref = self.references.get(target_word)
        if ref is None:
            return None
        raw_canon = ref.get("representative_sequence_raw_canon")
        if raw_canon is None:
            return None
        return extract_skeleton_sequence(raw_canon)

    def get_attempt_skeletons(self, keypoints: list[list[float]]) -> dict:
        """Live attempt's skeleton in TWO spaces:
          - "raw": exactly as the camera saw it, correctly left/right-sided
            -- what the user should look at to confirm it looks like them.
          - "canonical": after canonicalize_handedness (mirrored if the
            signer's dominant hand isn't "right") -- the space scoring
            actually happens in, directly comparable to the reference.
        Both include both hands + full pose (see mouna.coach.skeleton)."""
        raw = np.array(keypoints, dtype=np.float32)
        canon = canonicalize_handedness(raw, target="right")
        return {
            "raw": extract_skeleton_sequence(raw),
            "canonical": extract_skeleton_sequence(canon),
        }

    def get_attempt_snapshots(self, keypoints: list[list[float]], divergent_frames: dict) -> dict:
        """One real skeleton frame per mismatched parameter -- the exact
        attempt frame that diverged most from the reference's typical
        value (see mouna.coach.parameters.find_divergent_frame), not an
        evenly-sampled generic moment. Canonical space, since this is
        specifically about the comparison against the reference."""
        if not divergent_frames:
            return {}
        raw = np.array(keypoints, dtype=np.float32)
        canon = canonicalize_handedness(raw, target="right")
        snapshots = {}
        for param, frame_idx in divergent_frames.items():
            frame = extract_skeleton_sequence(canon, frame_indices=[frame_idx])
            if frame:
                snapshots[param] = frame[0]
        return snapshots

    def check_facial_step(self, face_frames: list, target_word: str, nms_type: str) -> dict:
        """One discrete prep-step check: does the live attempt show the
        target sign's required non-manual marker? UNVALIDATED first pass --
        see mouna.coach.facial's module docstring. face_frames: list of
        keypoint_service.extract_face_from_base64() outputs (or None per
        frame where no face was detected)."""
        live_signal = from_live_frames(face_frames)
        result = check_nms(live_signal, nms_type)

        ref = self.references.get(target_word, {})
        ref_face_signal = ref.get("representative_face_signal")
        reference_check = check_nms(ref_face_signal, nms_type) if ref_face_signal else None

        return {
            "step": "facial",
            "nms_type": nms_type,
            "present": result["present"],
            "detected": result["detected"],
            "reference_detected": reference_check["detected"] if reference_check else None,
            "validated": False,  # see facial.py -- no within/across-sign check run yet
        }


model_service = ModelService()
