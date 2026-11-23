import base64
import logging
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from config import settings

logger = logging.getLogger(__name__)

# Two detected hands whose 21 points are on average closer than this
# (normalized image units) are one hand detected twice.
DUPLICATE_HAND_DIST = 0.015


def is_duplicate_hand(a: np.ndarray, b: np.ndarray) -> bool:
    """True when two detected hands (21x2 image points each) are the same
    hand reported twice. Real pressed-together hands are mirror images and
    measured >= 0.029 apart on bed/cook/school (n=401 frames)."""
    return float(np.mean(np.linalg.norm(a - b, axis=1))) < DUPLICATE_HAND_DIST

# /app/mediapipe_models is the Docker-container layout (image root is /app);
# settings.mediapipe_model_dir overrides this for local dev, where / is
# read-only and there's no such directory.
MODEL_DIR = Path(settings.mediapipe_model_dir)
POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
FACE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"


class KeypointService:
    def __init__(self) -> None:
        self.pose = None
        self.hands = None
        self.face = None
        self._initialized = False

    def _download_model(self, url: str, name: str) -> str:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        path = MODEL_DIR / name
        if not path.exists():
            logger.info("Downloading %s...", name)
            urllib.request.urlretrieve(url, str(path))
            logger.info("Downloaded %s (%d bytes)", name, path.stat().st_size)
        return str(path)

    def initialize(self) -> None:
        if self._initialized:
            return

        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        pose_model = self._download_model(POSE_MODEL_URL, "pose_landmarker.task")
        hand_model = self._download_model(HAND_MODEL_URL, "hand_landmarker.task")

        pose_options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=pose_model),
            running_mode=vision.RunningMode.IMAGE,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.pose = vision.PoseLandmarker.create_from_options(pose_options)

        hand_options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=hand_model),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.hands = vision.HandLandmarker.create_from_options(hand_options)

        face_model = self._download_model(FACE_MODEL_URL, "face_landmarker.task")
        face_options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=face_model),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self.face = vision.FaceLandmarker.create_from_options(face_options)

        self._initialized = True
        logger.info("MediaPipe Tasks initialized (pose + hands + face)")

    def extract_from_base64(self, b64_frame: str) -> list[float]:
        """Extract 225-dim keypoints from a base64-encoded JPEG frame."""
        try:
            img_bytes = base64.b64decode(b64_frame)
            img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
            frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            if frame is None:
                return [0.0] * 225
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return self._extract(rgb)
        except Exception:
            logger.exception("Frame extraction error")
            return [0.0] * 225

    def _extract(self, rgb: np.ndarray) -> list[float]:
        import mediapipe as mp

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        keypoints: list[float] = []

        # Pose: 33 landmarks x 3
        pose_result = self.pose.detect(mp_image)
        if pose_result.pose_landmarks and len(pose_result.pose_landmarks) > 0:
            for lm in pose_result.pose_landmarks[0]:
                keypoints.extend([lm.x, lm.y, lm.z])
        else:
            keypoints.extend([0.0] * 99)

        # Hands
        hand_result = self.hands.detect(mp_image)
        left_hand = [0.0] * 63
        right_hand = [0.0] * 63

        if hand_result.hand_landmarks and hand_result.handedness:
            detections = list(zip(hand_result.hand_landmarks, hand_result.handedness))
            # One real hand is sometimes reported twice (once per label) at
            # nearly the same spot: in the 3D view that showed as two hands
            # inside each other, and it can fake a "two hands" count. Keep
            # the more confident one when the two palms almost coincide.
            if len(detections) == 2:
                # Same hand twice = all 21 points nearly identical. Two real
                # hands pressed together (bed, cook) are mirror images --
                # thumbs on opposite sides -- so they stay well above this.
                (a_lm, _), (b_lm, _) = detections
                a = np.array([[p.x, p.y] for p in a_lm])
                b = np.array([[p.x, p.y] for p in b_lm])
                if is_duplicate_hand(a, b):
                    detections = [max(detections, key=lambda d: d[1][0].score)]
            for hand_lm, handedness in detections:
                label = handedness[0].category_name
                coords = []
                for lm in hand_lm:
                    coords.extend([lm.x, lm.y, lm.z])
                # Direct mapping (NOT flipped) -- must match
                # src/mouna/data/preprocessing.py's KeypointExtractor exactly,
                # since that's the convention the deployed model was trained
                # on. The original flip here was self-consistent for the
                # TGCN this file used to serve, but flipping would silently
                # feed canonicalize_handedness mirrored-when-it-shouldn't-be
                # data for the PoseTransformer -- a real correctness bug,
                # not a style choice.
                if label == "Left":
                    left_hand = coords
                elif label == "Right":
                    right_hand = coords

        keypoints.extend(left_hand)
        keypoints.extend(right_hand)

        return keypoints  # 225 floats

    def extract_face_from_base64(self, b64_frame: str) -> dict | None:
        """Single-frame facial signal: blendshape scores (dict name->score),
        head yaw in degrees, and the raw 478-point face mesh (x,y only, same
        normalized image-plane space pose/hands come in) -- or None if no
        face was detected. Opt-in, not called on every coach-session frame
        -- only during an NMS-check step (see routers/coach_frames.py) since
        it's an extra model pass per frame on top of pose+hands.

        landmarks_xy is the newer addition (see face_mesh.py) -- raw points
        only, no contour selection here, so the choice of which points trace
        a renderable outline stays in one place (face_mesh.REGIONS)."""
        import mediapipe as mp

        try:
            img_bytes = base64.b64decode(b64_frame)
            img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
            frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            if frame is None:
                return None
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self.face.detect(mp_image)

            if not result.face_blendshapes:
                return None
            blendshapes = {c.category_name: float(c.score) for c in result.face_blendshapes[0]}

            yaw_deg = None
            if result.facial_transformation_matrixes:
                m = np.array(result.facial_transformation_matrixes[0])
                yaw_deg = float(np.degrees(np.arctan2(-m[2, 0], np.sqrt(m[2, 1] ** 2 + m[2, 2] ** 2))))

            landmarks_xy = landmarks_xyz = None
            if result.face_landmarks:
                landmarks_xy = [[lm.x, lm.y] for lm in result.face_landmarks[0]]
                # with depth, for the 3D face skin on the results screen (v3)
                landmarks_xyz = [[lm.x, lm.y, lm.z] for lm in result.face_landmarks[0]]

            return {"blendshapes": blendshapes, "head_yaw_deg": yaw_deg, "landmarks_xy": landmarks_xy,
                    "landmarks_xyz": landmarks_xyz}
        except Exception:
            logger.exception("Face extraction error")
            return None


keypoint_service = KeypointService()
