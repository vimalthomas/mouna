import json
import logging

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from services.keypoint_service import keypoint_service
from services.model_service import model_service

logger = logging.getLogger(__name__)

router = APIRouter()


def _analyze_keypoints(keypoints: list[list[float]]) -> dict:
    """Deep analysis of keypoint data to diagnose issues."""
    n_frames = len(keypoints)
    arr = np.array(keypoints, dtype=np.float32)

    # Check each landmark group
    # 225 = 33 pose * 3 + 21 left hand * 3 + 21 right hand * 3
    pose_data = arr[:, :99]       # 33 landmarks * 3
    left_hand = arr[:, 99:162]    # 21 landmarks * 3
    right_hand = arr[:, 162:225]  # 21 landmarks * 3

    pose_nonzero = np.count_nonzero(pose_data)
    left_nonzero = np.count_nonzero(left_hand)
    right_nonzero = np.count_nonzero(right_hand)

    # Hand position analysis
    hand_info = {}
    for name, data, offset in [("left_hand", left_hand, 99), ("right_hand", right_hand, 162)]:
        if np.count_nonzero(data) > 0:
            # Get wrist position (first landmark)
            wrist_x = data[:, 0].mean()
            wrist_y = data[:, 1].mean()
            # Get movement range
            xs = data[:, 0::3]
            ys = data[:, 1::3]
            x_range = xs[xs != 0].max() - xs[xs != 0].min() if np.count_nonzero(xs) > 0 else 0
            y_range = ys[ys != 0].max() - ys[ys != 0].min() if np.count_nonzero(ys) > 0 else 0
            hand_info[name] = {
                "detected": True,
                "wrist_avg_x": round(float(wrist_x), 3),
                "wrist_avg_y": round(float(wrist_y), 3),
                "x_movement": round(float(x_range), 4),
                "y_movement": round(float(y_range), 4),
                "frames_with_data": int(np.any(data != 0, axis=1).sum()),
            }
        else:
            hand_info[name] = {"detected": False}

    # Body position
    body_info = {}
    if pose_nonzero > 0:
        landmarks_75 = arr.reshape(n_frames, 75, 3)
        nose = landmarks_75[:, 0, :2].mean(axis=0)
        l_shoulder = landmarks_75[:, 11, :2].mean(axis=0)
        r_shoulder = landmarks_75[:, 12, :2].mean(axis=0)
        l_wrist = landmarks_75[:, 15, :2].mean(axis=0)
        r_wrist = landmarks_75[:, 16, :2].mean(axis=0)
        body_info = {
            "nose": [round(float(v), 3) for v in nose],
            "left_shoulder": [round(float(v), 3) for v in l_shoulder],
            "right_shoulder": [round(float(v), 3) for v in r_shoulder],
            "left_wrist": [round(float(v), 3) for v in l_wrist],
            "right_wrist": [round(float(v), 3) for v in r_wrist],
        }

    return {
        "input": {
            "frames": n_frames,
            "dim_per_frame": len(keypoints[0]) if keypoints else 0,
        },
        "mediapipe_detection": {
            "pose_values": int(pose_nonzero),
            "left_hand_values": int(left_nonzero),
            "right_hand_values": int(right_nonzero),
            "pose_detected": pose_nonzero > 0,
            "left_hand_detected": left_nonzero > 0,
            "right_hand_detected": right_nonzero > 0,
        },
        "hands": hand_info,
        "body": body_info,
    }


@router.websocket("/ws/debug")
async def debug_detect(websocket: WebSocket) -> None:
    await websocket.accept()
    logger.info("Debug WebSocket connected")

    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)

            # Support both raw keypoints and base64 frames
            frames = data.get("frames", [])
            keypoints = data.get("keypoints", [])

            if frames:
                keypoints = [keypoint_service.extract_from_base64(f) for f in frames]

            if not keypoints or len(keypoints) < 5:
                await websocket.send_json({"type": "error", "message": "Need >= 5 frames"})
                continue

            # Analyze
            analysis = _analyze_keypoints(keypoints)

            # Predict
            result = await model_service.predict(keypoints)

            await websocket.send_json({
                "type": "debug",
                "analysis": analysis,
                "prediction": {
                    "sign": result.prediction,
                    "confidence": result.confidence,
                    "top3": [{"sign": p.sign, "confidence": p.confidence} for p in result.top3],
                    "uncertain": result.uncertain,
                    "inference_ms": result.inference_time_ms,
                },
                "diagnosis": _diagnose(analysis, result.prediction),
            })

    except WebSocketDisconnect:
        logger.info("Debug WebSocket disconnected")
    except Exception:
        logger.exception("Debug WebSocket error")


def _diagnose(analysis: dict, prediction: str) -> list[str]:
    issues = []

    mp = analysis["mediapipe_detection"]
    if not mp["pose_detected"]:
        issues.append("CRITICAL: No pose detected — make sure your body is visible")
    if not mp["left_hand_detected"] and not mp["right_hand_detected"]:
        issues.append("CRITICAL: No hands detected — MediaPipe hand detection not working. Model CANNOT recognize signs without hand data.")
    elif not mp["left_hand_detected"]:
        issues.append("WARNING: Left hand not detected")
    elif not mp["right_hand_detected"]:
        issues.append("WARNING: Right hand not detected")

    hands = analysis["hands"]
    for h in ["left_hand", "right_hand"]:
        if hands[h].get("detected"):
            movement = hands[h].get("x_movement", 0) + hands[h].get("y_movement", 0)
            if movement < 0.01:
                issues.append(f"WARNING: {h} detected but not moving — static hand")

    if not issues:
        issues.append("OK: All data looks good — pose and hands detected")

    return issues
