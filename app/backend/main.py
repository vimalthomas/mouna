"""
Mouna FastAPI backend
- WebSocket /ws/detect      : accepts keypoint buffer, returns top-3 predictions
- GET        /signs          : returns the full sign dictionary
- GET        /video/{word}   : returns URL + timestamps of a sample video for that sign
- GET        /health         : liveness check
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from mouna.models.transformer import PoseTransformer

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH    = Path(__file__).parent.parent.parent / "data" / "best_model_transformer.pth"
SIGNS_PATH    = Path(__file__).parent / "signs.json"
VIDEO_DIR     = Path(__file__).parent.parent.parent / "data" / "raw" / "wlasl" / "videos"
MANIFEST_PATH = VIDEO_DIR / "manifest.json"
WLASL_JSON    = Path(__file__).parent.parent.parent / "data" / "raw" / "wlasl" / "WLASL_v0.3.json"
MAX_SEQ_LEN   = 150
CONF_THRESH   = 0.25

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Mouna Sign Language API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if VIDEO_DIR.exists():
    app.mount("/videos", StaticFiles(directory=str(VIDEO_DIR)), name="videos")

# ── State ─────────────────────────────────────────────────────────────────────
model = None
idx_to_label = None
# gloss → {"filename": str, "start": float|None, "end": float|None}
video_index: dict[str, dict] = {}


@app.on_event("startup")
def load_model():
    global model, idx_to_label, video_index

    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    cfg  = ckpt["config"]
    model = PoseTransformer(
        input_dim      = cfg["input_dim"],
        d_model        = cfg["d_model"],
        nhead          = cfg["nhead"],
        num_layers     = cfg["num_layers"],
        dim_feedforward= cfg["dim_feedforward"],
        dropout        = cfg.get("dropout", 0.3),
        num_classes    = cfg["num_classes"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    idx_to_label = ckpt["idx_to_label"]
    print(f"Model loaded — {cfg['num_classes']} classes | {sum(p.numel() for p in model.parameters()):,} params")

    # Build video_id → timestamps from WLASL JSON
    ts_index: dict[str, dict] = {}
    if WLASL_JSON.exists():
        with open(WLASL_JSON) as f:
            wlasl = json.load(f)
        for entry in wlasl:
            for inst in entry.get("instances", []):
                vid_id  = str(inst["video_id"])
                fps     = float(inst.get("fps", 25) or 25)
                fs      = inst.get("frame_start", 1) or 1
                fe      = inst.get("frame_end", -1)
                start   = round((fs - 1) / fps, 2) if fs > 1 else None
                end     = round(fe / fps, 2)        if fe  > 0 else None
                ts_index[vid_id] = {"start": start, "end": end}

    # Build gloss → first available video + timestamps
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH) as f:
            manifest = json.load(f)
        for entry in manifest:
            gloss = entry["gloss"].lower()
            if gloss in video_index:
                continue
            fname = Path(entry["path"]).name
            if not (VIDEO_DIR / fname).exists():
                continue
            ts = ts_index.get(str(entry["video_id"]), {})
            video_index[gloss] = {
                "filename": fname,
                "start":    ts.get("start"),
                "end":      ts.get("end"),
            }
        print(f"Video index: {len(video_index)} glosses with sample videos")


def run_inference(keypoints: list[list[float]]) -> dict:
    flat    = np.array(keypoints, dtype=np.float32)
    T       = len(flat)
    mean    = flat.mean(axis=0, keepdims=True)
    std     = flat.std(axis=0,  keepdims=True) + 1e-8
    flat    = (flat - mean) / std
    seq_len = min(T, MAX_SEQ_LEN)
    if T < MAX_SEQ_LEN:
        flat = np.vstack([flat, np.zeros((MAX_SEQ_LEN - T, flat.shape[1]), dtype=np.float32)])
    else:
        flat = flat[:MAX_SEQ_LEN]

    x       = torch.from_numpy(flat).unsqueeze(0)
    lengths = torch.tensor([seq_len], dtype=torch.long)
    with torch.no_grad():
        logits = model(x, lengths)
        probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

    top3_idx = probs.argsort()[::-1][:3]
    top3     = [{"label": idx_to_label[int(i)], "confidence": float(probs[i])} for i in top3_idx]
    return {
        "prediction": top3[0]["label"] if top3[0]["confidence"] >= CONF_THRESH else "uncertain",
        "confidence": top3[0]["confidence"],
        "top3":       top3,
        "uncertain":  top3[0]["confidence"] < CONF_THRESH,
    }


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/signs")
def get_signs():
    with open(SIGNS_PATH) as f:
        return json.load(f)


@app.get("/video/{word}")
def get_video(word: str):
    info = video_index.get(word.lower())
    if not info:
        return {"url": None, "start": None, "end": None}
    return {
        "url":   f"/videos/{info['filename']}",
        "start": info["start"],
        "end":   info["end"],
    }


@app.websocket("/ws/detect")
async def detect(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data      = await ws.receive_json()
            keypoints = data.get("keypoints", [])
            if not keypoints:
                await ws.send_json({"error": "no keypoints received"})
                continue
            result = run_inference(keypoints)
            await ws.send_json(result)
    except WebSocketDisconnect:
        pass
