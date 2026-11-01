#!/usr/bin/env python3
"""Local keyboard-driven review tool for the QA queue built by
qa_build_review_queue.py. Not published anywhere -- runs on localhost only,
serves real video files straight off disk.

Decisions persist to data/silver/qa/review_decisions.csv (keyed by video_id,
upserted on re-review) so progress survives restarts and feeds back into the
dataset-build scripts as an exclude list.

Run: PYTHONPATH=src python3 scripts/qa_review_server.py
Then open http://127.0.0.1:8777
"""

import csv
import json
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, "src")

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

QUEUE_PATH = Path(os.environ.get("MOUNA_QA_QUEUE", "data/silver/qa/review_queue.json"))
DECISIONS_PATH = Path("data/silver/qa/review_decisions.csv")  # always shared -- same underlying clips
DATASET_PATH = Path("data/silver/keypoints/keypoints_official_split_v2norm_aslc.parquet")
PORT = int(os.environ.get("MOUNA_QA_PORT", "8777"))

app = FastAPI()
app.mount("/raw", StaticFiles(directory="data/raw"), name="raw")

_kp_index = None  # lazy video_id -> keypoints_pickle lookup, built once on first request


def get_keypoints_for(video_id: str):
    global _kp_index
    if _kp_index is None:
        df = pd.read_parquet(DATASET_PATH, columns=["video_id", "keypoints_pickle"])
        _kp_index = dict(zip(df["video_id"].astype(str), df["keypoints_pickle"]))
    raw = _kp_index.get(video_id)
    if raw is None:
        return None
    flat = pickle.loads(raw)  # (T, 225): pose(0:99) + left_hand(99:162) + right_hand(162:225)
    pose = flat[:, 0:99].reshape(-1, 33, 3)[:, :, :2]
    left = flat[:, 99:162].reshape(-1, 21, 3)[:, :, :2]
    right = flat[:, 162:225].reshape(-1, 21, 3)[:, :, :2]
    return {
        "pose": pose.tolist(),
        "left_hand": left.tolist(),
        "right_hand": right.tolist(),
        "left_active": bool(not np.allclose(left, 0)),
        "right_active": bool(not np.allclose(right, 0)),
        "num_frames": int(flat.shape[0]),
    }


_raw_kp_index = None  # lazy video_id -> RAW (pre-normalization, [0,1]-space) keypoints


def _build_raw_kp_index():
    idx = {}
    for path in [
        "data/silver/keypoints/keypoints_raw.parquet",
        "data/silver/keypoints/keypoints_msasl_raw.parquet",
        "data/silver/keypoints/keypoints_asl_citizen_raw.parquet",
    ]:
        p = Path(path)
        if not p.exists():
            continue
        df = pd.read_parquet(p, columns=["video_id", "keypoints_pickle", "success"])
        df = df[df["success"]]
        idx.update(dict(zip(df["video_id"].astype(str), df["keypoints_pickle"])))
    return idx


def get_raw_keypoints_for(video_id: str):
    global _raw_kp_index
    if _raw_kp_index is None:
        _raw_kp_index = _build_raw_kp_index()
    raw = _raw_kp_index.get(video_id)
    if raw is not None:
        return pickle.loads(raw)
    # fall back to WLASL OpenPose recovery for ids not in any raw parquet
    try:
        from mouna.data.openpose_convert import convert_video

        return convert_video(video_id)
    except Exception:
        return None


def compute_hand_bbox(video_id: str, pad: float = 0.25):
    """Dominant hand + its [0,1]-space bounding box unioned across all frames,
    padded, for a static crop-zoom on the video element (tracks the real
    hand position regardless of camera framing)."""
    from mouna.data.handedness import _hand_activity

    flat = get_raw_keypoints_for(video_id)
    if flat is None or len(flat) == 0:
        return None
    arr = flat.reshape(-1, 75, 3)
    lh, rh = arr[:, 33:54], arr[:, 54:75]
    l_act, r_act = _hand_activity(lh), _hand_activity(rh)
    dominant_arr, dominant = (rh, "right") if r_act >= l_act else (lh, "left")

    pts = dominant_arr[np.any(dominant_arr != 0, axis=-1)]
    if len(pts) == 0:
        return None
    # percentile, not strict min/max -- a handful of misdetected/outlier
    # frames would otherwise blow the box up to the whole frame
    xmin, ymin = np.percentile(pts[:, 0], 3), np.percentile(pts[:, 1], 3)
    xmax, ymax = np.percentile(pts[:, 0], 97), np.percentile(pts[:, 1], 97)
    w, h = xmax - xmin, ymax - ymin
    xmin -= w * pad; xmax += w * pad
    ymin -= h * pad; ymax += h * pad
    xmin, ymin = max(0.0, xmin), max(0.0, ymin)
    xmax, ymax = min(1.0, xmax), min(1.0, ymax)
    return {"dominant": dominant, "bbox": [float(xmin), float(ymin), float(xmax), float(ymax)]}


def load_decisions() -> dict:
    if not DECISIONS_PATH.exists():
        return {}
    with open(DECISIONS_PATH) as f:
        return {
            row["video_id"]: {"decision": row["decision"], "comment": row.get("comment", "") or ""}
            for row in csv.DictReader(f)
        }


def save_decision(video_id: str, decision: str, comment: str) -> None:
    decisions = load_decisions()
    decisions[video_id] = {"decision": decision, "comment": comment}
    DECISIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DECISIONS_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "decision", "comment"])
        for vid, d in decisions.items():
            writer.writerow([vid, d["decision"], d["comment"]])


class Decision(BaseModel):
    video_id: str
    decision: str
    comment: str = ""


@app.get("/api/queue")
def get_queue():
    queue = json.load(open(QUEUE_PATH))
    for item in queue:
        if item["video_path"]:
            # expose under the /raw mount: data/raw/X -> /raw/X
            item["video_url"] = "/raw/" + item["video_path"].removeprefix("data/raw/")
        else:
            item["video_url"] = None
    return JSONResponse(queue)


@app.get("/api/decisions")
def get_decisions():
    return JSONResponse(load_decisions())


@app.get("/api/keypoints/{video_id}")
def get_keypoints(video_id: str):
    kp = get_keypoints_for(video_id)
    if kp is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse(kp)


@app.get("/api/hand_crop/{video_id}")
def get_hand_crop(video_id: str):
    crop = compute_hand_bbox(video_id)
    if crop is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse(crop)


@app.post("/api/decision")
def post_decision(d: Decision):
    save_decision(d.video_id, d.decision, d.comment)
    return {"ok": True}


@app.get("/", response_class=HTMLResponse)
def index():
    return r"""
<!doctype html><html><head><meta charset="utf-8">
<title>Mouna QA Review</title>
<style>
  * { box-sizing: border-box; }
  body { font-family: -apple-system, sans-serif; background: #15181c; color: #e8e8e8; margin: 0; padding: 20px; }
  #wrap { max-width: 1400px; margin: 0 auto; }
  #progress { color: #888; font-size: 14px; margin-bottom: 4px; }
  #word { font-size: 36px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; }
  #desc { color: #aac; font-size: 15px; margin: 4px 0 16px; }
  #cols { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; align-items: start; }
  @media (max-width: 900px) { #cols { grid-template-columns: 1fr; } }
  #left video { width: 100%; max-height: 560px; background: #000; border-radius: 8px; }
  #vidwrap { width: 100%; max-height: 560px; aspect-ratio: 4/3; overflow: hidden; border-radius: 8px;
    background: #000; position: relative; }
  #vidwrap video { position: absolute; max-height: none; }
  #zoom-btn { margin-top: 8px; }
  #zoom-btn.active { background: #2d4a3a; border-color: #4caf6e; color: #9ddfb0; }
  #zoom-label { margin-left: 8px; }
  #novideo { padding: 16px 20px 20px; text-align: center; background: #262a30; border-radius: 8px; color: #999; }
  #novideo .label { font-size: 13px; margin-bottom: 10px; }
  #skel-panels { display: flex; gap: 10px; }
  #skel-panels .panel { flex: 1; background: #0e1012; border-radius: 6px; }
  #skel-panels .panel-label { font-size: 11px; color: #777; text-align: center; margin-top: 4px; }
  #skel-panels canvas { width: 100%; display: block; }
  #windownote { font-size: 12px; color: #8ab; margin-top: 6px; }
  #right { background: #1a1d21; border: 1px solid #2a2f36; border-radius: 8px; padding: 14px;
    max-height: 560px; overflow-y: auto; }
  #right pre { margin: 0; font-family: "SF Mono", Menlo, monospace; font-size: 12.5px; line-height: 1.6;
    white-space: pre-wrap; word-break: break-word; }
  .jkey { color: #7aa2d9; }
  .jstr { color: #9fd99f; }
  .jnum { color: #d9a441; }
  .jbool-true { color: #4caf6e; font-weight: 600; }
  .jbool-false { color: #d9534f; font-weight: 600; }
  .jnull { color: #666; }
  #bottom { margin-top: 20px; }
  #keys { display: flex; gap: 10px; }
  #comment { width: 100%; margin-bottom: 14px; padding: 10px; font-size: 14px;
    background: #1d2024; border: 1px solid #343a42; border-radius: 8px; color: #e8e8e8; resize: vertical; }
  #comment::placeholder { color: #666; }
  #tags { display: flex; gap: 8px; margin-bottom: 12px; }
  .tag-btn { padding: 6px 12px; font-size: 13px; border: 1px solid #3a4550; border-radius: 14px;
    background: #1d2024; color: #9ab; cursor: pointer; }
  .tag-btn:hover { background: #262a30; }
  .tag-btn.active { background: #2d4a3a; border-color: #4caf6e; color: #9ddfb0; }
  #keys button { flex: 1; padding: 14px; font-size: 15px; border: none; border-radius: 8px; cursor: pointer; font-weight: 600; }
  .k-ok { background: #2d7d4a; color: white; }
  .k-wrong { background: #b3482b; color: white; }
  .k-bad { background: #8a6d1f; color: white; }
  .k-unsure { background: #3a3f46; color: #ddd; }
  #hint { color: #666; font-size: 12px; margin-top: 10px; text-align: center; }
  #done-banner { display:none; text-align:center; padding: 80px 20px; color: #4caf6e; font-size: 20px; }
  .flagpill { background: #5a3030 !important; color: #ffb3b3; }
</style></head>
<body><div id="wrap">
  <div id="progress"></div>
  <div id="word"></div>
  <div id="desc"></div>
  <div id="cols">
    <div id="left">
      <div id="player"></div>
    </div>
    <div id="right"><pre id="jsonpanel"></pre></div>
  </div>
  <div id="bottom">
    <textarea id="comment" rows="2" placeholder="Optional note for fine-tuning (e.g. 'regional variant', 'hand clipped at edge of frame')"></textarea>
    <div id="tags">
      <button class="tag-btn" onclick="addTag('variation')">+ variation</button>
      <button class="tag-btn" onclick="addTag('mirror hand')">+ mirror hand</button>
    </div>
    <div id="keys">
      <button class="k-ok" onclick="decide('ok')">1 &middot; OK</button>
      <button class="k-wrong" onclick="decide('wrong_sign')">2 &middot; Wrong sign</button>
      <button class="k-bad" onclick="decide('bad_video')">3 &middot; Bad video</button>
      <button class="k-unsure" onclick="decide('unsure')">4 &middot; Unsure</button>
    </div>
    <div id="hint">keys 1-4 to decide (disabled while typing a note) &middot; &larr;/&rarr; to navigate without deciding &middot; v/m to tag variation/mirror</div>
  </div>
  <div id="done-banner">All reviewed. You can close this tab.</div>
</div>
<script>
let queue = [], decisions = {}, idx = 0;

function syntaxHighlight(obj) {
  const esc = (s) => String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  let json = JSON.stringify(obj, null, 2);
  json = esc(json);
  return json.replace(/("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false)\b|\bnull\b|-?\d+(\.\d*)?([eE][+-]?\d+)?)/g,
    (match) => {
      let cls = 'jnum';
      if (/^"/.test(match)) cls = /:$/.test(match) ? 'jkey' : 'jstr';
      else if (/true/.test(match)) cls = 'jbool-true';
      else if (/false/.test(match)) cls = 'jbool-false';
      else if (/null/.test(match)) cls = 'jnull';
      return `<span class="${cls}">${match}</span>`;
    });
}

async function load() {
  queue = await (await fetch('/api/queue')).json();
  decisions = await (await fetch('/api/decisions')).json();
  idx = queue.findIndex(q => !(q.video_id in decisions));
  if (idx === -1) idx = 0;
  render();
}

function render() {
  if (idx >= queue.length) { document.getElementById('done-banner').style.display = 'block'; return; }
  const q = queue[idx];
  const done = Object.keys(decisions).length;
  document.getElementById('progress').textContent =
    `${idx+1} / ${queue.length}  (${done} decided)  -- ${q.reason}`;
  document.getElementById('word').textContent = q.gloss;
  document.getElementById('desc').textContent = q.description || '';
  const prior = decisions[q.video_id];
  document.getElementById('comment').value = prior ? (prior.comment || '') : '';
  syncTagButtons();

  const TRIM_LABELS = {
    real_wlasl_window: 'trimmed to the WLASL-declared sign window',
    whole_file_declared_trimmed: 'WLASL marks whole file as pre-trimmed',
    pretrimmed_by_wlasl_unverifiable: 'pre-trimmed by the WLASL pipeline -- no video to verify',
    pretrimmed_at_acquisition: 'pre-trimmed at download/collection time',
  };
  const panel = {
    video_id: q.video_id,
    gloss: q.gloss,
    source: q.source,
    reason_in_queue: q.reason,
    flags: q.flags && q.flags.length ? q.flags : null,
    trim_status: TRIM_LABELS[q.trim_status] || q.trim_status,
    hand_detect_rate: q.hand_detect_rate != null ? +(q.hand_detect_rate*100).toFixed(1) + '%' : null,
    num_frames: q.num_frames,
    model_confidence: q.confidence ? {
      claimed_label_probability: +(q.confidence.claimed_prob*100).toFixed(1) + '%',
      top1_prediction: q.confidence.top1_gloss,
      top1_probability: +(q.confidence.top1_prob*100).toFixed(1) + '%',
      agrees_with_claimed_label: q.confidence.agrees,
      method: '5-fold cross-validated, never trained on this clip',
    } : 'pending (5-fold CV training in progress)',
    provenance: q.provenance,
    prior_review: prior ? { decision: prior.decision, comment: prior.comment || null } : null,
  };
  document.getElementById('jsonpanel').innerHTML = syntaxHighlight(panel);

  if (skelAnimHandle) { clearInterval(skelAnimHandle); skelAnimHandle = null; }
  handZoomOn = false;
  const player = document.getElementById('player');
  if (q.video_url) {
    const windowed = q.start_time != null;
    player.innerHTML = `<div id="vidwrap"><video id="vid" src="${q.video_url}" controls autoplay
      ${windowed ? '' : 'loop'} muted></video></div>
      <button id="zoom-btn" class="tag-btn" onclick="toggleHandZoom('${q.video_id}')">zoom to dominant hand</button>
      <span id="zoom-label" class="panel-label"></span>` +
      (windowed ? `<div id="windownote">playing trimmed window only: ${q.start_time.toFixed(1)}s
        &ndash; ${q.end_time != null ? q.end_time.toFixed(1)+'s' : 'end'} (what the model actually sees,
        not the full source video)</div>` : '');
    if (windowed) {
      const v = document.getElementById('vid');
      const start = q.start_time, end = q.end_time;
      v.addEventListener('loadedmetadata', () => { v.currentTime = start; });
      v.addEventListener('timeupdate', () => {
        if (end != null && v.currentTime >= end) v.currentTime = start;
      });
      v.addEventListener('ended', () => { v.currentTime = start; v.play(); });
    }
  } else {
    player.innerHTML = `<div id="novideo"><div class="label">No video for this source -- rendered from
      the extracted skeleton keypoints instead</div>
      <div id="skel-panels">
        <div class="panel"><canvas id="skel-pose" width="200" height="200"></canvas><div class="panel-label">body</div></div>
        <div class="panel"><canvas id="skel-left" width="200" height="200"></canvas><div class="panel-label" id="label-left">left hand</div></div>
        <div class="panel"><canvas id="skel-right" width="200" height="200"></canvas><div class="panel-label" id="label-right">right hand</div></div>
      </div></div>`;
    loadSkeleton(q.video_id);
  }
}

let skelAnimHandle = null;

let handZoomOn = false;

async function toggleHandZoom(videoId) {
  handZoomOn = !handZoomOn;
  const btn = document.getElementById('zoom-btn');
  const label = document.getElementById('zoom-label');
  const vid = document.getElementById('vid');
  if (!handZoomOn) {
    btn.classList.remove('active');
    label.textContent = '';
    vid.style.width = ''; vid.style.height = ''; vid.style.left = ''; vid.style.top = '';
    return;
  }
  btn.classList.add('active');
  label.textContent = 'loading...';
  const res = await fetch(`/api/hand_crop/${videoId}`);
  if (!res.ok) { label.textContent = 'no hand data available'; handZoomOn = false; btn.classList.remove('active'); return; }
  const c = await res.json();
  const [x0, y0, x1, y1] = c.bbox;
  const bw = Math.max(0.02, x1 - x0), bh = Math.max(0.02, y1 - y0);
  vid.style.width = `${100 / bw}%`;
  vid.style.height = `${100 / bh}%`;
  vid.style.left = `${-100 * x0 / bw}%`;
  vid.style.top = `${-100 * y0 / bh}%`;
  label.textContent = `tracking ${c.dominant} hand`;
}

async function loadSkeleton(videoId) {
  if (skelAnimHandle) { clearInterval(skelAnimHandle); skelAnimHandle = null; }
  const res = await fetch(`/api/keypoints/${videoId}`);
  if (!res.ok) return;
  const kp = await res.json();
  document.getElementById('label-left').textContent =
    'left hand' + (kp.left_active ? '' : ' (no data)');
  document.getElementById('label-right').textContent =
    'right hand' + (kp.right_active ? '' : ' (no data)');

  const POSE_EDGES = [[11,12],[11,13],[13,15],[12,14],[14,16],[0,11],[0,12]];
  const HAND_EDGES = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],
    [0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20],[5,9],[9,13],[13,17]];

  function bounds(frames) {
    let xs=[], ys=[];
    for (const f of frames) for (const [x,y] of f) if (x!==0 || y!==0) { xs.push(x); ys.push(y); }
    if (!xs.length) return {minX:0,maxX:1,minY:0,maxY:1};
    return {minX:Math.min(...xs), maxX:Math.max(...xs), minY:Math.min(...ys), maxY:Math.max(...ys)};
  }
  function drawPanel(canvasId, frames, edges, b) {
    const canvas = document.getElementById(canvasId);
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height, pad = 20;
    const spanX = Math.max(0.05, b.maxX-b.minX), spanY = Math.max(0.05, b.maxY-b.minY);
    const scale = Math.min((W-2*pad)/spanX, (H-2*pad)/spanY);
    const toPx = ([x,y]) => [pad+(x-b.minX)*scale, pad+(y-b.minY)*scale];
    return (frame) => {
      ctx.clearRect(0,0,W,H);
      ctx.strokeStyle = '#4caf6e'; ctx.lineWidth = 2; ctx.fillStyle = '#7ad99a';
      for (const [a,c] of edges) {
        if ((frame[a][0]===0 && frame[a][1]===0) || (frame[c][0]===0 && frame[c][1]===0)) continue;
        const [ax,ay] = toPx(frame[a]), [cx,cy] = toPx(frame[c]);
        ctx.beginPath(); ctx.moveTo(ax,ay); ctx.lineTo(cx,cy); ctx.stroke();
      }
      for (const pt of frame) {
        if (pt[0]===0 && pt[1]===0) continue;
        const [px,py] = toPx(pt);
        ctx.beginPath(); ctx.arc(px,py,2.5,0,7); ctx.fill();
      }
    };
  }
  const drawPose = drawPanel('skel-pose', kp.pose, POSE_EDGES, bounds(kp.pose));
  const drawLeft = drawPanel('skel-left', kp.left_hand, HAND_EDGES, bounds(kp.left_hand));
  const drawRight = drawPanel('skel-right', kp.right_hand, HAND_EDGES, bounds(kp.right_hand));

  let i = 0;
  const n = kp.num_frames;
  skelAnimHandle = setInterval(() => {
    drawPose(kp.pose[i]); drawLeft(kp.left_hand[i]); drawRight(kp.right_hand[i]);
    i = (i+1) % n;
  }, 1000/12);
}

async function decide(d) {
  const q = queue[idx];
  const comment = document.getElementById('comment').value;
  decisions[q.video_id] = {decision: d, comment: comment};
  fetch('/api/decision', {method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({video_id: q.video_id, decision: d, comment: comment})});
  idx++;
  render();
}

function addTag(tag) {
  const box = document.getElementById('comment');
  const cur = box.value.trim();
  const tagRe = new RegExp('(^|,\\s*)' + tag.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + '(\\s*,|$)', 'i');
  if (tagRe.test(cur)) {
    box.value = cur.replace(tagRe, (m, pre, post) => (pre && post) ? ', ' : '').replace(/^,\s*|,\s*$/g, '').trim();
  } else {
    box.value = cur ? `${cur}, ${tag}` : tag;
  }
  syncTagButtons();
}

function syncTagButtons() {
  const text = document.getElementById('comment').value.toLowerCase();
  document.querySelectorAll('.tag-btn').forEach(btn => {
    const tag = btn.textContent.replace('+ ', '').toLowerCase();
    btn.classList.toggle('active', text.includes(tag));
  });
}

document.getElementById('comment').addEventListener('input', syncTagButtons);

document.addEventListener('keydown', (e) => {
  if (document.activeElement.id === 'comment') {
    if (e.key === 'Escape') document.activeElement.blur();
    return;  // let typing happen, don't trigger decisions
  }
  if (e.key === '1') decide('ok');
  else if (e.key === '2') decide('wrong_sign');
  else if (e.key === '3') decide('bad_video');
  else if (e.key === '4') decide('unsure');
  else if (e.key === 'v') addTag('variation');
  else if (e.key === 'm') addTag('mirror hand');
  else if (e.key === 'ArrowRight') { idx = Math.min(idx+1, queue.length); render(); }
  else if (e.key === 'ArrowLeft') { idx = Math.max(idx-1, 0); render(); }
});

load();
</script>
</body></html>
"""


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=PORT)
