#!/usr/bin/env python3
"""Assemble the public backend container (Docker) for the Mouna demo.

Runs on Google Cloud Run (see scripts/deploy_cloud_run.sh), and also works as
a Hugging Face Docker Space or on Railway/Fly. Writes deploy/backend/ (not in
git: it holds videos) with exactly what the
backend needs to serve the practice signs:
  app/backend/            API code (no local DB, .env or unused models)
  src/mouna/              shared code the backend imports
  data/best_model_v4canon.pth, data/references/   model, references, 3D skin
  app/wlasl/videos/       only the practice signs' example, variation and
                          "other signers" clips; app/wlasl/WLASL_v0.3.json
                          trimmed to those clips
  Dockerfile, start.sh, requirements.txt, README.md (Space card)

The container runs in guest mode (no sign-in), keeps its database in /tmp
(resets on restart), and allows a Vercel-hosted frontend (*.vercel.app).

Run:  ./venv/bin/python scripts/build_hf_space.py
Then deploy it: scripts/deploy_cloud_run.sh
"""
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deploy" / "backend"
BACKEND = ROOT / "app" / "backend"
VIDEOS = ROOT / "app" / "wlasl" / "videos"
OTHER_SIGNERS = 3  # matches the Learn screen's "Other people signing it" (routers/signs.py)

DOCKERFILE = """\
FROM python:3.13-slim

# OpenCV / MediaPipe runtime libraries
RUN apt-get update && apt-get install -y --no-install-recommends \\
        libgl1 libglib2.0-0 libgles2 libegl1 \\
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces run as user 1000
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

# CPU-only PyTorch (the default wheel pulls in GPU libraries we don't need)
RUN pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .
USER user

# Paths are relative to app/backend (the working directory of start.sh)
ENV MODEL_CHECKPOINT_PATH=../../data/best_model_v4canon.pth \\
    SIGNS_JSON_PATH=./data/signs.json \\
    WLASL_JSON_PATH=../wlasl/WLASL_v0.3.json \\
    VIDEOS_DIR=../wlasl/videos \\
    MEDIAPIPE_MODEL_DIR=./data/mediapipe_models \\
    DATABASE_URL=sqlite:////tmp/mouna.db \\
    NUM_CLASSES=100 MAX_SEQ_LEN=150 FEATURE_DIM=225 CONFIDENCE_THRESHOLD=0.25 \\
    MIN_ATTEMPTS_FOR_COMPLETION=5 COMPLETION_ACCURACY_THRESHOLD=0.70 \\
    ALLOW_GUEST=true ENABLE_DEBUG_ROUTES=false \\
    CORS_ORIGINS=http://localhost:4200 \\
    CORS_ORIGIN_REGEX=https://.*\\\\.vercel\\\\.app \\
    PYTHONUNBUFFERED=1

EXPOSE 7860
CMD ["sh", "start.sh"]
"""

START = """\
#!/bin/sh
set -e
cd app/backend
# A session-signing secret: set JWT_SECRET_KEY as a Space secret to keep logins
# valid across restarts; otherwise a random one is made at each start.
export JWT_SECRET_KEY="${JWT_SECRET_KEY:-$(python -c 'import secrets; print(secrets.token_urlsafe(48))')}"
alembic upgrade head
# Cloud Run (and most hosts) say which port to use in $PORT; Hugging Face uses 7860
exec uvicorn main:app --host 0.0.0.0 --port "${PORT:-7860}" --proxy-headers --forwarded-allow-ips='*'
"""

REQUIREMENTS = """\
fastapi==0.144.0
uvicorn[standard]==0.54.0
pydantic==2.14.1
pydantic-settings==2.16.2
python-dotenv==1.2.4
numpy==2.5.4
scipy==1.18.1
mediapipe==1.1.1
opencv-python-headless==5.0.0.93
sqlalchemy==2.1.6
alembic==1.20.1
bcrypt==5.0.0
python-jose[cryptography]==3.5.0
email-validator==2.3.0
anthropic==2.12.0
google-genai==2.34.0
"""

README = """\
---
title: Mouna Backend
emoji: 🤟
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
short_description: API for Mouna, an ASL practice coach (research demo)
---

# Mouna backend

The API behind Mouna, an ASL practice coach: the practice signs, their example
videos and 3D skin data, and live grading of a learner's attempt (MediaPipe
tracking + a pose-transformer model + coaching points).

Research demo, non-commercial. Example clips come from public research
datasets (WLASL, MS-ASL, ASL Citizen) and remain under their own terms.

Runs in guest mode: no sign-in; everyone shares one guest account, and the
database resets when the Space restarts.

Optional secrets (Settings -> Variables and secrets):
- `GEMINI_API_KEY` or `ANTHROPIC_API_KEY` -- AI wording for coach messages and the chat
- `JWT_SECRET_KEY` -- keeps sessions valid across restarts
"""


def practice_signs() -> list[str]:
    sys.path.insert(0, str(BACKEND))
    cwd = os.getcwd()
    os.chdir(BACKEND)
    try:
        from services.example_videos import all_choices, practice_ready
        return sorted(w for w in all_choices() if practice_ready(w))
    finally:
        os.chdir(cwd)


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    ignore_backend = shutil.ignore_patterns("__pycache__", "*.pyc", "*.db", ".env", "trained_model", "tgcn_wlasl")
    shutil.copytree(BACKEND, OUT / "app/backend", ignore=ignore_backend)
    shutil.copytree(ROOT / "src/mouna", OUT / "src/mouna", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (OUT / "data").mkdir()
    shutil.copy2(ROOT / "data/best_model_v4canon.pth", OUT / "data")
    shutil.copytree(ROOT / "data/references", OUT / "data/references",
                    ignore=shutil.ignore_patterns("backups", "skin_raw", "*.csv"))

    # videos: each practice sign's example, its labelled variations, and a few other signers
    signs = practice_signs()
    examples = json.loads((ROOT / "data/references/example_videos.json").read_text())
    wlasl = json.loads((ROOT / "app/wlasl/WLASL_v0.3.json").read_text())
    keep: set[str] = set()
    for w in signs:
        keep.add(examples[w]["video_id"])
        keep.update(v["video_id"] for v in examples[w].get("variations", []))
    trimmed = []
    for entry in wlasl:
        if entry["gloss"] not in signs:
            continue
        on_disk = [i for i in entry["instances"] if (VIDEOS / f"{i['video_id']}.mp4").exists()]
        others = [i for i in on_disk if i["video_id"] not in keep][:OTHER_SIGNERS]
        keep.update(i["video_id"] for i in others)
        insts = [i for i in on_disk if i["video_id"] in keep]
        if insts:
            trimmed.append({**entry, "instances": insts})
    (OUT / "app/wlasl/videos").mkdir(parents=True)
    size = 0
    for vid in sorted(keep):
        src = VIDEOS / f"{vid}.mp4"
        if src.exists():
            shutil.copy2(src, OUT / "app/wlasl/videos" / src.name)
            size += src.stat().st_size
    (OUT / "app/wlasl/WLASL_v0.3.json").write_text(json.dumps(trimmed))

    (OUT / "Dockerfile").write_text(DOCKERFILE)
    (OUT / "start.sh").write_text(START)
    (OUT / "requirements.txt").write_text(REQUIREMENTS)
    (OUT / "README.md").write_text(README)
    (OUT / ".dockerignore").write_text("__pycache__\n*.pyc\n.git\n")
    # Uploaders like `railway up` honour the repo's .gitignore (which drops
    # *.mp4 and *.pth); this folder must ship the videos and the model.
    (OUT / ".railwayignore").write_text("!*.mp4\n!*.pth\n")
    (OUT / ".gitignore").write_text("!*.mp4\n!*.pth\n")

    total = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print(f"{len(signs)} practice signs, {len(keep)} videos ({size / 1e6:.0f} MB); image folder {total / 1e6:.0f} MB -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
