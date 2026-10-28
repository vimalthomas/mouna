#!/usr/bin/env python3
"""Download official MS-ASL videos for our 100-sign target vocabulary, across
all three splits (train/val/test) — 4,149 total instances covering 98 of our
100 signs (confirmed via MSASL_classes.json; 'chair' and 'cheat' aren't in
MS-ASL's vocabulary at all). This is the OFFICIAL Microsoft-released dataset
(download.microsoft.com), not the ad-hoc 797-video set downloaded earlier in
this project whose source/completeness couldn't be verified.

Trims each clip to [start_time, end_time] via ffmpeg after download, since
MS-ASL entries are sub-clips of longer YouTube videos (same trim-after-
download approach as WLASL's frame_start/frame_end handling).
"""

import json
import os
import subprocess
import time
from pathlib import Path

MSASL_DIR = Path("data/raw/msasl_official")
OUTPUT_DIR = Path("data/raw/msasl_official/videos")

# Batch-then-pause: YouTube's bot-detection triggered twice on a long unbroken
# run (once ~1400 requests in, once ~2419 in, despite a slower per-request
# delay). Spreading real network requests into smaller bursts with a cooldown
# between them is a gentler request pattern less likely to trip it.
BATCH_SIZE = int(os.environ.get("MOUNA_MSASL_BATCH_SIZE", 300))
PAUSE_SECONDS = int(os.environ.get("MOUNA_MSASL_PAUSE_SECONDS", 120))
TARGET_WORDS_PATH = Path("app/backend/signs.json")


def load_target_words() -> set:
    signs = json.load(open(TARGET_WORDS_PATH))
    return set(s["word"] for s in signs["signs"])


COOKIES_PATH = Path.home() / "mouna" / "youtube_cookies.txt"


def download_and_trim(url: str, start_time: float, end_time: float, dest: Path) -> bool:
    tmp = dest.with_suffix(".raw.mp4")
    try:
        cmd = ["yt-dlp", "--js-runtimes", "node"]
        if COOKIES_PATH.exists():
            cmd += ["--cookies", str(COOKIES_PATH)]
        cmd += [
            "-f", "mp4/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "--merge-output-format", "mp4",
            "-o", str(tmp),
            "--quiet", "--no-warnings",
            url,
        ]
        result = subprocess.run(cmd, timeout=60, capture_output=True, text=True)
        if result.returncode != 0 or not tmp.exists() or tmp.stat().st_size < 1000:
            reason = (result.stderr or result.stdout or "").strip().splitlines()
            if reason:
                print(f"    {reason[-1]}", end=" ")
            return False

        duration = max(0.1, end_time - start_time)
        trim = subprocess.run(
            [
                "ffmpeg", "-y", "-ss", str(start_time), "-i", str(tmp),
                "-t", str(duration), "-c", "copy", str(dest),
            ],
            timeout=30, capture_output=True, text=True,
        )
        tmp.unlink(missing_ok=True)
        if trim.returncode != 0 or not dest.exists() or dest.stat().st_size < 500:
            print(f"    ffmpeg trim failed", end=" ")
            return False
        return True
    except Exception as e:
        print(f"    ERR: {e}", end=" ")
        tmp.unlink(missing_ok=True)
        return False


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    target_words = load_target_words()
    print(f"Target vocabulary: {len(target_words)} words")

    entries = []
    for split_name in ["train", "val", "test"]:
        split = json.load(open(MSASL_DIR / f"MSASL_{split_name}.json"))
        for e in split:
            if e["clean_text"] in target_words:
                e["_split"] = split_name
                entries.append(e)
    print(f"Matching instances across train/val/test: {len(entries)}")

    manifest = []
    ok, fail = 0, 0
    network_requests_this_batch = 0
    for i, e in enumerate(entries):
        gloss = e["clean_text"]
        video_id = f"msasl2_{gloss}_{e['_split']}_{e['label']}_{i}"
        dest = OUTPUT_DIR / f"{video_id}.mp4"

        if dest.exists() and dest.stat().st_size > 500:
            manifest.append({"video_id": video_id, "gloss": gloss, "path": str(dest)})
            ok += 1
            continue

        print(f"[{i+1}/{len(entries)}] {gloss} ({video_id})...", end=" ", flush=True)
        success = download_and_trim(e["url"], e["start_time"], e["end_time"], dest)
        if success:
            print(f"OK ({dest.stat().st_size // 1024} KB)")
            manifest.append({"video_id": video_id, "gloss": gloss, "path": str(dest)})
            ok += 1
        else:
            print("FAIL")
            fail += 1
        time.sleep(1.0)  # bumped from 0.2s after hitting YouTube's bot-detection rate limit earlier

        network_requests_this_batch += 1
        if network_requests_this_batch >= BATCH_SIZE:
            json.dump(manifest, open(OUTPUT_DIR / "manifest.json", "w"), indent=2)
            print(f"\n--- batch of {BATCH_SIZE} real requests done ({ok} ok, {fail} fail so far) — "
                  f"pausing {PAUSE_SECONDS}s before next batch ---")
            time.sleep(PAUSE_SECONDS)
            network_requests_this_batch = 0

        if (i + 1) % 200 == 0:
            print(f"  progress: {i+1}/{len(entries)} ({ok} ok, {fail} fail)")
            # Write incrementally — a killed/interrupted run (e.g. rate-limit
            # cooldown) must not lose already-downloaded videos' manifest entries.
            json.dump(manifest, open(OUTPUT_DIR / "manifest.json", "w"), indent=2)

    manifest_path = OUTPUT_DIR / "manifest.json"
    json.dump(manifest, open(manifest_path, "w"), indent=2)
    print(f"\nTotal: {ok}/{ok+fail} downloaded")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
