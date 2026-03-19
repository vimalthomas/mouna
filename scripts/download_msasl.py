#!/usr/bin/env python3
"""Download MS-ASL clips for glosses that overlap with our WLASL-100 target set.

Steps:
  1. Reads MSASL_train/val/test.json from the MS-ASL/ folder in project root
  2. Filters for glosses in our WLASL-100 target list
  3. Downloads each clip with yt-dlp, trimmed to [start_time, end_time]
  4. Writes a manifest.json compatible with extract_keypoints.py

Usage:
    python3 scripts/download_msasl.py
"""

import json
import subprocess
import time
from pathlib import Path

MSASL_DIR = Path("MS-ASL")   # folder copied into project root
OUTPUT_DIR = Path("data/raw/msasl/videos")
MAX_PER_GLOSS = 30

WLASL_100_GLOSSES = {
    "accident", "africa", "all", "apple", "basketball", "bed", "before",
    "bird", "birthday", "black", "blue", "book", "bowling", "brown", "but",
    "can", "candy", "chair", "change", "cheat", "city", "clothes", "color",
    "computer", "cook", "cool", "corn", "cousin", "cow", "dance", "dark",
    "deaf", "decide", "doctor", "dog", "drink", "eat", "enjoy", "family",
    "fine", "finish", "fish", "forget", "full", "give", "go", "graduate",
    "hat", "hearing", "help", "hot", "how", "jacket", "kiss", "language",
    "last", "later", "letter", "like", "man", "many", "medicine", "meet",
    "mother", "need", "no", "now", "orange", "paint", "paper", "pink",
    "pizza", "play", "pull", "purple", "right", "same", "school",
    "secretary", "shirt", "short", "son", "study", "table", "tall", "tell",
    "thanksgiving", "thin", "thursday", "time", "walk", "want", "what",
    "white", "who", "woman", "work", "wrong", "year", "yes",
}


def fetch_msasl_json() -> list:
    """Load all splits from the MS-ASL folder in project root."""
    samples = []
    for split in ("MSASL_train.json", "MSASL_val.json", "MSASL_test.json"):
        path = MSASL_DIR / split
        with open(path) as f:
            entries = json.load(f)
        print(f"  {split}: {len(entries)} samples")
        samples.extend(entries)
    return samples


def download_clip(url: str, start: float, end: float, dest: Path) -> bool:
    """Download a trimmed clip from YouTube using yt-dlp."""
    section = f"*{start:.2f}-{end:.2f}"
    try:
        result = subprocess.run(
            [
                "yt-dlp",
                "-f", "mp4/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
                "--merge-output-format", "mp4",
                "--download-sections", section,
                "--force-keyframes-at-cuts",
                "-o", str(dest),
                "--quiet", "--no-warnings",
                url,
            ],
            timeout=90,
            capture_output=True,
        )
        if result.returncode == 0 and dest.exists() and dest.stat().st_size > 1000:
            return True
        return False
    except Exception as e:
        print(f"    ERR yt-dlp: {e}")
        return False


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    samples = fetch_msasl_json()
    print(f"MS-ASL total samples: {len(samples)}")

    # Group by gloss text (lowercase) — MS-ASL uses "text" field for the word
    by_gloss: dict[str, list] = {}
    for s in samples:
        text = s.get("text", s.get("clean_text", "")).lower().strip()
        if text in WLASL_100_GLOSSES:
            by_gloss.setdefault(text, []).append(s)

    matched = sorted(by_gloss.keys())
    missing = sorted(WLASL_100_GLOSSES - set(matched))
    print(f"\nGlosses matched in MS-ASL: {len(matched)}/100")
    if missing:
        print(f"Glosses NOT in MS-ASL: {missing}")

    total_ok = 0
    manifest = []

    for gloss in matched:
        instances = by_gloss[gloss]
        pool = instances[:MAX_PER_GLOSS]

        ok = 0
        print(f"\n[{gloss}] {len(instances)} instances, downloading up to {MAX_PER_GLOSS}")

        for inst in pool:
            url = inst.get("url", "")
            if not url:
                continue

            # unique id: use label_id + signer_id or a hash
            vid_id = f"msasl_{gloss}_{inst.get('label_id', 0)}_{inst.get('signer_id', 0)}_{ok}"
            dest = OUTPUT_DIR / f"{vid_id}.mp4"

            if dest.exists() and dest.stat().st_size > 1000:
                print(f"  cached {vid_id}")
                manifest.append({"video_id": vid_id, "gloss": gloss, "path": str(dest)})
                ok += 1
                continue

            start = float(inst.get("start_time", 0))
            end = float(inst.get("end_time", start + 3))
            print(f"  {vid_id} [{start:.1f}s-{end:.1f}s] ...", end=" ", flush=True)

            success = download_clip(url, start, end, dest)
            if success:
                size_kb = dest.stat().st_size // 1024
                print(f"OK ({size_kb} KB)")
                manifest.append({"video_id": vid_id, "gloss": gloss, "path": str(dest)})
                ok += 1
            else:
                if dest.exists():
                    dest.unlink()
                print("FAIL")

            time.sleep(0.3)

        print(f"  => {ok} for '{gloss}'")
        total_ok += ok

    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nTotal: {total_ok} MS-ASL clips across {len(matched)} glosses")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
