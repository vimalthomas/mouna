#!/usr/bin/env python3
"""Download WLASL videos for the top 100 glosses (by instance count).
- Direct MP4 URLs: urllib
- YouTube URLs: yt-dlp
"""

import json
import subprocess
import time
import urllib.request
from pathlib import Path

WLASL_JSON = "data/raw/wlasl/WLASL_v0.3.json"
OUTPUT_DIR = Path("data/raw/wlasl/videos")
TOP_N_GLOSSES = 100   # top N glosses by instance count (WLASL-100 benchmark)
MAX_PER_GLOSS = 999   # effectively unlimited — grab every available instance per gloss


def is_youtube(url: str) -> bool:
    return "youtube.com" in url or "youtu.be" in url


def download_direct(url: str, dest: Path, timeout: int = 15) -> bool:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        if len(data) < 1000:
            return False
        dest.write_bytes(data)
        return True
    except Exception as e:
        print(f"    ERR: {e}")
        return False


def download_youtube(url: str, dest: Path) -> bool:
    try:
        result = subprocess.run(
            [
                "yt-dlp", "--js-runtimes", "node",
                "-f", "mp4/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
                "--merge-output-format", "mp4",
                "-o", str(dest),
                "--quiet", "--no-warnings",
                url,
            ],
            timeout=60,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and dest.exists() and dest.stat().st_size > 1000:
            return True
        # Surface the real reason instead of a bare FAIL, so skipped videos are distinguishable
        # (dead/private/unavailable) from anything actually worth retrying.
        reason = (result.stderr or result.stdout or "").strip().splitlines()
        if reason:
            print(f"    {reason[-1]}", end=" ")
        return False
    except Exception as e:
        print(f"    ERR yt-dlp: {e}")
        return False


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(WLASL_JSON) as f:
        wlasl = json.load(f)

    # Sort glosses by instance count descending, pick top N
    glosses_sorted = sorted(wlasl, key=lambda x: len(x["instances"]), reverse=True)
    target_glosses = [entry["gloss"] for entry in glosses_sorted[:TOP_N_GLOSSES]]
    gloss_map = {entry["gloss"]: entry["instances"] for entry in wlasl}

    print(f"Targeting top {TOP_N_GLOSSES} glosses (max {MAX_PER_GLOSS} videos each)")
    print(f"Glosses: {target_glosses[:10]}... (and {len(target_glosses)-10} more)")

    total_ok = 0
    manifest = []

    for gloss in target_glosses:
        instances = gloss_map.get(gloss, [])
        ok = 0
        print(f"\n[{gloss}] {len(instances)} instances, downloading up to {MAX_PER_GLOSS}")

        for inst in instances:
            if ok >= MAX_PER_GLOSS:
                break
            video_id = str(inst["video_id"])
            url = inst.get("url", "")
            if not url:
                continue

            dest = OUTPUT_DIR / f"{video_id}.mp4"
            if dest.exists() and dest.stat().st_size > 10_000:
                print(f"  cached {video_id}")
                manifest.append({"video_id": video_id, "gloss": gloss, "path": str(dest)})
                ok += 1
                continue

            print(f"  {video_id} ...", end=" ", flush=True)
            if is_youtube(url):
                success = download_youtube(url, dest)
            else:
                success = download_direct(url, dest)

            if success:
                size_kb = dest.stat().st_size // 1024
                print(f"OK ({size_kb} KB)")
                manifest.append({"video_id": video_id, "gloss": gloss, "path": str(dest)})
                ok += 1
            else:
                if dest.exists():
                    dest.unlink()
                print("FAIL")
            time.sleep(0.2)

        print(f"  => {ok} for '{gloss}'")
        total_ok += ok

    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nTotal: {total_ok} videos across {len(target_glosses)} glosses")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
