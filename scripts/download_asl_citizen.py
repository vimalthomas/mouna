#!/usr/bin/env python3
"""Download the subset of ASL Citizen (Microsoft Research / Boston University,
non-commercial research license) matching our 100-sign vocabulary.

Uses HTTP range requests via `remotezip` to pull only the matching video
entries out of the 46GB archive -- no need to download the full dataset.
ASL Citizen's gloss labels use ASL-LEX numbered variants for a single English
word (e.g. WANT1/WANT2), so a word counts as covered if its bare form or any
`<WORD><digits>` variant exists.
"""

import csv
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

from loguru import logger
from remotezip import RemoteZip

ZIP_URL = "https://download.microsoft.com/download/b/8/8/b88c0bae-e6c1-43e1-8726-98cf5af36ca4/ASL_Citizen.zip"
SIGNS_JSON = Path("app/backend/signs.json")
OUTPUT_DIR = Path("data/raw/asl_citizen")
VIDEOS_DIR = OUTPUT_DIR / "videos"

# (connect_timeout, read_timeout) in seconds -- a single stalled range request
# otherwise hangs forever (requests has no default timeout). Discovered when
# one file sat stuck for 27 minutes with no progress and no error.
REQUEST_TIMEOUT = (10, 30)
MAX_RETRIES = 3


def build_word_to_glosses(rz: RemoteZip, our_words: list[str]) -> dict[str, list[str]]:
    all_glosses = set()
    for split in ["train", "val", "test"]:
        with rz.open(f"ASL_Citizen/splits/{split}.csv") as f:
            for row in csv.DictReader(line.decode() for line in f):
                all_glosses.add(row["Gloss"].strip().upper())

    mapping = {}
    for word in our_words:
        wu = word.upper()
        variants = sorted(g for g in all_glosses if g == wu or re.match(rf"^{wu}[0-9]+$", g))
        mapping[word] = variants
    return mapping


def _arg(name: str) -> str | None:
    for i, a in enumerate(sys.argv):
        if a == name and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return None


def main():
    # --words a,b,c : fetch these words instead of the app vocabulary
    # --dry-run     : only report what the archive has (clips, signers,
    #                 numbered variants); downloads nothing
    words_arg = _arg("--words")
    dry_run = "--dry-run" in sys.argv
    our_words = [w.strip().lower() for w in words_arg.split(",")] if words_arg else [s["word"] for s in json.load(open(SIGNS_JSON))["signs"]]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

    rz = RemoteZip(ZIP_URL, timeout=REQUEST_TIMEOUT)
    word_to_glosses = build_word_to_glosses(rz, our_words)

    uncovered = [w for w, g in word_to_glosses.items() if not g]
    logger.info(f"Vocabulary coverage: {len(our_words) - len(uncovered)}/{len(our_words)}; uncovered: {uncovered}")

    gloss_to_word = {g: w for w, glosses in word_to_glosses.items() for g in glosses}

    rows_by_gloss = {}
    for split in ["train", "val", "test"]:
        with rz.open(f"ASL_Citizen/splits/{split}.csv") as f:
            for row in csv.DictReader(line.decode() for line in f):
                gloss = row["Gloss"].strip().upper()
                if gloss in gloss_to_word:
                    rows_by_gloss.setdefault(gloss, []).append(
                        {
                            "video_file": row["Video file"],
                            "asl_citizen_gloss": gloss,
                            "word": gloss_to_word[gloss],
                            "participant_id": row["Participant ID"],
                            "asl_citizen_split": split,
                        }
                    )

    manifest = [r for rows in rows_by_gloss.values() for r in rows]

    if dry_run:
        for word in our_words:
            for gloss in word_to_glosses.get(word, []):
                rows = rows_by_gloss.get(gloss, [])
                people = {r["participant_id"] for r in rows}
                print(f"{word:10s} {gloss:12s} clips={len(rows):3d} signers={len(people):3d}")
            if not word_to_glosses.get(word):
                print(f"{word:10s} -- not in ASL Citizen")
        return 0

    # Merge with what's already downloaded instead of overwriting the
    # manifest (a --words run must not drop the existing vocabulary's rows).
    manifest_path = OUTPUT_DIR / "manifest.json"
    if manifest_path.exists():
        existing = json.load(open(manifest_path))
        have = {r["video_file"] for r in existing}
        manifest = existing + [r for r in manifest if r["video_file"] not in have]
    max_videos = os.environ.get("MOUNA_ASLC_MAX")
    if max_videos is not None:
        manifest = manifest[: int(max_videos)]
    logger.info(f"Total videos to download: {len(manifest)}")

    failed_path = OUTPUT_DIR / "failed.json"
    done, failed = 0, []
    nested = VIDEOS_DIR / "ASL_Citizen"

    for entry in manifest:
        dest = VIDEOS_DIR / entry["video_file"]
        if dest.exists():
            done += 1
            continue

        extracted = nested / "videos" / entry["video_file"]
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                rz.extract(f"ASL_Citizen/videos/{entry['video_file']}", path=str(VIDEOS_DIR))
                extracted.rename(dest)
                done += 1
                break
            except Exception as e:
                extracted.unlink(missing_ok=True)
                if attempt == MAX_RETRIES:
                    logger.warning(f"giving up on {entry['video_file']} after {MAX_RETRIES} attempts: {e}")
                    failed.append(entry["video_file"])
                else:
                    logger.warning(f"retry {attempt}/{MAX_RETRIES} for {entry['video_file']}: {e}")
                    time.sleep(2 * attempt)
                    rz = RemoteZip(ZIP_URL, timeout=REQUEST_TIMEOUT)  # fresh connection

        if (done + len(failed)) % 200 == 0 or (done + len(failed)) == len(manifest):
            logger.info(f"{done} downloaded, {len(failed)} failed, {len(manifest) - done - len(failed)} remaining")
            json.dump(manifest, open(manifest_path, "w"), indent=2)
            json.dump(failed, open(failed_path, "w"), indent=2)

    if nested.exists():
        shutil.rmtree(nested)

    json.dump(manifest, open(manifest_path, "w"), indent=2)
    json.dump(failed, open(failed_path, "w"), indent=2)
    logger.info(f"Done. {done} videos in {VIDEOS_DIR}, {len(failed)} failed (see {failed_path})")


if __name__ == "__main__":
    sys.exit(main())
