#!/usr/bin/env python3
"""Upload validated WLASL videos to Azure Blob Storage (bronze container).

Organises as: sign-videos-bronze/videos/{gloss}/{video_id}.mp4
Also uploads manifest.json for use by the Databricks extraction job.

Requires env vars:
  AZURE_STORAGE_ACCOUNT  (default: mounastorage2025)
  AZURE_STORAGE_KEY
"""

import json
import os
from pathlib import Path

from azure.storage.blob import BlobServiceClient

MANIFEST = Path("data/raw/wlasl/videos/manifest.json")
CONTAINER = "sign-videos-bronze"


def main():
    account = os.environ.get("AZURE_STORAGE_ACCOUNT", "mounastorage2025")
    key = os.environ["AZURE_STORAGE_KEY"]

    service = BlobServiceClient(
        account_url=f"https://{account}.blob.core.windows.net",
        credential=key,
    )
    container_client = service.get_container_client(CONTAINER)

    with open(MANIFEST) as f:
        manifest = json.load(f)

    print(f"Uploading {len(manifest)} videos to {account}/{CONTAINER}...")

    uploaded, skipped, missing = 0, 0, 0
    for i, entry in enumerate(manifest):
        video_id = entry["video_id"]
        gloss = entry["gloss"]
        path = Path(entry["path"])

        if not path.exists():
            print(f"  SKIP {video_id} — file missing locally")
            missing += 1
            continue

        blob_name = f"videos/{gloss}/{video_id}.mp4"
        blob_client = container_client.get_blob_client(blob_name)

        # Skip if already uploaded with same size
        try:
            props = blob_client.get_blob_properties()
            if props.size == path.stat().st_size:
                skipped += 1
                continue
        except Exception:
            pass

        with open(path, "rb") as f:
            blob_client.upload_blob(f, overwrite=True)

        size_kb = path.stat().st_size // 1024
        print(f"  [{i+1}/{len(manifest)}] {gloss}/{video_id}  {size_kb} KB")
        uploaded += 1

    # Upload manifest so Databricks can read it
    manifest_blob = container_client.get_blob_client("videos/manifest.json")
    with open(MANIFEST, "rb") as f:
        manifest_blob.upload_blob(f, overwrite=True)

    print(f"\nDone: {uploaded} uploaded, {skipped} already current, {missing} missing")
    print(f"Manifest → {CONTAINER}/videos/manifest.json")


if __name__ == "__main__":
    main()
