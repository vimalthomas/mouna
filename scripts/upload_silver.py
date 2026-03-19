#!/usr/bin/env python3
"""Upload local keypoints parquet to Azure silver container."""

import os
from pathlib import Path
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
load_dotenv()

PARQUET = Path("data/silver/keypoints/keypoints.parquet")
CONTAINER = "sign-videos-silver"
BLOB_NAME = "keypoints/keypoints.parquet"


def main():
    account = os.environ.get("AZURE_STORAGE_ACCOUNT", "mounastorage2025")
    key = os.environ["AZURE_STORAGE_KEY"]

    service = BlobServiceClient(
        account_url=f"https://{account}.blob.core.windows.net",
        credential=key,
    )
    blob_client = service.get_container_client(CONTAINER).get_blob_client(BLOB_NAME)

    size_mb = PARQUET.stat().st_size / 1_048_576
    print(f"Uploading {PARQUET} ({size_mb:.1f} MB) → {CONTAINER}/{BLOB_NAME}")
    with open(PARQUET, "rb") as f:
        blob_client.upload_blob(f, overwrite=True)
    print("Done.")


if __name__ == "__main__":
    main()
