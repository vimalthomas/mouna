# Databricks notebook source
# MAGIC %md
# MAGIC # Extract Keypoints (Classic Compute + Spark + MediaPipe)
# MAGIC
# MAGIC Reads videos from Azure bronze container, runs MediaPipe on each executor,
# MAGIC writes keypoints parquet to silver container.
# MAGIC
# MAGIC **Prerequisites:**
# MAGIC - Classic compute cluster with libraries: `mediapipe`, `opencv-python-headless`, `azure-storage-blob`, `loguru`
# MAGIC - Cluster env var `AZURE_STORAGE_KEY` set (or use Databricks secret below)
# MAGIC - Bronze container populated: run `scripts/upload_bronze.py` locally first

# COMMAND ----------

import os
import sys

# Make mouna package importable from the deployed bundle workspace path
_nb_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
_src_path = "/Workspace/" + "/".join(_nb_path.split("/")[1:-3]) + "/src"
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

STORAGE_ACCOUNT = "mounastorage2025"
BRONZE_PATH = f"abfss://sign-videos-bronze@{STORAGE_ACCOUNT}.dfs.core.windows.net"
SILVER_PATH = f"abfss://sign-videos-silver@{STORAGE_ACCOUNT}.dfs.core.windows.net"

# Configure ADLS access — reads from cluster env var set in databricks.yml
storage_key = os.environ.get("AZURE_STORAGE_KEY", "")
if storage_key:
    spark.conf.set(
        f"fs.azure.account.key.{STORAGE_ACCOUNT}.dfs.core.windows.net",
        storage_key,
    )
    print("Storage key configured from env var.")
else:
    print("No AZURE_STORAGE_KEY env var — assuming Unity Catalog external location.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load manifest from bronze

# COMMAND ----------

import json

manifest_path = f"{BRONZE_PATH}/videos/manifest.json"
manifest_local = "/tmp/manifest.json"
dbutils.fs.cp(manifest_path, f"file://{manifest_local}")

with open(manifest_local) as f:
    manifest = json.load(f)

print(f"Manifest entries: {len(manifest)}")

# Create a Spark DataFrame: one row per video
from pyspark.sql import Row

rows = [
    Row(video_id=e["video_id"], gloss=e["gloss"],
        blob_path=f"videos/{e['gloss']}/{e['video_id']}.mp4")
    for e in manifest
]
videos_df = spark.createDataFrame(rows)
print(f"Videos to process: {videos_df.count()}")
videos_df.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Extract keypoints via Spark mapInPandas

# COMMAND ----------

from pyspark.sql.types import (
    BinaryType, BooleanType, IntegerType, StringType, StructField, StructType,
)

output_schema = StructType([
    StructField("video_id",          StringType(),  False),
    StructField("gloss",             StringType(),  False),
    StructField("keypoints_pickle",  BinaryType(),  True),
    StructField("num_frames",        IntegerType(), False),
    StructField("success",           BooleanType(), False),
])

# Broadcast storage credentials so each executor can download from blob
_storage_account_bc = spark.sparkContext.broadcast(STORAGE_ACCOUNT)
_storage_key_bc     = spark.sparkContext.broadcast(storage_key)


def extract_batch(batch_df):
    """Run MediaPipe extraction for a partition of videos."""
    import os
    import pickle
    import tempfile

    import numpy as np
    import pandas as pd
    from azure.storage.blob import BlobServiceClient

    # Lazy import of KeypointExtractor (mediapipe init is expensive — do once per partition)
    from mouna.data.preprocessing import KeypointExtractor

    storage_account = _storage_account_bc.value
    storage_key     = _storage_key_bc.value

    blob_service = BlobServiceClient(
        account_url=f"https://{storage_account}.blob.core.windows.net",
        credential=storage_key,
    )
    container = blob_service.get_container_client("sign-videos-bronze")

    extractor = KeypointExtractor()
    results = []

    for _, row in batch_df.iterrows():
        video_id  = row["video_id"]
        gloss     = row["gloss"]
        blob_path = row["blob_path"]

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp_path = tmp.name
                blob_client = container.get_blob_client(blob_path)
                download = blob_client.download_blob()
                tmp.write(download.readall())

            kp     = extractor.extract_from_video(tmp_path)
            flat   = extractor.flatten_keypoints(kp)
            normed = extractor.normalize_keypoints(flat)

            if len(normed) == 0:
                raise ValueError("zero frames extracted")

            results.append({
                "video_id":         video_id,
                "gloss":            gloss,
                "keypoints_pickle": pickle.dumps(normed),
                "num_frames":       len(normed),
                "success":          True,
            })

        except Exception as exc:
            print(f"ERR {video_id}: {exc}")
            results.append({
                "video_id":         video_id,
                "gloss":            gloss,
                "keypoints_pickle": None,
                "num_frames":       0,
                "success":          False,
            })
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    return pd.DataFrame(results)


# Repartition to match number of workers for parallelism
num_workers = int(spark.conf.get("spark.databricks.clusterUsageTags.clusterTargetWorkers", "4"))
videos_repartitioned = videos_df.repartition(max(num_workers * 2, 8))

print(f"Extracting keypoints across {videos_repartitioned.rdd.getNumPartitions()} partitions...")
keypoints_df = videos_repartitioned.mapInPandas(extract_batch, schema=output_schema)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save to silver layer

# COMMAND ----------

silver_parquet = f"{SILVER_PATH}/keypoints/keypoints.parquet"

keypoints_df.write.mode("overwrite").parquet(silver_parquet)

# Read back for summary
result = spark.read.parquet(silver_parquet)
success = result.filter("success = true")

print(f"\nTotal rows:    {result.count()}")
print(f"Successful:    {success.count()}")
print(f"Failed:        {result.filter('success = false').count()}")
print(f"\nPer-class counts:")
success.groupBy("gloss").count().orderBy("count", ascending=False).show(110)

print(f"\nKeypoints saved to: {silver_parquet}")
