# Databricks notebook source
# MAGIC %md
# MAGIC # Bronze Layer: WLASL Data Ingestion
# MAGIC Download WLASL metadata and sample videos into the bronze ADLS container.

# COMMAND ----------

import json
import requests
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, BinaryType

BRONZE_PATH = "abfss://sign-videos-bronze@mounastorage2025.dfs.core.windows.net/"
WLASL_JSON_URL = "https://raw.githubusercontent.com/dxli94/WLASL/master/start_kit/WLASL_v0.3.json"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parameters

# COMMAND ----------

sample_size = int(dbutils.widgets.get("sample_size"))
print(f"Sample size: {sample_size}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Download WLASL Metadata

# COMMAND ----------

response = requests.get(WLASL_JSON_URL, timeout=30)
response.raise_for_status()
metadata = response.json()
print(f"Total glosses in WLASL: {len(metadata)}")

dbutils.fs.put(f"{BRONZE_PATH}metadata/wlasl_v0.3.json", json.dumps(metadata), overwrite=True)
print("Metadata saved to bronze layer")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build Video Inventory as Delta Table

# COMMAND ----------

inventory_schema = StructType([
    StructField("gloss",        StringType(),  True),
    StructField("video_id",     StringType(),  True),
    StructField("url",          StringType(),  True),
    StructField("split",        StringType(),  True),
    StructField("signer_id",    IntegerType(), True),
    StructField("fps",          IntegerType(), True),
    StructField("frame_start",  IntegerType(), True),
    StructField("frame_end",    IntegerType(), True),
])

records = [
    {
        "gloss":       entry.get("gloss", ""),
        "video_id":    inst.get("video_id"),
        "url":         inst.get("url"),
        "split":       inst.get("split", "train"),
        "signer_id":   inst.get("signer_id", -1),
        "fps":         inst.get("fps", 30),
        "frame_start": inst.get("frame_start", 0),
        "frame_end":   inst.get("frame_end", -1),
    }
    for entry in metadata
    for inst in entry.get("instances", [])
]

inventory_df = spark.createDataFrame(records, schema=inventory_schema)
print(f"Total videos in inventory: {inventory_df.count()}")
inventory_df.show(5, truncate=False)

inventory_df.write.format("delta").mode("overwrite").save(f"{BRONZE_PATH}inventory/")
print("Inventory saved as Delta table")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Download Sample Videos to Bronze
# MAGIC
# MAGIC On Databricks serverless, `dbutils.fs.cp` with `file://` is forbidden.
# MAGIC Videos are downloaded to memory and stored as binary rows in a Delta table.

# COMMAND ----------

video_schema = StructType([
    StructField("gloss",    StringType(), True),
    StructField("video_id", StringType(), True),
    StructField("content",  BinaryType(), True),
])

# Try a larger candidate pool — WLASL has many dead URLs
candidate_rows = (
    inventory_df
    .filter(inventory_df.url.isNotNull())
    .limit(sample_size * 20)
    .collect()
)

video_rows = []
for row in candidate_rows:
    if len(video_rows) >= sample_size:
        break
    try:
        resp = requests.get(row.url, timeout=60, stream=True)
        resp.raise_for_status()
        content = resp.content          # read into memory (sign language clips are small)
        if len(content) < 1024:        # skip suspiciously tiny responses
            print(f"  SKIP {row.video_id}: response too small ({len(content)} bytes)")
            continue
        video_rows.append({
            "gloss":    row.gloss,
            "video_id": row.video_id,
            "content":  bytearray(content),
        })
        print(f"  Downloaded: {row.video_id}  ({len(content):,} bytes)")
    except Exception as e:
        print(f"  SKIP {row.video_id}: {e}")

print(f"\nDownloaded {len(video_rows)}/{sample_size} videos")

if video_rows:
    videos_df = spark.createDataFrame(video_rows, schema=video_schema)
    videos_df.write.format("delta").mode("overwrite").save(f"{BRONZE_PATH}raw_videos/")
    print("Videos saved as Delta table → bronze/raw_videos/")
else:
    raise RuntimeError("No videos downloaded. Check WLASL URL availability.")
