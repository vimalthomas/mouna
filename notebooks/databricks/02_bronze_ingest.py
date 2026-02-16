# Databricks notebook source
# MAGIC %md
# MAGIC # Bronze Layer: WLASL Data Ingestion
# MAGIC
# MAGIC Download WLASL videos and save to bronze layer

# COMMAND ----------

# MAGIC %run ./01_setup_environment

# COMMAND ----------

import json
import requests
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from mouna.data.ingestion import WLASLDownloader
import tempfile
import os

spark = SparkSession.builder.getOrCreate()

BRONZE_PATH = "abfss://sign-videos-bronze@mysignstorage.dfs.core.windows.net/"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Download WLASL Metadata

# COMMAND ----------

# Use mouna's WLASL downloader
downloader = WLASLDownloader(output_dir="/tmp/wlasl")
metadata = downloader.download_metadata()

print(f"Total glosses: {len(metadata)}")
print(f"Sample: {metadata[0]}")

# Save metadata to bronze layer
metadata_json = json.dumps(metadata)
dbutils.fs.put(f"{BRONZE_PATH}metadata/wlasl_v0.3.json", metadata_json, True)

print(f"✅ Metadata saved to bronze layer")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Video Inventory

# COMMAND ----------

# Build inventory from metadata
video_records = []

for gloss_entry in metadata:
    gloss = gloss_entry.get("gloss", "")
    instances = gloss_entry.get("instances", [])

    for instance in instances:
        video_records.append({
            "gloss": gloss,
            "video_id": instance.get("video_id"),
            "url": instance.get("url"),
            "split": instance.get("split", "train"),
            "signer_id": instance.get("signer_id", -1),
            "fps": instance.get("fps", 30),
            "frame_start": instance.get("frame_start", 0),
            "frame_end": instance.get("frame_end", -1),
        })

# Create DataFrame
schema = StructType([
    StructField("gloss", StringType(), True),
    StructField("video_id", StringType(), True),
    StructField("url", StringType(), True),
    StructField("split", StringType(), True),
    StructField("signer_id", IntegerType(), True),
    StructField("fps", IntegerType(), True),
    StructField("frame_start", IntegerType(), True),
    StructField("frame_end", IntegerType(), True),
])

videos_df = spark.createDataFrame(video_records, schema=schema)

print(f"Total videos: {videos_df.count()}")
videos_df.show(5)

# Save as Delta table
videos_df.write.mode("overwrite").format("delta").save(f"{BRONZE_PATH}inventory/")

print(f"✅ Inventory saved")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Download Videos (Distributed)

# COMMAND ----------

from pyspark.sql.functions import udf
from pyspark.sql.types import BooleanType
import requests

def download_video(video_url, video_id, gloss):
    """Download a single video and upload to bronze layer"""
    if not video_url or video_url == "None":
        return False

    try:
        # Download video
        response = requests.get(video_url, timeout=30, stream=True)
        response.raise_for_status()

        # Save to temp
        temp_path = f"/tmp/{video_id}.mp4"
        with open(temp_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        # Upload to bronze
        bronze_video_path = f"{BRONZE_PATH}videos/{gloss}/{video_id}.mp4"
        dbutils.fs.cp(f"file://{temp_path}", bronze_video_path)

        # Cleanup
        os.remove(temp_path)

        return True
    except Exception as e:
        return False

# Register UDF
download_udf = udf(download_video, BooleanType())

# COMMAND ----------

# Download sample (100 videos) - change limit for full dataset
sample_size = 100
sample_df = videos_df.limit(sample_size)

print(f"Downloading {sample_size} videos...")

# Use pandas UDF for distributed download
@pandas_udf(BooleanType())
def download_video_pandas(urls, video_ids, glosses):
    import pandas as pd
    results = []
    for url, vid, gloss in zip(urls, video_ids, glosses):
        result = download_video(url, vid, gloss)
        results.append(result)
    return pd.Series(results)

# Apply download
from pyspark.sql.functions import col
result_df = sample_df.withColumn(
    "downloaded",
    download_video_pandas(col("url"), col("video_id"), col("gloss"))
)

# Show results
success_count = result_df.filter(col("downloaded") == True).count()
total_count = result_df.count()

print(f"✅ Downloaded {success_count}/{total_count} videos")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify Bronze Layer

# COMMAND ----------

# Count videos in bronze
video_count = len(dbutils.fs.ls(f"{BRONZE_PATH}videos/"))
print(f"Video directories in bronze: {video_count}")

# Show sample
sample_files = dbutils.fs.ls(f"{BRONZE_PATH}")
for f in sample_files:
    print(f"  {f.path}")

print(f"\n✅ Bronze layer ingestion complete!")
