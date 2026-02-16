# Databricks notebook source
# MAGIC %md
# MAGIC # Silver Layer: Keypoint Extraction
# MAGIC
# MAGIC Extract MediaPipe keypoints from videos using mouna package

# COMMAND ----------

# MAGIC %run ./01_setup_environment

# COMMAND ----------

from mouna.data.preprocessing import KeypointExtractor, VideoPreprocessor
from pyspark.sql.functions import pandas_udf, col
from pyspark.sql.types import *
import pandas as pd
import numpy as np
import pickle

BRONZE_PATH = "abfss://sign-videos-bronze@mysignstorage.dfs.core.windows.net/"
SILVER_PATH = "abfss://sign-videos-silver@mysignstorage.dfs.core.windows.net/"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Video Inventory from Bronze

# COMMAND ----------

inventory_df = spark.read.format("delta").load(f"{BRONZE_PATH}inventory/")
print(f"Total videos in inventory: {inventory_df.count()}")
inventory_df.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Extract Keypoints (Distributed)

# COMMAND ----------

# Define schema for keypoints
keypoints_schema = StructType([
    StructField("video_id", StringType(), False),
    StructField("gloss", StringType(), False),
    StructField("keypoints_pickle", BinaryType(), False),
    StructField("num_frames", IntegerType(), False),
    StructField("success", BooleanType(), False),
])

@pandas_udf(keypoints_schema)
def extract_keypoints_udf(video_ids: pd.Series, glosses: pd.Series) -> pd.DataFrame:
    """Extract keypoints from videos using MediaPipe"""
    from mouna.data.preprocessing import KeypointExtractor
    import pickle

    extractor = KeypointExtractor()
    results = []

    for video_id, gloss in zip(video_ids, glosses):
        try:
            # Download video from bronze to temp
            video_path = f"{BRONZE_PATH}videos/{gloss}/{video_id}.mp4"
            temp_path = f"/tmp/{video_id}.mp4"

            dbutils.fs.cp(video_path, f"file://{temp_path}")

            # Extract keypoints
            keypoints = extractor.extract_from_video(temp_path)

            # Flatten keypoints
            flattened = extractor.flatten_keypoints(keypoints)

            # Normalize
            normalized = extractor.normalize_keypoints(flattened)

            # Serialize
            keypoints_bytes = pickle.dumps(normalized)

            results.append({
                "video_id": video_id,
                "gloss": gloss,
                "keypoints_pickle": keypoints_bytes,
                "num_frames": len(normalized),
                "success": True
            })

            # Cleanup
            os.remove(temp_path)

        except Exception as e:
            results.append({
                "video_id": video_id,
                "gloss": gloss,
                "keypoints_pickle": None,
                "num_frames": 0,
                "success": False
            })

    return pd.DataFrame(results)

# COMMAND ----------

# Process videos
keypoints_df = inventory_df.select("video_id", "gloss").groupBy(
    "video_id", "gloss"
).apply(extract_keypoints_udf)

# Show results
success_count = keypoints_df.filter(col("success") == True).count()
total = keypoints_df.count()

print(f"✅ Extracted keypoints from {success_count}/{total} videos")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save to Silver Layer

# COMMAND ----------

# Save as Delta table
keypoints_df.write.mode("overwrite").format("delta").save(f"{SILVER_PATH}keypoints/")

print(f"✅ Keypoints saved to silver layer")

# Verify
silver_df = spark.read.format("delta").load(f"{SILVER_PATH}keypoints/")
print(f"Total keypoint records: {silver_df.count()}")
silver_df.show(5)
