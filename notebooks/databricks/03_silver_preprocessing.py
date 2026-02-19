# Databricks notebook source
# MAGIC %md
# MAGIC # Silver Layer: Keypoint Extraction
# MAGIC Extract MediaPipe keypoints from videos stored in the bronze Delta table
# MAGIC and write results to the silver Delta table.

# COMMAND ----------

import sys
import os
import pickle
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, BinaryType, BooleanType

# Inline sys.path — makes the mouna package importable
_nb_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
_src_path = "/Workspace/" + "/".join(_nb_path.split("/")[1:-3]) + "/src"
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

BRONZE_PATH = "abfss://sign-videos-bronze@mounastorage2025.dfs.core.windows.net/"
SILVER_PATH = "abfss://sign-videos-silver@mounastorage2025.dfs.core.windows.net/"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Videos from Bronze Delta Table

# COMMAND ----------

videos_df = spark.read.format("delta").load(f"{BRONZE_PATH}raw_videos/")
videos_pd  = videos_df.toPandas()

print(f"Videos in bronze: {len(videos_pd)}")
if videos_pd.empty:
    raise RuntimeError("No videos in bronze/raw_videos/. Re-run 02_bronze_ingest first.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Extract Keypoints (driver loop)
# MAGIC
# MAGIC Videos are in memory as bytes. Write each to /tmp for mediapipe processing,
# MAGIC then delete immediately. Native Python file I/O is allowed on serverless.

# COMMAND ----------

from mouna.data.preprocessing import KeypointExtractor

extractor = KeypointExtractor(model_complexity=0)

keypoints_schema = StructType([
    StructField("video_id",         StringType(),  True),
    StructField("gloss",            StringType(),  True),
    StructField("keypoints_pickle", BinaryType(),  True),
    StructField("num_frames",       IntegerType(), True),
    StructField("success",          BooleanType(), True),
])

results = []
for _, row in videos_pd.iterrows():
    temp_path = f"/tmp/{row['video_id']}.mp4"
    try:
        # Write video bytes to local /tmp — native Python I/O is allowed on serverless
        with open(temp_path, "wb") as f:
            f.write(bytes(row["content"]))

        kp     = extractor.extract_from_video(temp_path)
        flat   = extractor.flatten_keypoints(kp)
        normed = extractor.normalize_keypoints(flat)

        results.append({
            "video_id":         row["video_id"],
            "gloss":            row["gloss"],
            "keypoints_pickle": pickle.dumps(normed),
            "num_frames":       int(len(normed)),
            "success":          True,
        })
        print(f"  OK  {row['video_id']}  ({len(normed)} frames)")
    except Exception as e:
        results.append({
            "video_id":         row["video_id"],
            "gloss":            row["gloss"],
            "keypoints_pickle": None,
            "num_frames":       0,
            "success":          False,
        })
        print(f"  ERR {row['video_id']}: {e}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write to Silver Delta Table

# COMMAND ----------

keypoints_df = spark.createDataFrame(results, schema=keypoints_schema)
keypoints_df.write.format("delta").mode("overwrite").save(f"{SILVER_PATH}keypoints/")

success = sum(r["success"] for r in results)
print(f"\nExtracted {success}/{len(results)} keypoint sequences → silver layer")
