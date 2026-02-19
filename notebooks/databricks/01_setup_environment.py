# Databricks notebook source
# MAGIC %md
# MAGIC # Setup Environment
# MAGIC
# MAGIC Configure environment and verify package access

# COMMAND ----------

# MAGIC %md
# MAGIC ## Add Package to Path

# COMMAND ----------

import sys

# Resolve the src path relative to the notebook's workspace location
# Notebook is at: .../files/notebooks/databricks/<notebook>.py
# src is at:      .../files/src/
notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
bundle_files_root = "/".join(notebook_path.split("/")[:-3])  # go up past notebooks/databricks/<notebook>
src_path = f"/Workspace{bundle_files_root}/src"

if src_path not in sys.path:
    sys.path.insert(0, src_path)

print(f"✅ Added to sys.path: {src_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configure Azure Storage with Managed Identity

# COMMAND ----------

# External locations are already configured via Unity Catalog
BRONZE_PATH = "abfss://sign-videos-bronze@mysignstorage.dfs.core.windows.net/"
SILVER_PATH = "abfss://sign-videos-silver@mysignstorage.dfs.core.windows.net/"
GOLD_PATH = "abfss://sign-videos-gold@mysignstorage.dfs.core.windows.net/"

print(f"✅ Bronze layer: {BRONZE_PATH}")
print(f"✅ Silver layer: {SILVER_PATH}")
print(f"✅ Gold layer: {GOLD_PATH}")

# Test write access
test_file = f"{BRONZE_PATH}test.txt"
dbutils.fs.put(test_file, "test content", True)
print(f"✅ Write access verified")
dbutils.fs.rm(test_file)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Environment Ready!
# MAGIC
# MAGIC Next step: Run notebook 02_bronze_ingest
