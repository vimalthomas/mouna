# Databricks notebook source
# MAGIC %md
# MAGIC # Setup Environment
# MAGIC
# MAGIC Install the mouna package from GitHub and configure environment

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install Package from GitHub

# COMMAND ----------

# MAGIC %pip install git+https://github.com/vimalthomas-db/mouna.git

# COMMAND ----------

# Restart Python to use the new package
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify Installation

# COMMAND ----------

import mouna
from mouna.utils.logging import setup_logger
from mouna.data.ingestion import WLASLDownloader, AzureBlobUploader

print(f"✅ Mouna version: {mouna.__version__}")
print(f"✅ Package installed successfully")

# Setup logging
setup_logger(log_level="INFO")

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
