"""Data ingestion module for downloading WLASL and uploading to Azure."""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

import requests
from loguru import logger
from tqdm import tqdm


class WLASLDownloader:
    """Download WLASL dataset videos."""

    WLASL_JSON_URL = "https://www.cihancamgoz.com/files/wlasl/WLASL_v0.3.json"
    WLASL_SPLITS_URL = "https://raw.githubusercontent.com/dxli94/WLASL/master/start_kit/WLASL_v0.3.json"

    def __init__(self, output_dir: str = "data/raw/wlasl"):
        """
        Initialize WLASL downloader.

        Args:
            output_dir: Directory to save downloaded videos.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata: Dict = {}

    def download_metadata(self) -> Dict:
        """
        Download WLASL metadata JSON.

        Returns:
            WLASL metadata dictionary.
        """
        logger.info("Downloading WLASL metadata...")
        response = requests.get(self.WLASL_SPLITS_URL)
        response.raise_for_status()
        self.metadata = response.json()

        # Save metadata locally
        metadata_path = self.output_dir / "wlasl_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)

        logger.info(f"Metadata saved to {metadata_path}")
        return self.metadata

    def download_videos(
        self, max_videos: Optional[int] = None, subset: Optional[str] = None
    ) -> List[str]:
        """
        Download WLASL videos.

        Args:
            max_videos: Maximum number of videos to download (for testing).
            subset: Download specific subset (e.g., 'WLASL100', 'WLASL300', 'WLASL2000').

        Returns:
            List of downloaded video paths.
        """
        if not self.metadata:
            self.download_metadata()

        downloaded_files = []
        video_count = 0

        logger.info(f"Starting video download to {self.output_dir}")

        for gloss_entry in tqdm(self.metadata, desc="Downloading videos"):
            gloss = gloss_entry.get("gloss", "unknown")
            instances = gloss_entry.get("instances", [])

            # Create directory for this gloss
            gloss_dir = self.output_dir / gloss
            gloss_dir.mkdir(parents=True, exist_ok=True)

            for instance in instances:
                if max_videos and video_count >= max_videos:
                    logger.info(f"Reached max_videos limit: {max_videos}")
                    return downloaded_files

                video_id = instance.get("video_id")
                url = instance.get("url")

                if not url:
                    continue

                # Download video
                video_path = gloss_dir / f"{video_id}.mp4"

                if video_path.exists():
                    logger.debug(f"Video already exists: {video_path}")
                    downloaded_files.append(str(video_path))
                    continue

                try:
                    self._download_file(url, video_path)
                    downloaded_files.append(str(video_path))
                    video_count += 1
                except Exception as e:
                    logger.error(f"Failed to download {url}: {e}")

        logger.info(f"Downloaded {len(downloaded_files)} videos")
        return downloaded_files

    def _download_file(self, url: str, output_path: Path) -> None:
        """Download a single file from URL."""
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)


class AzureBlobUploader:
    """Upload videos to Azure Blob Storage."""

    def __init__(self, config):
        """
        Initialize Azure Blob uploader.

        Args:
            config: Azure configuration with credentials.
        """
        from azure.storage.blob import BlobServiceClient
        self.config = config
        self.blob_service_client = BlobServiceClient.from_connection_string(
            config.connection_string
        )

    def create_container(self, container_name: str) -> None:
        """
        Create a blob container if it doesn't exist.

        Args:
            container_name: Name of the container to create.
        """
        try:
            container_client = self.blob_service_client.create_container(container_name)
            logger.info(f"Container '{container_name}' created successfully")
        except Exception as e:
            if "ContainerAlreadyExists" in str(e):
                logger.info(f"Container '{container_name}' already exists")
            else:
                logger.error(f"Error creating container: {e}")
                raise

    def upload_file(
        self, local_path: str, container_name: str, blob_name: Optional[str] = None
    ) -> str:
        """
        Upload a single file to Azure Blob Storage.

        Args:
            local_path: Path to local file.
            container_name: Name of the container.
            blob_name: Name for the blob. If None, uses local filename.

        Returns:
            Blob URL.
        """
        if blob_name is None:
            blob_name = Path(local_path).name

        blob_client = self.blob_service_client.get_blob_client(
            container=container_name, blob=blob_name
        )

        with open(local_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)

        logger.debug(f"Uploaded {local_path} to {container_name}/{blob_name}")
        return blob_client.url

    def upload_directory(
        self, local_dir: str, container_name: str, prefix: str = ""
    ) -> List[str]:
        """
        Upload all files in a directory to Azure Blob Storage.

        Args:
            local_dir: Path to local directory.
            container_name: Name of the container.
            prefix: Prefix for blob names (like a folder path).

        Returns:
            List of blob URLs.
        """
        local_path = Path(local_dir)
        uploaded_urls = []

        # Get all video files
        video_files = list(local_path.rglob("*.mp4"))
        logger.info(f"Found {len(video_files)} videos to upload")

        for video_file in tqdm(video_files, desc="Uploading to Azure"):
            # Maintain directory structure in blob name
            relative_path = video_file.relative_to(local_path)
            blob_name = str(Path(prefix) / relative_path) if prefix else str(relative_path)

            try:
                url = self.upload_file(str(video_file), container_name, blob_name)
                uploaded_urls.append(url)
            except Exception as e:
                logger.error(f"Failed to upload {video_file}: {e}")

        logger.info(f"Uploaded {len(uploaded_urls)} videos to {container_name}")
        return uploaded_urls

    def list_blobs(self, container_name: str, prefix: Optional[str] = None) -> List[str]:
        """
        List all blobs in a container.

        Args:
            container_name: Name of the container.
            prefix: Filter blobs by prefix.

        Returns:
            List of blob names.
        """
        container_client = self.blob_service_client.get_container_client(container_name)
        blob_list = container_client.list_blobs(name_starts_with=prefix)
        return [blob.name for blob in blob_list]


def download_and_upload_wlasl(
    azure_config: AzureConfig,
    output_dir: str = "data/raw/wlasl",
    container_name: str = "sign-videos-bronze",
    max_videos: Optional[int] = None,
) -> None:
    """
    End-to-end pipeline: Download WLASL and upload to Azure.

    Args:
        azure_config: Azure configuration.
        output_dir: Local directory for temporary storage.
        container_name: Azure container name.
        max_videos: Maximum videos to download (for testing).
    """
    # Download WLASL
    downloader = WLASLDownloader(output_dir)
    downloader.download_metadata()
    video_paths = downloader.download_videos(max_videos=max_videos)

    # Upload to Azure
    uploader = AzureBlobUploader(azure_config)
    uploader.create_container(container_name)
    uploader.upload_directory(output_dir, container_name)

    logger.info("WLASL download and upload complete!")
