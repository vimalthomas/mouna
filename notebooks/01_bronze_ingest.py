import os
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient

# Load environment variables from .env file
load_dotenv()

# Azure configuration
subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
resource_group = os.getenv("AZURE_RESOURCE_GROUP")
storage_account = os.getenv("AZURE_STORAGE_ACCOUNT")
region = os.getenv("AZURE_REGION")
storage_key = os.getenv("AZURE_STORAGE_KEY")

# Azure Storage Account connection string
connection_string = f"DefaultEndpointsProtocol=https;AccountName={storage_account};AccountKey={storage_key};EndpointSuffix=core.windows.net"

# Initialize BlobServiceClient
blob_service_client = BlobServiceClient.from_connection_string(connection_string)

# Create a container for sign videos
container_name = "sign-videos"
try:
    container_client = blob_service_client.create_container(container_name)
    print(f"Container '{container_name}' created successfully.")
except Exception as e:
    print(f"Error creating container: {e}")