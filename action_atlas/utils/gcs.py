import time
from typing import Optional
import os

from loguru import logger
from google.cloud import storage
from google.oauth2 import service_account


BUCKET_NAME_TO_SERVICE_ACCOUNT_CREDENTIALS_PATH = {
    "bucket1": "path/to/credentials1.json",
}


def parse_gcs_url(gcs_url: str):
    """Parses a GCS URL into (bucket_name, blob_name)."""
    # Remove the 'gs://' prefix
    gcs_path = gcs_url.replace('gs://', '')

    # Split the path into (bucket_name, blob_name)
    path_parts = gcs_path.split('/', 1)

    if len(path_parts) > 1:
        # Both bucket_name and blob_name are present
        bucket_name, blob_name = path_parts
    elif len(path_parts) == 1:
        # Only bucket_name is present
        bucket_name = path_parts[0]
        blob_name = ''
    else:
        raise ValueError('Invalid GCS URL', gcs_url)

    return bucket_name, blob_name


def download_gcs_blob(
    blob_path: str,
    dest_fpath: str,
    credentials_path: Optional[str] = None,
    check_existence: bool = True,
    max_retries: int = 5,
    verbose: bool = True
) -> int:
    """Downloads a blob from a Google Cloud Storage bucket to a local file.
    Args:
        bucket_name (str): The name of the GCS bucket.
        blob_name (str): The name of the blob in the GCS bucket.
        dest_fpath (str): The local destination file path.
        credentials_path (Optional[str]): Path to the service account credentials file.
        check_existence (bool): Whether to check if the blob exists before downloading.
        max_retries (int): Maximum number of retries for downloading.
        verbose (bool): Whether to log detailed information.

    Returns:
        int: 0 if successful, -1 if failed.
    """
    bucket_name, blob_name = parse_gcs_url(blob_path)
    credentials_path = BUCKET_NAME_TO_SERVICE_ACCOUNT_CREDENTIALS_PATH.get(bucket_name)
    if not credentials_path:
        logger.error(f"No credentials path found for bucket {bucket_name}.")
        return -1

    for attempt in range(max_retries):
        try:
            credentials = service_account.Credentials.from_service_account_file(credentials_path)
            storage_client = storage.Client(credentials=credentials)
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            if check_existence and not blob.exists():
                logger.warning(f"Blob {blob_name} does not exist in bucket {bucket_name}.")
                return -1

            blob.download_to_filename(dest_fpath)
            if verbose:
                logger.info(f"Blob {blob_path} downloaded to {dest_fpath}.")
            return 0
        except Exception as e:
            logger.warning(f"Error downloading blob {blob_name} from bucket {bucket_name}: {e}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying downloading blob {blob_name} from bucket {bucket_name}... (Attempt {attempt + 2}/{max_retries})")
                time.sleep(5)
            else:
                logger.error(f"Failed to download blob {blob_name} from bucket {bucket_name} after {max_retries} attempts.")
                return -1
    return -1


def upload_gcs_blob(
    source_file_name: str,
    dest_blob_path: str,
    max_retries=5,
    verbose: bool=False,
    remove_original_file: bool=False
):
    """
    Uploads a file to a Google Cloud Storage bucket.

    Args:
        source_file_name (str): Path to the source file to upload.
        dest_blob_path (str): Destination GCS path (e.g., gs://bucket_name/blob_name).
        max_retries (int): Maximum number of retries in case of failure.
        verbose (bool): Whether to log detailed information.
        remove_original_file (bool): Whether to remove the original file after upload.

    Returns:
        int: 0 if successful, -1 if failed.
    """
    try:
        bucket_name, destination_blob_name = parse_gcs_url(dest_blob_path)
    except ValueError as e:
        logger.error(f"Error parsing GCS URL: {e}")
        return -1
    credentials_path = BUCKET_NAME_TO_SERVICE_ACCOUNT_CREDENTIALS_PATH.get(bucket_name)
    if not credentials_path:
        logger.error(f"No credentials path found for bucket {bucket_name}.")
        return -1

    credentials = service_account.Credentials.from_service_account_file(credentials_path)
    storage_client = storage.Client(credentials=credentials)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    for attempt in range(max_retries):
        try:
            if verbose:
                logger.info(f"Uploading {source_file_name} to gs://{bucket_name}/{destination_blob_name}...")
            blob.upload_from_filename(source_file_name)
            if verbose:
                logger.info(f"File {source_file_name} uploaded to gs://{bucket_name}/{destination_blob_name}.")
            if remove_original_file:
                if verbose:
                    logger.info(f"Removing {source_file_name}")
                os.remove(source_file_name)
            return 0
        except Exception as e: 
            logger.info(f"Error uploading blob {destination_blob_name} to bucket {bucket_name}: {e}.")
            if attempt < max_retries - 1:
                logger.info(f"Retrying upload (Attempt {attempt + 2}/{max_retries})...")
                time.sleep(5)

    logger.error(f"Failed to upload blob {destination_blob_name} to bucket {bucket_name} after {max_retries} attempts.")
    return -1