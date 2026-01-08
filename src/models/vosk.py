"""Vosk model utilities."""

import os
import ssl
import urllib.request
import zipfile
import certifi


BASE_VOSK_URL = "https://alphacephei.com/vosk/models"

def download_vosk_model(model_name, logger) -> str:
    """
    Download Vosk model

    Args:
        model_name: Name of the Vosk model to download
        logger: Optional logger instance

    Returns:
        Path to the model directory
    """

    data_path = os.getenv("VIAM_MODULE_DATA")
    model_dir = data_path + "/" + model_name

    if os.path.exists(model_dir):
        if logger:
            logger.info(f"Vosk model already exists at {model_dir}")
        return model_dir

    url = f"{BASE_VOSK_URL}/{model_name}.zip"
    zip_path = f"/tmp/{model_name}.zip"

    logger.debug(f"Downloading Vosk model from {url}...")

    try:
        # Use certifi's certificate bundle for SSL verification
        ssl_context = ssl.create_default_context(cafile=certifi.where())

        # Download file with SSL context
        with urllib.request.urlopen(url, context=ssl_context) as response:
            with open(zip_path, 'wb') as out_file:
                out_file.write(response.read())

        logger.debug(f"Extracting to {data_path}...")

        # Extract zip to data_path (zip contains folder named model_name)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_path)
        os.remove(zip_path)
        logger.debug(f"Vosk model downloaded successfully to {model_dir}")

        return model_dir

    except Exception as e:
        logger.error(f"Failed to download Vosk model: {e}")
        raise RuntimeError(f"Failed to download Vosk model: {e}")
