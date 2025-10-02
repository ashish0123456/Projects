import os
from datetime import datetime
from app.core.config import settings

def ensure_storage_path():
    """Ensure the storage path exists."""
    os.makedirs(settings.STORAGE_PATH, exist_ok=True)

def save_upload(file_bytes: bytes, filename: str) -> str:
    """
    Save the uploaded bytes to the storage path and return absolute path.
    Filenames are prefixed with timestamp to avoid collisions.
    """
    ensure_storage_path()
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    safe_filename = f"{timestamp}_{filename}"
    file_path = os.path.join(settings.STORAGE_PATH, safe_filename)
    
    with open(file_path, "wb") as f:
        f.write(file_bytes)
    
    return file_path