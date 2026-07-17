"""
gdrive.py

Uploads pipeline output files to a Google Drive folder in a personal Gmail
account, using OAuth (not a service account — personal Drive accounts give
service accounts no storage quota of their own, so uploads under a service
account fail there).

First run opens a browser for a one-time consent screen, then caches a
refresh token in token.json so every run after that is headless.

Setup (one-time):
    1. In Google Cloud Console: enable the Google Drive API for your project.
    2. Credentials -> Create Credentials -> OAuth client ID -> Desktop app.
    3. Download the JSON, save as credentials.json next to this file (or
       point to it with --credentials / GDRIVE_CREDENTIALS_FILE).
    4. Add credentials.json and token.json to .gitignore -- both are secrets.

Usage as a library:
    from gdrive import get_drive_service, upload_or_update_file

    service = get_drive_service()
    upload_or_update_file(service, Path("data/processed/videos_final.csv"), folder_id)
"""

import logging
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/drive.file"]
DEFAULT_CREDENTIALS_FILE = Path(__file__).parent / "credentials.json"
DEFAULT_TOKEN_FILE = Path(__file__).parent / "token.json"


def get_drive_service(
    credentials_file: Path = DEFAULT_CREDENTIALS_FILE,
    token_file: Path = DEFAULT_TOKEN_FILE,
):
    """
    Returns an authenticated Drive v3 service. Uses the cached token if
    present and valid; refreshes it silently if expired; otherwise runs
    the one-time interactive OAuth flow and caches the result.
    """
    creds = None
    if token_file.exists():
        creds = Credentials.from_authorized_user_file(str(token_file), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not credentials_file.exists():
                raise FileNotFoundError(
                    f"{credentials_file} not found. Download OAuth client "
                    f"credentials from Google Cloud Console (Desktop app "
                    f"type) and save them there."
                )
            flow = InstalledAppFlow.from_client_secrets_file(str(credentials_file), SCOPES)
            creds = flow.run_local_server(port=0)
        token_file.write_text(creds.to_json())
        logger.info(f"Saved refreshed/new token to {token_file}")

    return build("drive", "v3", credentials=creds)


def _find_existing_file(service, filename: str, folder_id: str):
    """Looks for a file with this name already in the target folder."""
    query = f"name = '{filename}' and '{folder_id}' in parents and trashed = false"
    response = service.files().list(q=query, fields="files(id, name)").execute()
    files = response.get("files", [])
    return files[0]["id"] if files else None


def upload_or_update_file(service, filepath: Path, folder_id: str):
    """
    Uploads filepath to folder_id. If a file with the same name already
    exists there, updates its content in place (new revision, same file
    ID/link) instead of creating a duplicate.
    """
    media = MediaFileUpload(str(filepath), resumable=True)
    existing_id = _find_existing_file(service, filepath.name, folder_id)

    if existing_id:
        service.files().update(fileId=existing_id, media_body=media).execute()
        logger.info(f"Updated {filepath.name} in Drive (fileId={existing_id})")
    else:
        metadata = {"name": filepath.name, "parents": [folder_id]}
        created = service.files().create(body=metadata, media_body=media, fields="id").execute()
        logger.info(f"Uploaded {filepath.name} to Drive (fileId={created['id']})")
