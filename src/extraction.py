"""
extraction.py

Pulls channel-level and video-level statistics for a configured set of
YouTube channels via the YouTube Data API

Refactored from extraction.ipynb into a reusable, re-runnable script:
    - Channel list lives in channels.json, not hardcoded (add a channel
      without touching code)
    - Logging instead of scattered print statements
    - Retries on transient API failures
    - Re-running appends a timestamped raw snapshot and updates a
      deduplicated "current" processed file, instead of overwriting history

Usage:
    python extraction.py
    python extraction.py --channels-file path/to/channels.json --data-dir data
"""

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import os

import gdrive

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2


def build_youtube_client(api_key: str):
    """Builds and returns an authenticated YouTube Data API v3 client."""
    return build("youtube", "v3", developerKey=api_key)


def load_channels(channels_file: Path) -> dict:
    """Loads the {channel_name: channel_id} mapping from a JSON config file."""
    with open(channels_file, "r") as f:
        channels = json.load(f)
    logger.info(f"Loaded {len(channels)} channels from {channels_file}")
    return channels


def _execute_with_retry(request, context: str):
    """Executes an API request, retrying on transient errors."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return request.execute()
        except HttpError as e:
            if attempt == MAX_RETRIES:
                logger.error(f"Failed after {MAX_RETRIES} attempts: {context}")
                raise
            wait = RETRY_BACKOFF_SECONDS * attempt
            logger.warning(
                f"API error on {context} (attempt {attempt}/{MAX_RETRIES}): "
                f"{e}. Retrying in {wait}s..."
            )
            time.sleep(wait)


def get_channel_stats(youtube, channel_ids: list) -> pd.DataFrame:
    """
    Retrieves statistics and information about the specified YouTube channels.

    Parameters:
        youtube: The YouTube API resource object.
        channel_ids (list): A list of channel IDs for which statistics are
            to be retrieved.

    Returns:
        pandas.DataFrame: channelName, subscribers, views, totalVideos,
        playlistId.
    """
    all_data = []

    request = youtube.channels().list(
        part="snippet,contentDetails,statistics", id=",".join(channel_ids)
    )
    response = _execute_with_retry(request, "channels().list")

    for item in response["items"]:
        data = {
            "channelName": item["snippet"]["title"],
            "subscribers": item["statistics"]["subscriberCount"],
            "views": item["statistics"]["viewCount"],
            "totalVideos": item["statistics"]["videoCount"],
            "playlistId": item["contentDetails"]["relatedPlaylists"]["uploads"],
        }
        all_data.append(data)

    logger.info(f"Pulled channel stats for {len(all_data)} channels")
    return pd.DataFrame(all_data)


def get_video_ids(youtube, playlist_id: str) -> list:
    """Retrieves every video ID in a channel's uploads playlist, paginated."""
    video_ids = []
    next_page_token = None

    while True:
        request = youtube.playlistItems().list(
            part="snippet,contentDetails",
            playlistId=playlist_id,
            maxResults=50,
            pageToken=next_page_token,
        )
        response = _execute_with_retry(request, f"playlistItems().list ({playlist_id})")

        for item in response["items"]:
            video_ids.append(item["contentDetails"]["videoId"])

        next_page_token = response.get("nextPageToken")
        if next_page_token is None:
            break

    return video_ids


def get_video_details(youtube, videos: list) -> pd.DataFrame:
    """
    Retrieves details of YouTube videos based on their IDs, batched in
    groups of 50 (the YouTube API limit per request).

    Parameters:
        youtube: The YouTube API resource object.
        videos (list): A list of video IDs for which details are retrieved.

    Returns:
        pandas.DataFrame: A DataFrame containing details of the specified
        YouTube videos.
    """
    all_video_info = []
    stats_to_keep = {
        "snippet": ["channelTitle", "title", "description", "tags", "publishedAt"],
        "statistics": ["viewCount", "likeCount", "commentCount"],
        "contentDetails": ["duration", "definition", "caption"],
    }

    for i in range(0, len(videos), 50):
        batch = videos[i : i + 50]
        request = youtube.videos().list(part="snippet,contentDetails,statistics", id=batch)
        response = _execute_with_retry(request, f"videos().list (batch {i}:{i + 50})")

        for video in response["items"]:
            video_info = {"videoId": video["id"]}
            for part, fields in stats_to_keep.items():
                for field in fields:
                    video_info[field] = video.get(part, {}).get(field)
            all_video_info.append(video_info)

    return pd.DataFrame(all_video_info)


def extract_all(youtube, channels: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Runs the full extraction pipeline for all configured channels.

    Returns:
        (video_df, channel_stats): both stamped with a pulled_at timestamp.
    """
    pulled_at = datetime.now()

    channel_stats = get_channel_stats(youtube, list(channels.values()))

    video_ids = {
        row["channelName"]: get_video_ids(youtube, row["playlistId"])
        for _, row in channel_stats.iterrows()
    }
    logger.info(f"Retrieved video IDs: { {k: len(v) for k, v in video_ids.items()} }")

    video_dfs = {name: get_video_details(youtube, ids) for name, ids in video_ids.items()}
    video_df = pd.concat(
        [df.assign(channel=name) for name, df in video_dfs.items()], ignore_index=True
    )
    video_df = video_df.drop(columns=["channel"]).rename(columns={"title": "videoTitle"})

    channel_stats = channel_stats.drop(columns=["playlistId"])

    video_df["pulled_at"] = pulled_at
    channel_stats["pulled_at"] = pulled_at

    logger.info(f"Extraction complete: {len(video_df)} videos, {len(channel_stats)} channels")
    return video_df, channel_stats


def save_snapshot(video_df: pd.DataFrame, channel_stats: pd.DataFrame, data_dir: Path) -> tuple[Path, Path]:
    """
    Saves this run's pull in two ways:
      1. A timestamped raw snapshot under data/raw/ — preserves history,
         never overwritten.
      2. An updated deduplicated "current" file under data/processed/ —
         merges with any existing data, keeping the most recent pull per
         videoId/channelName.

    NOTE: this writes CSVs today. Swapping this function's internals for a
    SQLite (or Postgres) load is the next step on the project roadmap —
    everything upstream of this function doesn't need to change.

    Returns:
        (raw_video_path, raw_channel_path): this run's timestamped raw
        snapshot files, for callers that want to upload them elsewhere.
    """
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_video_path = raw_dir / f"videos_{stamp}.csv"
    raw_channel_path = raw_dir / f"channel_stats_{stamp}.csv"
    video_df.to_csv(raw_video_path, index=False)
    channel_stats.to_csv(raw_channel_path, index=False)
    logger.info(f"Saved raw snapshot to {raw_dir}")

    video_path = processed_dir / "videos.csv"
    channel_path = processed_dir / "channel_stats.csv"

    if video_path.exists():
        existing = pd.read_csv(video_path, parse_dates=["pulled_at"])
        combined = pd.concat([existing, video_df], ignore_index=True)
        combined = combined.sort_values("pulled_at").drop_duplicates(
            subset="videoId", keep="last"
        )
    else:
        combined = video_df
    combined.to_csv(video_path, index=False)

    if channel_path.exists():
        existing = pd.read_csv(channel_path, parse_dates=["pulled_at"])
        combined_channels = pd.concat([existing, channel_stats], ignore_index=True)
        combined_channels = combined_channels.sort_values("pulled_at").drop_duplicates(
            subset="channelName", keep="last"
        )
    else:
        combined_channels = channel_stats
    combined_channels.to_csv(channel_path, index=False)

    logger.info(
        f"Updated processed files: {video_path} ({len(combined)} videos), "
        f"{channel_path} ({len(combined_channels)} channels)"
    )

    return raw_video_path, raw_channel_path


def main():
    parser = argparse.ArgumentParser(description="Pull YouTube channel/video stats.")
    parser.add_argument(
        "--channels-file",
        type=Path,
        default=Path(__file__).parent / "channels.json",
        help="Path to JSON file mapping channel names to channel IDs.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data",
        help="Root data directory (expects/creates raw/ and processed/ subfolders).",
    )
    parser.add_argument("--upload-drive", action="store_true", help="Also upload this run's raw snapshot to Google Drive.")
    parser.add_argument(
        "--drive-folder-id",
        type=str,
        default=None,
        help="Google Drive folder ID to upload raw snapshots into. Falls back to DRIVE_RAW_FOLDER_ID in .env.",
    )
    args = parser.parse_args()

    load_dotenv()
    if args.drive_folder_id is None:
        args.drive_folder_id = os.environ.get("DRIVE_RAW_FOLDER_ID")
    api_key = os.environ.get("API_KEY")
    if not api_key:
        raise RuntimeError("API_KEY not found. Add it to your .env file.")

    youtube = build_youtube_client(api_key)
    channels = load_channels(args.channels_file)

    video_df, channel_stats = extract_all(youtube, channels)
    raw_video_path, raw_channel_path = save_snapshot(video_df, channel_stats, args.data_dir)

    if args.upload_drive:
        if not args.drive_folder_id:
            raise SystemExit("--upload-drive requires --drive-folder-id (or DRIVE_RAW_FOLDER_ID in .env)")
        service = gdrive.get_drive_service()
        gdrive.upload_or_update_file(service, raw_video_path, args.drive_folder_id)
        gdrive.upload_or_update_file(service, raw_channel_path, args.drive_folder_id)


if __name__ == "__main__":
    main()
