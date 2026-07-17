"""
transform.py

Cleans and transforms raw video data (data/processed/videos.csv, produced by
extraction.py) into the curated datasets used for analysis and Tableau:
  - vidoes.csv  (one row per video, cleaned + feature-engineered)
  - title_words.csv   (one row per title word, derived from videos.csv)
  - channel_stats.csv   (pass-through from extraction.py, cast + loaded as-is)

Both are optionally loaded straight into BigQuery.

Ported from analysis.ipynb's cleaning + feature-engineering cells (the
notebook's exploratory/modeling cells — charts, train/test split, SHAP —
are intentionally left out; those stay in the notebook) and from
title_words.py, which is now folded in here as extract_title_words() rather
than running as a separate script: it only needs columns transform.py
already has in memory, and the two outputs are always refreshed together.

Usage:
    python transform.py
    python transform.py --data-dir data
    python transform.py --load-bigquery --bq-project my-proj --bq-dataset youtube_analytics
    python transform.py --load-bigquery --bq-project my-proj --bq-dataset youtube_analytics --bq-location US
"""

import argparse
import logging
import re
from pathlib import Path
import ast
from google.cloud import bigquery
import isodate
import pandas as pd
from textblob import TextBlob
from dotenv import load_dotenv 
import time
import os

import gdrive

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

FINAL_COLUMNS = [
    "videoId", "videoTitle", "channelTitle", "publishedAt", "viewCount",
    "likeCount", "commentCount", "definition", "timeOfDay", "durationMins",
    "engagementRate", "tagCount", "titleWordCount", "titleCharCount",
    "hasQuestion", "hasNumber", "hasVs", "titleSentiment",
]

TITLE_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "is", "are", "this", "that", "my", "your", "i", "you",
    "it", "how", "what", "why", "from", "by", "as", "be", "was", "were",
    "do", "does", "did", "so", "if", "not", "no", "we", "us", "our",
}




def load_videos(processed_dir: Path) -> pd.DataFrame:
    path = processed_dir / "videos.csv"
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} rows from {path}")
    return df


def drop_incomplete_rows(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop(df[df["likeCount"].isnull() | df["commentCount"].isnull()].index)
    logger.info(f"Dropped {before - len(df)} rows missing likeCount/commentCount")
    return df


def cast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = ["viewCount", "likeCount", "commentCount"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce", axis=1)
    return df


def add_publish_timing(df: pd.DataFrame) -> pd.DataFrame:
    df["publishedAt"] = pd.to_datetime(df["publishedAt"], utc=True)
    df["publishedAt"] = df["publishedAt"].dt.tz_convert("America/New_York")
    df["publishDay"] = df["publishedAt"].dt.strftime("%A")

    def time_of_day(hour):
        if 5 <= hour < 12:
            return "Morning"
        elif 12 <= hour < 17:
            return "Afternoon"
        elif 17 <= hour < 21:
            return "Evening"
        return "Night"

    df["timeOfDay"] = df["publishedAt"].dt.hour.apply(time_of_day)
    return df


def add_duration_mins(df: pd.DataFrame) -> pd.DataFrame:
    df["durationSecs"] = df["duration"].apply(lambda x: isodate.parse_duration(x))
    df["durationSecs"] = df["durationSecs"].dt.total_seconds()
    df["durationMins"] = round(df["durationSecs"] / 60, 1)
    return df.drop(columns=["durationSecs", "duration"])


def add_engagement_rate(df: pd.DataFrame) -> pd.DataFrame:
    df["engagementRate"] = (df["likeCount"] + df["commentCount"]) / df["viewCount"]
    return df


def add_tag_count(df: pd.DataFrame) -> pd.DataFrame:
    def count_tags(x):
        if pd.isna(x):
            return 0
        try:
            return len(ast.literal_eval(x))
        except (ValueError, SyntaxError):
            return 0

    df["tagCount"] = df["tags"].apply(count_tags)
    return df


def add_title_features(df: pd.DataFrame) -> pd.DataFrame:
    def extract(title):
        if pd.isna(title):
            return pd.Series({
                "titleWordCount": 0, "titleCharCount": 0, "hasQuestion": 0,
                "hasNumber": 0, "hasVs": 0, "titleSentiment": 0.0,
            })
        return pd.Series({
            "titleWordCount": len(title.split()),
            "titleCharCount": len(title),
            "hasQuestion": int("?" in title),
            "hasNumber": int(bool(re.search(r"\d", title))),
            "hasVs": int(bool(re.search(r"\bvs\.?\b", title, re.IGNORECASE))),
            "titleSentiment": TextBlob(title).sentiment.polarity,
        })

    features = df["videoTitle"].apply(extract)
    return pd.concat([df, features], axis=1)


def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = drop_incomplete_rows(df)
    df = cast_numeric(df)
    df = add_publish_timing(df)
    df = add_duration_mins(df)
    df = add_engagement_rate(df)
    df = add_tag_count(df)
    df = df.drop(columns=["caption"], errors="ignore")
    df = add_title_features(df)
    logger.info(f"Feature engineering complete: {len(df)} rows")
    return df


def select_final_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df[FINAL_COLUMNS]


def clean_and_split(title: str) -> list:
    if pd.isna(title):
        return []
    cleaned = re.sub(r"[^a-z0-9\s]", "", title.lower())
    words = cleaned.split()
    return [w for w in words if w not in TITLE_STOPWORDS and len(w) > 2]


def extract_title_words(videos: pd.DataFrame) -> pd.DataFrame:
    df = videos[["videoId", "videoTitle", "channelTitle"]].copy()
    df["word"] = df["videoTitle"].apply(clean_and_split)
    title_words = df[["videoId", "channelTitle", "word"]].explode("word")
    title_words = title_words.dropna(subset=["word"])
    title_words = title_words[["videoId", "word", "channelTitle"]]
    logger.info(
        f"Title words: {len(title_words)} instances, "
        f"{title_words['word'].nunique()} unique"
    )
    return title_words


def load_channel_stats(processed_dir: Path) -> pd.DataFrame:
    path = processed_dir / "channel_stats.csv"
    df = pd.read_csv(path)
    numeric_cols = ["subscribers", "views", "totalVideos"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce", axis=1)
    logger.info(f"Loaded {len(df)} rows from {path}")
    return df

# --- output --------------------------------------------------------------

def save_local(df: pd.DataFrame, processed_dir: Path, filename: str) -> Path:
    out_path = processed_dir / filename
    df.to_csv(out_path, index=False)
    logger.info(f"Saved {out_path} ({len(df)} rows)")
    return out_path


def ensure_dataset(client, project: str, dataset: str, location: str):
    """
    Creates the dataset if it doesn't already exist. Idempotent — safe to
    call on every run. Keeps the script self-sufficient so you don't need
    to pre-create datasets by hand in the console.
    """
    from google.cloud import bigquery

    dataset_ref = bigquery.DatasetReference(project, dataset)
    ds = bigquery.Dataset(dataset_ref)
    ds.location = location
    client.create_dataset(ds, exists_ok=True)
    logger.info(f"Dataset ready: {project}.{dataset} (location={location})")


def load_to_bigquery(df: pd.DataFrame, project: str, dataset: str, table: str, location: str):
    """
    Replaces the curated table in BigQuery with this run's output
    (WRITE_TRUNCATE) — the table always reflects the current cleaned
    dataset, no manual versioning needed. Raw history should live
    separately in an append-only raw dataset, populated from extraction.py.
    """
    client = bigquery.Client(project=project)
    ensure_dataset(client, project, dataset, location)

    table_id = f"{project}.{dataset}.{table}"
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE",
        autodetect=True,
    )
    job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    job.result()
    logger.info(f"Loaded {len(df)} rows into BigQuery table {table_id}")


def main():
    start = time.time()
    print("Starting...")
    parser = argparse.ArgumentParser(description="Clean, enrich, and tokenize video data.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data",
        help="Root data directory (expects data/processed/videos.csv).",
    )
    load_dotenv()
    parser.add_argument("--load-bigquery", action="store_true", help="Also load results into BigQuery.")
    parser.add_argument("--bq-project", type=str, default=os.environ.get("BQ_PROJECT_ID"), help="GCP project ID.")
    parser.add_argument("--bq-dataset", type=str, default=os.environ.get("BQ_DATASET"), help="BigQuery dataset name.")
    parser.add_argument(
        "--bq-location",
        type=str,
        default="US",
        help="BigQuery dataset location (e.g. US, EU, us-east1). Used when creating the dataset.",
    )
    parser.add_argument("--upload-drive", action="store_true", help="Also upload results to Google Drive.")
    parser.add_argument(
        "--drive-folder-id",
        type=str,
        default=os.environ.get("DRIVE_FOLDER_ID"),
        help="Google Drive folder ID to upload into (the id in the folder's URL after /folders/).",
    )
    args = parser.parse_args()

    processed_dir = args.data_dir / "processed"

    videos_raw = load_videos(processed_dir)
    videos = select_final_columns(clean_and_engineer(videos_raw))
    title_words = extract_title_words(videos)
    channel_stats = load_channel_stats(processed_dir)

    videos_path = save_local(videos, processed_dir, "videos_final.csv")
    title_words_path = save_local(title_words, processed_dir, "title_words.csv")

    if args.load_bigquery:
        if not args.bq_project or not args.bq_dataset:
            raise SystemExit("--load-bigquery requires --bq-project and --bq-dataset")
        load_to_bigquery(videos, args.bq_project, args.bq_dataset, "videos", args.bq_location)
        load_to_bigquery(title_words, args.bq_project, args.bq_dataset, "title_words", args.bq_location)
        load_to_bigquery(channel_stats, args.bq_project, args.bq_dataset, "channel_stats", args.bq_location)

    if args.upload_drive:
        if not args.drive_folder_id:
            raise SystemExit("--upload-drive requires --drive-folder-id (or DRIVE_FOLDER_ID in .env)")
        service = gdrive.get_drive_service()
        gdrive.upload_or_update_file(service, videos_path, args.drive_folder_id)
        gdrive.upload_or_update_file(service, title_words_path, args.drive_folder_id)
        gdrive.upload_or_update_file(service, processed_dir / "channel_stats.csv", args.drive_folder_id)

    end = time.time()
    print(f"Script has finished running. {end-start} seconds elasped.")
if __name__ == "__main__":
    main()
