# Sports YouTube Analytics: What Drives Engagement Across Niche Sports Channels

End-to-end data pipeline featuring 5 of my favorite sports analysis YouTube channels (Thinking Basketball, Thinking Football, Hoop Intellect, Hoop Venue, and Daniel Li) to understand what publishing patterns and content features drive views and engagement.

**[Live Dashboard →](https://public.tableau.com/app/profile/edidiong.ekpoh8227/viz/YouTube_Analytics/YouTubeAnalytics)**

## Architecture
YouTube API → Extraction script → Cleaning script → BigQuery (warehouse) 
→ Google Drive (Tableau-compatible export) → Tableau Public

The Google Drive export was included only because it 
was the only way for this script to work with my Live Dashboard 
(Using the Free Version of Tableau Public)

```
extraction.py  →  data/raw/*  +  data/processed/videos.csv, channel_stats.csv
                        │
                        ▼
transform.py   →  data/processed/videos_final.csv, title_words.csv
                        │
                        ├──→ BigQuery (videos, title_words, channel_stats)
                        └──→ Google Drive (curated CSVs)
```

`data/raw/` is timestamped and append-only, `data/processed/` holds
"current state" files that get rebuilt each run. `extraction.py`'s raw snapshots are also
pushed to Drive independently, in a separate folder from the curated outputs.

## What's in this repo
- /src/channels.json — `{channel_name: channel_id}` config for the 5 tracked channels
- /src/extraction.py — Pulls channel and video data from YouTube Data API
- /src/transform.py — Cleans and transforms data, exports to BigQuery and Drive
- /src/gdrive.py - Helps /src/extraction.py and /src/transform.py connect to personal GMail and export to Drive 
- /notebooks/analysis.ipynb — EDA, modeling, SHAP analysis (work in progress)

## Data
- Source: YouTube Data API v3
- Refresh cadence: Manual
- Key CSV files: `channel_stats.csv`, `videos_final.csv`, `title_words.csv` (raw CSVs
  aren't committed to this repo — see data dictionary below for what each contains)

### Data dictionary

**`channel_stats.csv`** — one row per tracked channel
- channelName: Channel display name 
- subscribers: Subscriber count at pull time 
- views: Lifetime channel view count 
- totalVideos: Total videos on the channel
- pulled_at: Timestamp this row was fetched 

**`videos_final.csv`** — one row per video

- videoId: YouTube video ID 
- videoTitle: Video title
- channelTitle: Channel the video belongs to 
- publishedAt: Publish timestamp 
- viewCount: Views at pull time
- likeCount: Likes at pull time
- commentCount: Comments at pull time
- definition: Video quality: `hd` or `sd`
- timeOfDay: `Morning`/`Afternoon`/`Evening`/`Night`, derived from publish hour 
- durationMins: Decimal video length in minute
- engagementRate: `(likeCount + commentCount) / viewCount`
- tagCount:  Number of tags on the video
- titleWordCount: Word count of the title 
- titleCharCount: Character count of the title 
- hasQuestion: Title contains `?`
- hasNumber: Title contains a digit 
- hasVs: Title contains "vs"/"vs."
- titleSentiment: TextBlob polarity score of the title

**`title_words.csv`** — one row per surviving word per video (titles lowercased,
stopwords and words ≤2 characters removed)

- videoId: YouTube video ID 
- word: A single word from the title 
- channelTitle: Channel the video belongs to 

## Setup / How to run

### 1. Python environment
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m textblob.download_corpora   # needed for titleSentiment
```

### 2. `.env` (project root or `src/` — wherever `load_dotenv()` looks)
```
API_KEY=your_youtube_data_api_key
BQ_PROJECT_ID=your-gcp-project-id
BQ_DATASET=youtube_analytics
DRIVE_FOLDER_ID=<folder id for curated outputs>
DRIVE_RAW_FOLDER_ID=<folder id for raw snapshots>
```
*A Drive folder's ID is the string in its URL after `/folders/`.

### 3. BigQuery
1. Create/select a GCP project — this is `BQ_PROJECT_ID`. No manual dataset
   creation needed — `transform.py` creates it on first run.
2. Authenticate locally: `gcloud auth application-default login`.
3. Works with free version

### 4. Google Drive (OAuth, personal Gmail account)
Service accounts don't have storage quota on personal Drive, so this project
uses OAuth instead (uploads count against your own quota).

1. GCP Console → **APIs & Services → Library** → enable **Google Drive API**.
2. **OAuth consent screen**: set **User Type / Audience: External** (not
   Internal — Internal is Workspace-only and throws `403: org_internal` for
   personal Gmail accounts). Leave **Publishing status: Testing**, and add
   your Gmail under **Test users**.
3. **Credentials → Create Credentials → OAuth client ID**, type **Desktop
   app**. Download the JSON, save as `src/credentials.json`.
4. First run with `--upload-drive` opens a browser for one-time consent, then
   caches `src/token.json` — every run after that is headless. Testing-status
   tokens can expire after ~7 days of inactivity; if so, delete `token.json`
   to redo the browser flow.
5. Create two Drive folders (raw snapshots, curated outputs) and put their
   IDs in `.env`.

### 5. Run the pipeline
```bash
python src/extraction.py --upload-drive
python src/transform.py --load-bigquery --upload-drive
```


## Dashboard Features
- Channel comparison via parameter-driven filter
- KPI cards
- Upload timing heatmap
- Engagement/views time series analysis

## Deeper Analysis (analysis.ipynb)
For a more exploratory dive that includes EDA, Random Forest modeling, 
and SHAP-based feature importance on what drives video performance, see the analysis notebook. This is a work in progress and will be 
expanded over time.

## Future improvements
- Scheduled refresh / automation (cron / Cloud Scheduler)
- Migrate raw extraction output into an append-only BigQuery raw dataset
- Additional channels
- Trim redundant cleaning cells from analysis.ipynb now that transform.py owns that logic
- Notebook polish (SHAP visualizations, model documentation, and niche commentary)