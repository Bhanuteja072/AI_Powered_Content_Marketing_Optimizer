import os
import time
import pandas as pd
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Load .env explicitly
env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
loaded = load_dotenv(env_path)
print(f"[debug] .env loaded: {loaded} path: {env_path} exists: {os.path.exists(env_path)}")

API_KEY = os.getenv("YOUTUBE_API_KEY")
print(f"[debug] Raw API_KEY repr: {repr(API_KEY)} length: {0 if API_KEY is None else len(API_KEY)}")

if not API_KEY or API_KEY == "AIzaSyDH8F1QPmH_0id-nUuypDG36rAiuvzeHxU":
    raise RuntimeError("Replace placeholder with a real YouTube Data API key in .env (YOUTUBE_API_KEY=...).")

youtube = build("youtube", "v3", developerKey=API_KEY)

QUERY = "content generation"
TARGET = 200
PAGE_SIZE = 50
SLEEP_SEC = 0.3


# ----------------------------
# FUNCTION TO FETCH VIDEO STATS
# ----------------------------
def get_video_stats(video_ids):
    if not video_ids:
        return {}
    try:
        resp = youtube.videos().list(part="statistics,snippet", id=",".join(video_ids)).execute()
    except HttpError as e:
        print(f"[warn] stats request failed: {e}")
        return {}
    out = {}
    for item in resp.get("items", []):
        stats = item.get("statistics", {})
        snippet = item.get("snippet", {})
        out[item["id"]] = {
            "channel_id": snippet.get("channelId"),
            "view_count": int(stats.get("viewCount", 0)),
            "like_count": int(stats.get("likeCount", 0)) if stats.get("likeCount") else None,
            "comment_count": int(stats.get("commentCount", 0)) if stats.get("commentCount") else None,
        }
    return out


# ----------------------------
# MAIN SEARCH LOOP
# ----------------------------
rows = []
next_token = None

while len(rows) < TARGET:
    try:
        search_resp = youtube.search().list(
            part="snippet",
            q=QUERY,
            type="video",
            maxResults=PAGE_SIZE,
            pageToken=next_token
        ).execute()
    except HttpError as e:
        print(f"[error] search failed: {e}")
        break

    video_ids = [i["id"]["videoId"] for i in search_resp.get("items", [])]
    stats_map = get_video_stats(video_ids)

    for item in search_resp.get("items", []):
        vid = item["id"]["videoId"]
        snip = item["snippet"]
        st = stats_map.get(vid, {})
        rows.append({
            "video_id": vid,
            "channel_id": st.get("channel_id"),
            "title": snip.get("title"),
            "channel_title": snip.get("channelTitle"),
            "description": snip.get("description"),
            "publish_date": snip.get("publishedAt"),
            "thumbnail_url": snip.get("thumbnails", {}).get("high", {}).get("url"),
            "view_count": st.get("view_count"),
            "like_count": st.get("like_count"),
            "comment_count": st.get("comment_count"),
        })
        if len(rows) >= TARGET:
            break

    next_token = search_resp.get("nextPageToken")
    if not next_token:
        break
    time.sleep(SLEEP_SEC)


# ----------------------------
# SAVE TO CSV
# ----------------------------
df = pd.DataFrame(rows)
out_path = os.path.join(os.path.dirname(__file__), "..", "youtube_search_200_results.csv")
df.to_csv(out_path, index=False, encoding="utf-8")

print(f"Fetched: {len(rows)} Saved: {out_path}")

