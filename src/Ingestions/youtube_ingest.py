import os
from typing import List, Dict
import pandas as pd
from googleapiclient.discovery import build
from dotenv import load_dotenv
from pathlib import Path  # added

# Load environment variables from .env (located at repo root)
load_dotenv()

API_KEY = os.getenv("YOUTUBE_API_KEY")
if not API_KEY:
    raise RuntimeError("YOUTUBE_API_KEY not set in environment (.env).")
QUERIES = [
    "content generation",
    "AI content marketing",
    "AI marketing tools",
    "AI content strategy",
    "marketing automation AI",
    "AI video generation",
    "AI content analytics",
]
TOTAL_NEEDED_PER_QUERY = 50
PAGE_SIZE = 50

# Replace absolute path with project-relative path
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUTPUT_CSV = DATA_DIR / "youtube_search_200_results.csv"


def get_video_stats(youtube_client, video_ids: List[str]) -> Dict[str, Dict]:
    if not video_ids:
        return {}
    request = youtube_client.videos().list(
        part="statistics,snippet",
        id=",".join(video_ids)
    )
    response = request.execute()
    stats_data = {}
    for item in response.get("items", []):
        vid = item["id"]
        snippet = item.get("snippet", {})
        stats = item.get("statistics", {})
        stats_data[vid] = {
            "channel_id": snippet.get("channelId"),
            "view_count": stats.get("viewCount"),
            "like_count": stats.get("likeCount"),
            "comment_count": stats.get("commentCount"),
        }
    return stats_data


def fetch_videos(query: str, total_needed: int) -> pd.DataFrame:
    youtube_client = build("youtube", "v3", developerKey=API_KEY)
    all_rows = []
    next_page_token = None

    while len(all_rows) < total_needed:
        request = youtube_client.search().list(
            part="snippet",
            q=query,
            type="video",
            maxResults=PAGE_SIZE,
            pageToken=next_page_token
        )
        response = request.execute()

        items = response.get("items", [])
        video_ids = [item["id"]["videoId"] for item in items]
        stats_map = get_video_stats(youtube_client, video_ids)

        for item in items:
            vid = item["id"]["videoId"]
            snippet = item.get("snippet", {})
            stats = stats_map.get(vid, {})
            all_rows.append({
                "video_id": vid,
                "channel_id": stats.get("channel_id"),
                "title": snippet.get("title"),
                "channel_title": snippet.get("channelTitle"),
                "description": snippet.get("description"),
                "publish_date": snippet.get("publishedAt"),
                "thumbnail_url": snippet.get("thumbnails", {}).get("high", {}).get("url"),
                "view_count": stats.get("view_count"),
                "like_count": stats.get("like_count"),
                "comment_count": stats.get("comment_count"),
            })
            if len(all_rows) >= total_needed:
                break

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    return pd.DataFrame(all_rows)


def main():
    df_list = []
    for query in QUERIES:
        print(f"🔎 Fetching YouTube data for query: {query}")
        df_query = fetch_videos(query, TOTAL_NEEDED_PER_QUERY)
        df_query["search_query"] = query
        df_list.append(df_query)

    if not df_list:
        print("⚠️ No data fetched.")
        return

    final_df = pd.concat(df_list, ignore_index=True).drop_duplicates(subset=["video_id"])
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"Fetched results: {len(final_df)}")
    print(f"Saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()