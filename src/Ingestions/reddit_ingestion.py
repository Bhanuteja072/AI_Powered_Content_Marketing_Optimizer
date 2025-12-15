import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

SUBREDDITS = ["marketing", "digitalmarketing", "artificial"]
QUERY = "AI marketing OR content optimization OR automation"
BASE_URL = "https://www.reddit.com/r/{sub}/search.json"

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUTPUT_CSV = DATA_DIR / "reddit_search_100_results.csv"

HEADERS = {"User-Agent": "AIContentResearchScript/1.0"}

rows = []
for sub in SUBREDDITS:
    params = {
        "q": QUERY,
        "restrict_sr": 1,
        "limit": 200,
        "sort": "new",
    }
    resp = requests.get(
        BASE_URL.format(sub=sub),
        headers=HEADERS,
        params=params,
        timeout=30,
    )
    resp.raise_for_status()
    posts = resp.json().get("data", {}).get("children", [])
    fetch_ts = datetime.now(timezone.utc).isoformat()

    for post in posts:
        data = post.get("data", {})
        permalink = data.get("permalink")
        rows.append({
            "id": data.get("id"),
            "title": data.get("title"),
            "selftext": data.get("selftext") or "",
            "author": data.get("author"),
            "subreddit": data.get("subreddit"),
            "created_utc": data.get("created_utc"),
            "score": data.get("score"),
            "ups": data.get("ups"),
            "num_comments": data.get("num_comments"),
            "url": f"https://reddit.com{permalink}" if permalink else data.get("url"),
            "like_count": data.get("ups", 0),
            "comment_count": data.get("num_comments", 0),
            "share_count": data.get("crosspost_parent_list") and len(data["crosspost_parent_list"]) or 0,
            "permalink": f"https://reddit.com{permalink}" if permalink else "",
            "over_18": data.get("over_18"),
            "link_flair_text": data.get("link_flair_text"),
            "is_self": data.get("is_self"),
            "fetch_ts": fetch_ts,
        })

df = pd.DataFrame(rows)
DATA_DIR.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
print(f"Saved {len(df)} posts to {OUTPUT_CSV}")