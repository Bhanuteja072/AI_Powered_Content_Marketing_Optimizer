import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

SUBREDDIT = "marketing"
QUERY = "content generation"
URL = f"https://www.reddit.com/r/{SUBREDDIT}/search.json?q={QUERY}&restrict_sr=1&limit=100&sort=new"

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUTPUT_CSV = DATA_DIR / "reddit_search_100_results.csv"

HEADERS = {"User-Agent": "AIContentResearchScript/1.0"}

resp = requests.get(URL, headers=HEADERS, timeout=30)
if resp.status_code != 200:
    print(resp.text)
    raise Exception(f"Request failed: {resp.status_code}")

posts = resp.json().get("data", {}).get("children", [])
rows = []
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