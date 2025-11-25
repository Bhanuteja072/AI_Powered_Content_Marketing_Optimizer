import requests, pandas as pd
from pathlib import Path

subreddit = "marketing"
query = "content generation"
url = f"https://www.reddit.com/r/{subreddit}/search.json?q={query}&restrict_sr=1&limit=100&sort=new"
# Replace absolute path with project-relative path
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUTPUT_CSV = DATA_DIR / "reddit_search_100_results.csv"
headers = {"User-Agent": "AIContentResearchScript/1.0"}
resp = requests.get(url, headers=headers)
if resp.status_code != 200:
    print(resp.text)
    raise Exception(f"Request failed: {resp.status_code}")

posts = resp.json()["data"]["children"]
rows = []
for post in posts:
    data = post["data"]
    rows.append({
        "title": data.get("title"),
        "author": data.get("author"),
        "score": data.get("score"),
        "num_comments": data.get("num_comments"),
        "created_utc": data.get("created_utc"),
        "url": "https://reddit.com" + data.get("permalink")
    })

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
print(f"Saved {len(df)} posts to {OUTPUT_CSV}")