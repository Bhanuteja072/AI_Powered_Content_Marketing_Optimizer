import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path  # added


# Load environment variables from .env
load_dotenv()
BEARER_TOKEN = os.getenv("BEARER_TOKEN")

if not BEARER_TOKEN:
    raise ValueError("Bearer token not found. Please set BEARER_TOKEN in your .env file.")

# --- Configuration ---
QUERY = "content generation -is:retweet lang:en"
TOTAL_NEEDED = 10        # Number of tweets to fetch
PAGE_SIZE = 10   
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUTPUT_CSV = DATA_DIR / "twitter_search_10_results.csv"

HEADERS = {
    "Authorization": f"Bearer {BEARER_TOKEN}",
    "User-Agent": "AIContentColabScript/1.0"
}

SEARCH_URL = "https://api.twitter.com/2/tweets/search/recent"
TWEET_FIELDS = "id,text,author_id,created_at,public_metrics,lang"
USER_FIELDS = "id,name,username,public_metrics"
EXPANSIONS = "author_id"


def fetch_page(next_token=None):
    """Fetch one page of tweets from Twitter API."""
    params = {
        "query": QUERY,
        "max_results": PAGE_SIZE,
        "tweet.fields": TWEET_FIELDS,
        "user.fields": USER_FIELDS,
        "expansions": EXPANSIONS
    }
    if next_token:
        params["next_token"] = next_token

    resp = requests.get(SEARCH_URL, headers=HEADERS, params=params)
    if resp.status_code != 200:
        raise Exception(f"Request failed: {resp.status_code} {resp.text}")
    return resp.json()


def build_user_map(users):
    """Map author_id to user info."""
    mapping = {}
    for u in users or []:
        metrics = u.get("public_metrics", {})
        mapping[u["id"]] = {
            "username": u.get("username"),
            "name": u.get("name"),
            "followers": metrics.get("followers_count"),
            "following": metrics.get("following_count"),
        }
    return mapping


def fetch_tweets(total_needed):
    """Loop through pages until we reach the desired number of tweets."""
    all_rows = []
    next_token = None

    while len(all_rows) < total_needed:
        data = fetch_page(next_token)
        tweets = data.get("data", [])
        users = data.get("includes", {}).get("users", [])
        user_map = build_user_map(users)

        for t in tweets:
            metrics = t.get("public_metrics", {})
            user = user_map.get(t["author_id"], {})
            all_rows.append({
                "tweet_id": t["id"],
                "text": t["text"],
                "author_id": t["author_id"],
                "author_name": user.get("name"),
                "author_username": user.get("username"),
                "followers": user.get("followers"),
                "created_at": t.get("created_at"),
                "like_count": metrics.get("like_count"),
                "retweet_count": metrics.get("retweet_count"),
                "reply_count": metrics.get("reply_count"),
                "quote_count": metrics.get("quote_count"),
            })
            if len(all_rows) >= total_needed:
                break

        next_token = data.get("meta", {}).get("next_token")
        if not next_token:
            break

        time.sleep(15)  # Safe delay between requests

    return pd.DataFrame(all_rows)


def main():
    """Main entry point."""
    print("Fetching tweets...")
    df = fetch_tweets(TOTAL_NEEDED)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"Fetched {len(df)} tweets")
    print(f"Saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
