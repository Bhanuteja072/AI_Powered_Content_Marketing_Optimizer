import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Tuple

import pandas as pd
import yaml

WHITESPACE_RE = re.compile(r"[ \t]+")
LINEBREAK_RE = re.compile(r"\s*\n\s*")
HASHTAG_RE = re.compile(r"#([\w\d_]+)", re.UNICODE)

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
TARGET_COLUMNS = [
    "platform",
    "post_id",
    "author_id",
    "author_name",
    "posted_at",
    "text",
    "url",
    "like_count",
    "comment_count",
    "share_count",
    "view_count",
    "tags",
    "language",
    "fetch_ts",
    "source_meta",
    "text_len",
    "engagement_sum",
    "engagement_rate",
    "days_since_post",
    "sentiment",
]


def load_config() -> Dict[str, Any]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def ensure_output_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def base_row(platform: str) -> Dict[str, Any]:
    return {
        "platform": platform,
        "post_id": "",
        "author_id": "",
        "author_name": "",
        "posted_at": "",
        "text": "",
        "url": "",
        "like_count": 0,
        "comment_count": 0,
        "share_count": 0,
        "view_count": 0,
        "tags": "",
        "language": "",
        "fetch_ts": "",
        "source_meta": "",
        "text_len": 0,
        "engagement_sum": 0,
        "engagement_rate": 0.0,
        "days_since_post": "",
        "sentiment": "",
    }


def safe_int(value: Any) -> int:
    try:
        if pd.isna(value):
            return 0
        return int(float(value))
    except Exception:
        return 0


def safe_str(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def join_text(title: str, body: str, max_len: int = 3000) -> str:
    parts = [seg.strip() for seg in (title, body) if seg and isinstance(seg, str) and seg.strip()]
    combined = "\n\n".join(parts)
    return clean_text(combined, max_len=max_len)


def clean_text(text: str, max_len: int = 3000) -> str:
    if not text:
        return ""
    text = text.strip()
    text = LINEBREAK_RE.sub("\n", text)
    text = WHITESPACE_RE.sub(" ", text)
    return text[:max_len].strip()

def normalize_youtube(df: pd.DataFrame) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        out = base_row("youtube")
        vid = safe_str(row.get("video_id"))
        out["post_id"] = vid
        out["author_id"] = safe_str(row.get("channel_id"))
        out["author_name"] = safe_str(row.get("channel_title"))
        out["posted_at"] = parse_datetime(row.get("publish_date"))
        out["text"] = join_text(row.get("title"), row.get("description"))
        out["url"] = row.get("video_url") or (f"https://www.youtube.com/watch?v={vid}" if vid else "")
        out["like_count"] = safe_int(row.get("like_count"))
        out["comment_count"] = safe_int(row.get("comment_count"))
        out["view_count"] = safe_int(row.get("view_count"))
        tags_field = row.get("tags")
        if isinstance(tags_field, list):
            tags = [str(tag).lower() for tag in tags_field if tag]
        elif isinstance(tags_field, str):
            tags = [tag.strip().lower() for tag in tags_field.split("|") if tag.strip()]
        else:
            tags = []
        out["tags"] = "|".join(tags)
        out["language"] = safe_str(row.get("language")).lower()
        out["source_meta"] = json.dumps({
            "thumbnail_url": row.get("thumbnail_url"),
            "category_id": row.get("category_id"),
        })
        rows.append(out)
    return rows


# def normalize_google_trends(df: pd.DataFrame) -> List[Dict[str, Any]]:
#     rows: List[Dict[str, Any]] = []
#     if df.empty:
#         return rows

#     date_col = next((c for c in ("date", "time", "week") if c in df.columns), None)
#     fetch_col = "fetch_ts" if "fetch_ts" in df.columns else None

#     if "keyword" in df.columns:
#         long_df = df.copy()
#     else:
#         if not date_col:
#             return rows
#         value_cols = [c for c in df.columns if c not in {date_col, "isPartial", "partial", fetch_col}]
#         if not value_cols:
#             return rows
#         long_df = df.melt(id_vars=[date_col], value_vars=value_cols, var_name="keyword", value_name="interest")

#     for _, row in long_df.iterrows():
#         keyword = safe_str(row.get("keyword") or row.get("term") or row.get("query"))
#         if not keyword:
#             continue

#         out = base_row("google_trends")
#         date_value = row.get(date_col) if date_col else row.get("date")
#         iso_ts = parse_datetime(date_value) if date_value else ""
#         out["post_id"] = safe_str(f"{keyword}:{date_value}") or keyword
#         out["author_id"] = "google_trends"
#         out["author_name"] = "Google Trends"
#         out["posted_at"] = iso_ts

#         interest = row.get("interest")
#         if interest is None:
#             for candidate in ("value", "score", "popularity", keyword):
#                 if candidate in row and not pd.isna(row[candidate]):
#                     interest = row[candidate]
#                     break
#         interest = safe_int(interest)

#         out["text"] = clean_text(f"Google Trends interest for '{keyword}' on {safe_str(date_value)} is {interest}.")
#         out["url"] = ""
#         out["like_count"] = interest
#         out["view_count"] = interest
#         out["tags"] = keyword.lower()
#         out["language"] = "en"
#         out["fetch_ts"] = safe_str(row.get(fetch_col)) if fetch_col else ""
#         out["source_meta"] = json.dumps({
#             "keyword": keyword,
#             "interest": interest,
#             "raw_date": safe_str(date_value),
#         })
#         rows.append(out)

#     return rows

def normalize_twitter(df: pd.DataFrame) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        out = base_row("twitter")
        tweet_id = safe_str(row.get("tweet_id"))
        username = safe_str(row.get("author_username"))
        out["post_id"] = tweet_id
        out["author_id"] = safe_str(row.get("author_id"))
        out["author_name"] = username or safe_str(row.get("author_name"))
        out["posted_at"] = parse_datetime(row.get("created_at"))
        text = row.get("text") or ""
        out["text"] = clean_text(text)
        if tweet_id and username:
            out["url"] = f"https://twitter.com/{username}/status/{tweet_id}"
        out["like_count"] = safe_int(row.get("like_count"))
        out["comment_count"] = safe_int(row.get("reply_count"))
        retweets = safe_int(row.get("retweet_count"))
        quotes = safe_int(row.get("quote_count"))
        out["share_count"] = retweets + quotes
        hashtags = [match.group(1).lower() for match in HASHTAG_RE.finditer(text)]
        out["tags"] = "|".join(hashtags)
        out["language"] = (row.get("lang") or "").strip().lower()
        followers = safe_int(row.get("author_followers"))
        out["_followers"] = followers
        out["source_meta"] = json.dumps({"followers": followers})
        rows.append(out)
    return rows


def parse_datetime(value: Any) -> str:
    if not value or pd.isna(value):
        return ""
    try:
        return pd.to_datetime(value, utc=True).isoformat()
    except Exception:
        return ""


def epoch_to_iso(value: Any) -> str:
    try:
        if pd.isna(value):
            return ""
        return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat()
    except Exception:
        return ""


# def normalize_reddit(df: pd.DataFrame) -> List[Dict[str, Any]]:
#     rows: List[Dict[str, Any]] = []
#     for _, row in df.iterrows():
#         out = base_row("reddit")
#         out["post_id"] = safe_str(row.get("id"))
#         author = safe_str(row.get("author"))
#         out["author_id"] = author.strip()
#         out["author_name"] = author.strip()
#         out["posted_at"] = epoch_to_iso(row.get("created_utc"))
#         out["text"] = join_text(row.get("title"), row.get("selftext"))
#         out["url"] = (row.get("url") or row.get("permalink") or "").strip()
#         out["like_count"] = safe_int(row.get("ups"))
#         out["comment_count"] = safe_int(row.get("num_comments"))
#         subreddit = (row.get("subreddit") or "").strip()
#         tags = []
#         if subreddit:
#             tags.append(f"subreddit:{subreddit.lower()}")
#         selftext = safe_str(row.get("selftext"))
#         hashtags = [match.group(1).lower() for match in HASHTAG_RE.finditer(selftext)]
#         tags.extend(hashtags)
#         out["tags"] = "|".join(tags)
#         out["language"] = (row.get("language") or "").strip().lower()
#         out["source_meta"] = json.dumps({
#             "subreddit": subreddit,
#         })
#         rows.append(out)
#     return rows



def normalize_reddit(df: pd.DataFrame) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        out = base_row("reddit")
        out["post_id"] = safe_str(row.get("id"))
        author = safe_str(row.get("author"))
        out["author_id"] = author.strip()
        out["author_name"] = author.strip()
        out["posted_at"] = epoch_to_iso(row.get("created_utc"))
        out["text"] = join_text(row.get("title"), row.get("selftext"))
        out["url"] = (row.get("url") or row.get("permalink") or "").strip()

        out["like_count"] = safe_int(row.get("ups"))
        out["comment_count"] = safe_int(row.get("num_comments"))

        crosspost_list = row.get("crosspost_parent_list")
        crosspost_count = len(crosspost_list) if isinstance(crosspost_list, list) else 0
        out["share_count"] = safe_int(row.get("num_crossposts")) or crosspost_count

        derived_views = out["like_count"] + out["comment_count"] + out["share_count"]
        out["view_count"] = safe_int(row.get("view_count")) or max(derived_views, 1)

        subreddit = (row.get("subreddit") or "").strip()
        tags = []
        if subreddit:
            tags.append(f"subreddit:{subreddit.lower()}")
        selftext = safe_str(row.get("selftext"))
        hashtags = [match.group(1).lower() for match in HASHTAG_RE.finditer(selftext)]
        tags.extend(hashtags)
        out["tags"] = "|".join(tags)
        out["language"] = (row.get("language") or "").strip().lower()
        out["source_meta"] = json.dumps({
            "subreddit": subreddit,
        })
        rows.append(out)
    return rows

def normalize_pinterest(df: pd.DataFrame) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        out = base_row("pinterest")
        out["post_id"] = safe_str(row.get("pin_id"))
        author = safe_str(row.get("author"))
        out["author_id"] = author
        out["author_name"] = author
        out["posted_at"] = parse_datetime(row.get("created_at"))
        out["text"] = join_text(row.get("title"), row.get("description"))
        out["url"] = (row.get("link") or row.get("url") or "").strip()
        out["comment_count"] = safe_int(row.get("comment_count"))
        out["share_count"] = safe_int(row.get("repin_count"))
        tags_field = row.get("tags")
        if isinstance(tags_field, list):
            tags = [str(tag).lower() for tag in tags_field if tag]
        elif isinstance(tags_field, str):
            tags = [tag.strip().lower() for tag in tags_field.split("|") if tag.strip()]
        else:
            tags = []
        out["tags"] = "|".join(tags)
        out["language"] = (row.get("language") or "").strip().lower()
        out["source_meta"] = json.dumps({})
        rows.append(out)
    return rows


def filter_rows(rows: List[Dict[str, Any]], languages: List[str], min_len: int) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    lang_set = {lang.lower() for lang in languages if lang}
    for r in rows:
        lang = (r["language"] or "").lower()
        if lang_set and lang and lang not in lang_set:
            continue
        if len((r["text"] or "").strip()) < min_len:
            continue
        filtered.append(r)
    return filtered


def dedupe_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set[Tuple[str, str]] = set()
    seen_text: set[Tuple[str, str, str]] = set()
    deduped: List[Dict[str, Any]] = []
    for r in rows:
        key = (r["platform"], r["post_id"])
        if key in seen:
            continue
        text_key = (r["platform"], r["text"], r["posted_at"])
        if text_key in seen_text:
            continue
        seen.add(key)
        seen_text.add(text_key)
        deduped.append(r)
    return deduped


def load_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"[WARN] Input file missing: {path}", file=sys.stderr)
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.ParserError:
        print(f"[WARN] ParserError for {path}, retrying with python engine + on_bad_lines='skip'")
        return pd.read_csv(path, engine="python", on_bad_lines="skip")


def enrich_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    enriched: List[Dict[str, Any]] = []
    now = datetime.now(timezone.utc)
    for r in rows:
        text = (r["text"] or "").strip()
        r["text_len"] = len(text)
        like_count = safe_int(r["like_count"])
        comment_count = safe_int(r["comment_count"])
        share_count = safe_int(r["share_count"])
        r["engagement_sum"] = like_count + comment_count + share_count
        followers = safe_int(r.pop("_followers", 0))
        if followers > 0:
            r["engagement_rate"] = round(r["engagement_sum"] / max(1, followers), 6)
        else:
            r["engagement_rate"] = 0.0
        posted_at = r.get("posted_at")
        if posted_at:
            try:
                posted_dt = pd.to_datetime(posted_at, utc=True)
                if pd.notna(posted_dt):
                    r["days_since_post"] = (now - posted_dt).days
                else:
                    r["days_since_post"] = ""
            except Exception:
                r["days_since_post"] = ""
        else:
            r["days_since_post"] = ""
        r["sentiment"] = ""
        enriched.append(r)
    return enriched

PROJECT_ROOT = Path(__file__).resolve().parent.parent
def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main():
    cfg = load_config().get("normalization", {})
    languages = cfg.get("languages", ["en"])
    min_len = cfg.get("min_text_len", 15)
    raw_paths = cfg.get("raw_paths", {})
    output_path = resolve_path(cfg.get("output_path", "data/processed/combined_engagement_data.csv"))


    youtube_df = load_dataframe(resolve_path(raw_paths.get("youtube", "data/raw/youtube_search_200_results.csv")))
    twitter_df = load_dataframe(resolve_path(raw_paths.get("twitter", "data/raw/twitter_search_10_results.csv")))
    reddit_df = load_dataframe(resolve_path(raw_paths.get("reddit", "data/raw/reddit_search_100_results.csv")))
    pinterest_df = load_dataframe(resolve_path(raw_paths.get("pinterest", "data/raw/pinterest_posts_detailed.csv")))
    google_trends_df = load_dataframe(resolve_path(raw_paths.get("google_trends", "data/raw/google_trends_selected.csv")))

    rows: List[Dict[str, Any]] = []
    rows.extend(normalize_youtube(youtube_df))
    rows.extend(normalize_twitter(twitter_df))
    rows.extend(normalize_reddit(reddit_df))
    rows.extend(normalize_pinterest(pinterest_df))
    # rows.extend(normalize_google_trends(google_trends_df))

    rows = filter_rows(rows, languages, min_len)
    rows = dedupe_rows(rows)
    rows = enrich_rows(rows)

    output_df = pd.DataFrame(rows, columns=TARGET_COLUMNS).fillna("")
    for col in ["like_count", "comment_count", "share_count", "view_count", "engagement_sum"]:
        output_df[col] = output_df[col].apply(safe_int)

    ensure_output_dir(output_path)
    output_df.to_csv(output_path, index=False, encoding="utf-8")

    print("Rows per platform:")
    print(output_df["platform"].value_counts())
    total = len(output_df)
    missing_posted = (output_df["posted_at"] == "").sum()
    empty_text = (output_df["text"].str.strip() == "").sum()
    print(f"Total rows: {total}")
    print(f"Missing posted_at: {missing_posted} ({missing_posted / max(1, total):.2%})")
    print(f"Empty text: {empty_text} ({empty_text / max(1, total):.2%})")
    if not output_df.empty:
        top5 = output_df.sort_values("engagement_sum", ascending=False).head(5)
        print("Top 5 by engagement_sum:")
        print(top5[["platform", "post_id", "engagement_sum", "url"]])
    print(f"Saved combined dataset: {output_path} with {total} rows")


if __name__ == "__main__":
    main()

