import re
from textblob import TextBlob
import textstat
from typing import Dict, List
from pathlib import Path
import pandas as pd

def score_post(text, trending_keywords=None, hashtag_count_override: int | None = None):
    score = 0
    text_l = text.lower()

    wc = len(re.findall(r'\w+', text_l))
    if 20 <= wc <= 80:
        score += 2

    hashtags_in_text = len(re.findall(r'#\w+', text_l))
    hashtags = hashtag_count_override if hashtag_count_override is not None else hashtags_in_text
    if 1 <= hashtags <= 3:
        score += 1

    polarity = TextBlob(text_l).sentiment.polarity if TextBlob else 0.0
    score += round(polarity, 2)

    if trending_keywords:
        score += sum(1 for kw in trending_keywords if kw.lower() in text_l)

    try:
        grade = textstat.flesch_kincaid_grade(text_l) if textstat else 10
        score += max(0, 2 - (grade / 10))
    except Exception:
        pass

    if re.search(r"(discover|learn|try|join|explore|check)", text_l):
        score += 1
    if text_l.endswith("?"):
        score += 0.5

    return round(score, 2)




# Optional deps: TextBlob, textstat. Handle gracefully if missing.
try:
    from textblob import TextBlob
except Exception:
    TextBlob = None

try:
    import textstat
except Exception:
    textstat = None


def _sentiment_polarity(text: str) -> float:
    """
    Returns polarity in [-1, 1]. Falls back to 0 if TextBlob not available.
    """
    if TextBlob is None:
        return 0.0
    try:
        return float(TextBlob(text).sentiment.polarity)
    except Exception:
        return 0.0


def _readability_score(text: str) -> float:
    """
    Returns a 0..2 readability bonus (higher = easier to read).
    Uses Flesch-Kincaid grade; lower grade -> higher score.
    Falls back to 1.0 if textstat not available.
    """
    if textstat is None:
        return 1.0
    try:
        grade = textstat.flesch_kincaid_grade(text)
        # Map grade to 0..2: grade 0 => 2.0, grade 10 => 1.0, grade 20 => 0.0 (clamped)
        score = 2.0 - max(0.0, min(2.0, grade / 10.0 * 1.0))
        return float(max(0.0, min(2.0, score)))
    except Exception:
        return 1.0


def _keyword_hits(text: str, trending_keywords: List[str]) -> int:
    """
    Counts how many of the trending keywords appear (case-insensitive).
    """
    t = text.lower()
    return sum(1 for kw in (trending_keywords or []) if kw.lower() in t)


def _hashtag_count(text: str) -> int:
    return len(re.findall(r'#\w+', text))


def _word_count(text: str) -> int:
    return len(re.findall(r'\w+', text))


def optimize_post(text: str, trending_keywords: List[str], hashtag_count_override: int | None = None) -> Dict:
    wc = _word_count(text)
    hashtags_in_text = _hashtag_count(text)
    hashtags = hashtag_count_override if hashtag_count_override is not None else hashtags_in_text
    sentiment = _sentiment_polarity(text)
    kw_hits = _keyword_hits(text, trending_keywords)
    readability = _readability_score(text)

    if 20 <= wc <= 80:
        length_bonus = 2.0
    elif 10 <= wc < 20 or 80 < wc <= 120:
        length_bonus = 1.0
    else:
        length_bonus = 0.0

    hashtag_bonus = 1.0 if 1 <= hashtags <= 3 else 0.0

    score = (
        1.5 * kw_hits
        + 1.0 * sentiment
        + 1.0 * readability
        + length_bonus
        + hashtag_bonus
    )

    return {
        "word_count": wc,
        "hashtags": hashtags,
        "sentiment": round(sentiment, 3),
        "keyword_hits": kw_hits,
        "readability_bonus": round(readability, 3),
        "length_bonus": length_bonus,
        "hashtag_bonus": hashtag_bonus,
        "final_score": round(float(score), 3),
    }


def build_scoring_summary(df: pd.DataFrame, trending_keywords: List[str]) -> pd.DataFrame:
    """Return a scored summary for each row in df using scorer helpers."""
    if df.empty:
        return df.copy()

    records = []
    keywords = trending_keywords or []
    for _, row in df.iterrows():
        text = str(row.get("generated_text", ""))
        variation = row.get("variation_no")
        topic = row.get("topic", "")
        tone = row.get("tone", "")

        score_val = score_post(text, keywords)
        feature = optimize_post(text, keywords)

        records.append(
            {
                "topic": topic,
                "tone": tone,
                "variation_no": variation,
                "generated_text": text,
                "score": score_val,
                "word_count": feature["word_count"],
                "hashtags": feature["hashtags"],
                "sentiment": feature["sentiment"],
                "keyword_hits": feature["keyword_hits"],
                "readability_bonus": feature["readability_bonus"],
                "length_bonus": feature["length_bonus"],
                "hashtag_bonus": feature["hashtag_bonus"],
                "final_score": feature["final_score"],
            }
        )

    return pd.DataFrame(records).sort_values("final_score", ascending=False).reset_index(drop=True)

def load_hashtag_counts(hashtags_dir: Path) -> dict[str, int]:
    """
    Aggregate hashtag counts per post_id from the per‑platform hashtag CSVs.
    Returns {post_id: hashtag_count}.
    """
    lookup: dict[str, int] = {}
    for csv in hashtags_dir.glob("*_hashtags.csv"):
        try:
            df = pd.read_csv(csv)
        except Exception:
            continue
        if "post_id" not in df.columns:
            continue
        counts = df.groupby("post_id").size()
        for pid, cnt in counts.items():
            lookup[str(pid)] = int(cnt)
    return lookup
