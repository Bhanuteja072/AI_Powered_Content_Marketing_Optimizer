import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.llm.groq_generate import generate_content
from src.scorer import optimize_post, score_post


DATA_DIR = PROJECT_ROOT / "data" / "processed"
EDA_KEYWORDS_PATH = DATA_DIR / "eda_top_keywords.csv"
SENTIMENT_PATH = DATA_DIR / "eda_sentiment_summary.csv"
HASHTAGS_DIR = DATA_DIR / "hashtags"
GENERATED_POSTS_PATH = DATA_DIR / "generated_posts.csv"
OPTIMIZED_POSTS_PATH = DATA_DIR / "optimized_posts.csv"


@dataclass(frozen=True)
class GenerationContext:
    top_keywords: list[str]
    best_tone: str
    prompt_hashtags: list[str]


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path)


def load_generation_context(
    top_keywords_path: Path = EDA_KEYWORDS_PATH,
    sentiment_path: Path = SENTIMENT_PATH,
    hashtags_dir: Path = HASHTAGS_DIR,
    top_n_keywords: int | None = None,
    max_prompt_hashtags: int | None = None,
) -> GenerationContext:
    df_kw = _safe_read_csv(top_keywords_path)
    if "keyword" not in df_kw.columns:
        raise ValueError("Top keywords file must contain a 'keyword' column")
    if "avg_engagement_rate" not in df_kw.columns:
        raise ValueError("Top keywords file must contain 'avg_engagement_rate'")

    keywords_series = df_kw.sort_values("avg_engagement_rate", ascending=False)["keyword"].dropna().astype(str)
    if top_n_keywords is not None:
        keywords_series = keywords_series.head(top_n_keywords)
    keywords = keywords_series.tolist()
    if not keywords:
        raise ValueError("No keywords available for generation")

    df_tone = _safe_read_csv(sentiment_path)
    if "sentiment_label" not in df_tone.columns or "engagement_rate" not in df_tone.columns:
        raise ValueError(
            "Sentiment summary must contain 'sentiment_label' and 'engagement_rate'"
        )

    best_tone = (
        df_tone.sort_values("engagement_rate", ascending=False)
        .iloc[0]["sentiment_label"]
    )

    hashtags: list[str] = []
    if hashtags_dir.exists():
        for csv in hashtags_dir.glob("*_hashtags.csv"):
            try:
                df_h = pd.read_csv(csv)
            except Exception:
                continue
            column = "hashtag" if "hashtag" in df_h.columns else None
            if column is None and "hashtags" in df_h.columns:
                column = "hashtags"
            if column:
                hashtags.extend(df_h[column].dropna().astype(str).tolist())

    prompt_list = sorted({h.strip() for h in hashtags if h.strip()})
    if max_prompt_hashtags is not None:
        prompt_list = prompt_list[:max_prompt_hashtags]

    return GenerationContext(
        top_keywords=keywords,
        best_tone=str(best_tone),
        prompt_hashtags=prompt_list,
    )


def generate_variations(
    topic: str,
    tone: str,
    keywords: Sequence[str],
    hashtags: Sequence[str],
    num_variations: int = 3,
    max_words: int = 150,
) -> pd.DataFrame:
    generated_posts = []
    timestamp = datetime.utcnow().isoformat()
    for i in range(num_variations):
        text = generate_content(
            topic=topic,
            tone=tone,
            max_words=max_words,
            keywords=list(keywords),
            hashtags=list(hashtags),
        )
        generated_posts.append(
            {
                "topic": topic,
                "tone": tone,
                "keywords_used": ", ".join(keywords),
                "hashtags_pool": ", ".join(hashtags),
                "variation_no": i + 1,
                "generated_text": text.strip(),
                "generated_at": timestamp,
            }
        )

    return pd.DataFrame(generated_posts)


def score_posts(df_gen: pd.DataFrame, keywords: Sequence[str]) -> pd.DataFrame:
    df_scored = df_gen.copy()
    df_scored["score"] = df_scored["generated_text"].apply(
        lambda text: score_post(text, keywords)
    )
    return df_scored.sort_values("score", ascending=False).reset_index(drop=True)


def build_feature_table(df_scored: pd.DataFrame, keywords: Sequence[str]) -> pd.DataFrame:
    feature_rows = []
    for _, row in df_scored.iterrows():
        feats = optimize_post(row["generated_text"], list(keywords))
        feature_rows.append(
            {
                "topic": row["topic"],
                "tone": row["tone"],
                "keywords_used": row["keywords_used"],
                "variation_no": row["variation_no"],
                "generated_text": row["generated_text"],
                "word_count": feats["word_count"],
                "hashtags": feats["hashtags"],
                "sentiment": feats["sentiment"],
                "keyword_hits": feats["keyword_hits"],
                "readability_bonus": feats["readability_bonus"],
                "length_bonus": feats["length_bonus"],
                "hashtag_bonus": feats["hashtag_bonus"],
                "score": feats["final_score"],
                "generated_at": row.get("generated_at"),
            }
        )

    return pd.DataFrame(feature_rows).sort_values("score", ascending=False).reset_index(drop=True)


def append_with_dedupe(
    new_df: pd.DataFrame,
    path: Path,
    dedupe_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    if path.exists():
        existing = pd.read_csv(path)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df.copy()

    if dedupe_columns:
        combined = combined.drop_duplicates(subset=list(dedupe_columns), keep="last")

    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False, encoding="utf-8")
    return combined


def run_generation(
    topic: str,
    *,
    tone: str | None = None,
    context: GenerationContext | None = None,
    keywords: Sequence[str] | None = None,
    hashtags: Sequence[str] | None = None,
    num_variations: int = 3,
    max_words: int = 150,
    persist: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    context = context or load_generation_context()
    tone_to_use = tone or context.best_tone
    keywords_to_use = list(keywords or context.top_keywords)
    hashtags_to_use = list(hashtags or context.prompt_hashtags)

    df_gen = generate_variations(
        topic=topic,
        tone=tone_to_use,
        keywords=keywords_to_use,
        hashtags=hashtags_to_use,
        num_variations=num_variations,
        max_words=max_words,
    )
    df_scored = score_posts(df_gen, keywords_to_use)
    df_opt = build_feature_table(df_scored, keywords_to_use)

    if persist:
        append_with_dedupe(df_scored, GENERATED_POSTS_PATH, ["generated_text"])
        append_with_dedupe(df_opt, OPTIMIZED_POSTS_PATH, ["generated_text"])

    return df_scored, df_opt


if __name__ == "__main__":
    ctx = load_generation_context()
    topic_input = input("Enter the main topic for content generation: ").strip()
    if not topic_input:
        topic_input = "AI in Content Marketing"

    scored_df, optimized_df = run_generation(topic=topic_input, context=ctx)

    print(f"\nGenerated {len(scored_df)} posts for topic '{topic_input}'.")
    print("Top scored variation:")
    print(scored_df.iloc[0][["variation_no", "score", "generated_text"]])
    print("\nTop optimized variation:")
    print(optimized_df.iloc[0][["variation_no", "score", "generated_text"]])
