import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from IPython.display import display
import re
from collections import Counter
import nltk
from typing import Any
from nltk.corpus import stopwords

nltk.download("stopwords")

data_path = Path("../data/processed/combined_engagement_data.csv")
processed_dir = Path("../data/processed")
processed_dir.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(data_path)

print("Shape:", df.shape)

df.loc[df["platform"] == "google_trends", "text"] = ""
df = df.dropna(subset=["text"])
print("Shape after dropping missing text:", df.shape)

for col in ["like_count", "comment_count", "share_count", "view_count"]:
    if col in df.columns:
        df[col] = df[col].fillna(0)

df["total_engagement"] = df["like_count"] + df["comment_count"] + df["share_count"]
df["engagement_rate"] = df["total_engagement"] / df["view_count"].replace(0, np.nan)
df["engagement_rate"] = df["engagement_rate"].fillna(0)

df.to_csv(data_path, index=False)
print(f"💾 Updated engagement metrics saved to: {data_path.resolve()}")

stop_words = set(stopwords.words("english"))

def extract_keywords(text: str) -> list[str]:
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    return [w for w in words if w not in stop_words]

df["keywords"] = df["text"].apply(extract_keywords)

def extract_hashtags(text: str) -> list[str]:
    return re.findall(r"#\w+", text.lower())

df["hashtags"] = df["text"].apply(extract_hashtags)

hashtags_dir = processed_dir / "hashtags"
hashtags_dir.mkdir(parents=True, exist_ok=True)

hashtag_cols = ["platform", "post_id", "hashtags"]
for platform, platform_df in df[hashtag_cols].groupby("platform"):
    exploded = (
        platform_df.explode("hashtags")
        .dropna(subset=["hashtags"])
        .rename(columns={"hashtags": "hashtag"})
        .drop_duplicates()
    )
    if not exploded.empty:
        exploded.to_csv(
            hashtags_dir / f"{platform}_hashtags.csv",
            index=False,
            encoding="utf-8",
        )

print("✅ Hashtags extracted and saved by platform!")
display(df[["text", "keywords", "hashtags"]].head(10))




top_n_platform = 10
platform_keyword_rows: list[dict[str, Any]] = []

for platform, plat_df in df.groupby("platform"):
    keyword_counts = Counter([kw for kws in plat_df["keywords"] for kw in kws])
    if not keyword_counts:
        continue

    top_terms = [kw for kw, _ in keyword_counts.most_common(top_n_platform)]
    for term in top_terms:
        mask = plat_df["keywords"].apply(lambda kws, t=term: t in kws)
        platform_keyword_rows.append(
            {
                "platform": platform,
                "keyword": term,
                "avg_engagement_rate": plat_df.loc[mask, "engagement_rate"].mean(),
                "count": keyword_counts[term],
            }
        )

platform_keyword_df = pd.DataFrame(platform_keyword_rows)
platform_keyword_df.to_csv(processed_dir / "eda_platform_keywords.csv", index=False, encoding="utf-8")

all_keywords = [word for words in df["keywords"] for word in words]
keyword_counts = Counter(all_keywords)
top_keywords = [w for w, _ in keyword_counts.most_common(50)]

keyword_stats = []
for word in top_keywords:
    mask = df["keywords"].apply(lambda x: word in x)
    avg_eng = df.loc[mask, "engagement_rate"].mean()
    keyword_stats.append((word, avg_eng, keyword_counts[word]))

df_keywords = pd.DataFrame(keyword_stats, columns=["keyword", "avg_engagement_rate", "count"])
df_keywords = df_keywords.sort_values("avg_engagement_rate", ascending=False)
print("✅ Top Keywords by Engagement Rate")
display(df_keywords.head(10))





keyword_trends = {
    "content_generation": r"\b(content\s+generation|create\s+content)\b",
    "AI_marketing": r"\b(ai\s+marketing|ai-powered\s+marketing|ai\s+automation)\b",
    "social_media_campaigns": r"\bsocial\s+media\s+(campaign|campaigns)\b",
}

def flag_trend(row_text: str, row_tags: list[str], pattern: str) -> int:
    text_hit = bool(re.search(pattern, row_text, flags=re.IGNORECASE))
    tags_hit = any(re.search(pattern, tag, flags=re.IGNORECASE) for tag in row_tags)
    return int(text_hit or tags_hit)

for col, pattern in keyword_trends.items():
    df[col] = df.apply(
        lambda row: flag_trend(row["text"], row["hashtags"], pattern),
        axis=1
    )

trend_cols = [col for col in keyword_trends if df[col].sum() > 0]
if trend_cols:
    corr = df[trend_cols + ["engagement_rate"]].corr()
    print("✅ Correlation between trend flags and engagement rate:")
    display(corr)

    plt.figure(figsize=(6, 4))
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlation between Trend Mentions and Engagement Rate")
    plt.tight_layout()

    report_path = Path("../reports")
    report_path.mkdir(parents=True, exist_ok=True)
    plot_file = report_path / "trend_corr_heatmap.png"
    plt.savefig(plot_file, dpi=300)
    print(f"✅ Heatmap saved to {plot_file.resolve()}")
else:
    print("⚠️ No trend keyword columns available.")

report_path = Path("../reports")
report_path.mkdir(parents=True, exist_ok=True)


# 1️⃣ Engagement distribution
plt.figure(figsize=(6, 4))
sns.histplot(df["engagement_rate"], bins=30, color="skyblue")
plt.title("Engagement Rate Distribution")
plt.tight_layout()
plt.savefig(report_path / "engagement_rate_distribution.png", dpi=300)
plt.close()

# 2️⃣ Top keywords by engagement
plt.figure(figsize=(10, 5))
sns.barplot(
    data=df_keywords.head(10),
    x="avg_engagement_rate",
    y="keyword",
    hue="keyword",
    palette="viridis",
    legend=False,
)
plt.title("Top 10 Keywords by Average Engagement Rate")
plt.xlabel("Avg Engagement Rate")
plt.ylabel("Keyword")
plt.tight_layout()
plt.savefig(report_path / "top_keywords_engagement.png", dpi=300)
plt.close()

# 3️⃣ Platform-wise engagement
plt.figure(figsize=(6, 4))
sns.boxplot(
    data=df,
    x="platform",
    y="engagement_rate",
    hue="platform",
    palette="Set2",
    legend=False,
)
plt.title("Engagement Rate by Platform")
plt.tight_layout()
plt.savefig(report_path / "platform_engagement_boxplot.png", dpi=300)
plt.close()


# Save top keywords and correlation data
df_keywords.to_csv(processed_dir / "eda_top_keywords.csv", index=False, encoding="utf-8")

print("✅ Analysis results saved in data/processed/")

