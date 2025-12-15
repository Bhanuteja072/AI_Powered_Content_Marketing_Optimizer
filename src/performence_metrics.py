import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

# --- Define paths ---
DATA_DIR = Path("../data/processed")
REPORTS_DIR = Path("../reports")


# Input files
optimized_file = DATA_DIR / "optimized_posts.csv"
engagement_file = DATA_DIR / "combined_engagement_data.csv"
sentiment_file = Path("../data/sentiment_analyzed/sentiment_analyzed_posts.csv")



# Output files
output_file  = Path("../data/metrics/Post_performance_metrics.csv")
bar_chart = Path("../reports/metrics/avg_engagement_by_platform.png")
scatter_plot = Path("../reports/metrics/sentiment_vs_engagement.png")
heatmap_plot = Path("../reports/metrics/keyword_performance_heatmap.png")




# --- Step 1: Load all datasets ---
df_opt = pd.read_csv(optimized_file)
df_sent = pd.read_csv(sentiment_file)

df_opt = df_opt.merge(
    df_sent[["topic", "sentiment_score", "sentiment_label"]],
    on="topic",
    how="left",
)

df_eng = pd.read_csv(engagement_file)
df_eng["topic"] = df_eng["platform"]

metrics = ["like_count", "comment_count", "share_count", "view_count"]
for col in metrics:
    if col not in df_eng.columns:
        df_eng[col] = 0

df = df_opt.merge(df_eng, on="topic", how="left")

# Prefer normalized platform; if missing, fall back to the generated topic
df["platform"] = df["platform"].fillna(df["topic"])

df["total_engagement"] = (
    df["like_count"].fillna(0)
    + df["comment_count"].fillna(0)
    + df["share_count"].fillna(0)
)
df["engagement_rate"] = df["total_engagement"] / (df["view_count"].fillna(0) + 1)
df["engagement_rate"] = df["engagement_rate"].round(4)



# --- Step 4: Compute correlations and averages ---
avg_eng_by_platform = (
    df.groupby("platform")["engagement_rate"]
    .mean()
    .sort_values(ascending=False)
)
corr_value = df["sentiment_score"].corr(df["engagement_rate"])



print("📊 Average Engagement Rate by Platform:")
print(avg_eng_by_platform)
print(f"\n💡 Correlation between Sentiment Score and Engagement Rate: {corr_value:.3f}")


# --- Step 5: Save performance metrics table ---
output_file.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_file, index=False, encoding="utf-8")
print(f"\n✅ Performance metrics saved to {output_file}")


# --- Step 6: Visualizations ---

# (a) Bar chart: average engagement by platform
if not avg_eng_by_platform.empty:
    plt.figure(figsize=(6, 4))
    avg_eng_by_platform.plot(kind="bar", color="skyblue")
    plt.title("Average Engagement Rate by Platform")
    plt.ylabel("Engagement Rate")
    plt.tight_layout()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(bar_chart)
    plt.close()
else:
    print("No engagement data to plot for platforms.")

# (b) Scatter plot: sentiment vs engagement
plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x="sentiment_score", y="engagement_rate", hue="platform", alpha=0.7)
plt.title("Sentiment vs Engagement Rate")
plt.tight_layout()
plt.savefig(scatter_plot)
plt.close()

# (c) Heatmap: keyword vs engagement
if "keywords_used" in df.columns:
    keyword_perf = df.groupby("keywords_used")["engagement_rate"].mean().reset_index()
    pivot = keyword_perf.pivot_table(values="engagement_rate", columns="keywords_used")
    plt.figure(figsize=(8, 3))
    sns.heatmap(pivot, cmap="coolwarm", cbar_kws={"label": "Avg Engagement"})
    plt.title("Keyword Performance Heatmap")
    plt.tight_layout()
    plt.savefig(heatmap_plot)
    plt.close()

print(f"📈 Charts saved to {REPORTS_DIR}")

# --- Step 7: Save refined output with selected columns ---
KEEP_COLS = [
    "topic",
    "tone",
    "variation_no",
    "generated_text",
    "score",
    "sentiment_score",
    "engagement_rate",
    "total_engagement",
]
df_out = df[KEEP_COLS].drop_duplicates()
df_out.to_csv(output_file, index=False, encoding="utf-8")
print(f"✅ Refined output saved to {output_file}")

