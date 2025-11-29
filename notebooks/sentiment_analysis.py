import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
nltk.download('vader_lexicon')
from IPython.display import display


data_path = Path("../data/processed/combined_engagement_data.csv")
df = pd.read_csv(data_path)


print("✅ Data loaded for sentiment analysis")
print(df.shape)

sia = SentimentIntensityAnalyzer()
def get_sentiment_label(text):
    if not isinstance(text, str):
        return "neutral"  # handle missing or invalid text
    scores = sia.polarity_scores(text)
    compound = scores["compound"]  # main sentiment value (-1 to +1)
    if compound >= 0.05:
        return "positive"
    elif compound <= -0.05:
        return "negative"
    else:
        return "neutral"

df["sentiment_label"] = df["text"].apply(get_sentiment_label)

print("✅ Sentiment labels added!")
df["sentiment_label"].value_counts()

# Calculate average engagement per sentiment
sentiment_stats = df.groupby("sentiment_label")["engagement_rate"].mean().reset_index()
sentiment_stats = sentiment_stats.sort_values("engagement_rate", ascending=False)

print("✅ Average Engagement Rate by Sentiment:")
print(sentiment_stats)


plt.figure(figsize=(6,4))
sns.barplot(
    data=sentiment_stats,
    x="sentiment_label",
    y="engagement_rate",
    hue="sentiment_label",
    palette="coolwarm",
    legend=False,
)
plt.title("Average Engagement Rate by Sentiment")
plt.xlabel("Sentiment Type")
plt.ylabel("Average Engagement Rate")
plt.show()

output_path = Path("../data/processed/eda_sentiment_summary.csv")
sentiment_stats.to_csv(output_path, index=False, encoding="utf-8")
print(f"✅ Sentiment analysis results saved to {output_path}")

report_path = Path("../reports")
report_path.mkdir(parents=True, exist_ok=True)
plot_file = report_path / "sentiment_engagement_bar.png"

# Sentiment bar chart → save (no plt.show / emoji)
plt.figure(figsize=(6, 4))
sns.barplot(
    data=sentiment_stats,
    x="sentiment_label",
    y="engagement_rate",
    hue="sentiment_label",
    palette="coolwarm",
    legend=False,
)
plt.title("Average Engagement Rate by Sentiment")
plt.xlabel("Sentiment Type")
plt.ylabel("Average Engagement Rate")
plt.tight_layout()
plt.savefig(plot_file, dpi=300)
plt.close()
print(f"✅ Sentiment plot saved to {plot_file.resolve()}")



df["posted_at"] = pd.to_datetime(df["posted_at"], errors='coerce')
df = df.dropna(subset=["posted_at"])

dt_series = df["posted_at"]
if dt_series.dt.tz is None:
    df["posted_at"] = dt_series.dt.tz_localize("UTC", nonexistent="shift_forward", ambiguous="NaT")
else:
    df["posted_at"] = dt_series.dt.tz_convert("UTC")

# Extract day of week and hour of day
df["day_of_week"] = df["posted_at"].dt.day_name()
df["hour_of_day"] = df["posted_at"].dt.hour

print("✅ Extracted day_of_week and hour_of_day columns!")
display(df[["posted_at", "day_of_week", "hour_of_day"]].head())

# Compute average engagement rate by day and hour
engagement_by_day = df.groupby("day_of_week")["engagement_rate"].mean().reset_index()
engagement_by_hour = df.groupby("hour_of_day")["engagement_rate"].mean().reset_index()

# Ensure correct weekday order
week_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
engagement_by_day["day_of_week"] = pd.Categorical(engagement_by_day["day_of_week"], categories=week_order, ordered=True)
engagement_by_day = engagement_by_day.sort_values("day_of_week")


# Engagement by weekday/hour (saved, no emojis)
report_path = Path("../reports")
report_path.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(8, 4))
sns.barplot(
    data=engagement_by_day,
    x="day_of_week",
    y="engagement_rate",
    hue="day_of_week",
    palette="coolwarm",
    legend=False,
)
plt.title("Average Engagement Rate by Day of Week")
plt.xlabel("Day of Week")
plt.ylabel("Average Engagement Rate")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(report_path / "engagement_by_day.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 4))
sns.lineplot(
    data=engagement_by_hour,
    x="hour_of_day",
    y="engagement_rate",
    marker="o",
    color="green",
)
plt.title("Average Engagement Rate by Hour of Day (UTC)")
plt.xlabel("Hour of Day")
plt.ylabel("Average Engagement Rate")
plt.grid(True)
plt.tight_layout()
plt.savefig(report_path / "engagement_by_hour.png", dpi=300)
plt.close()



