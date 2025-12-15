import pandas as pd
from textblob import TextBlob
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

data_path = Path("../data/processed/")
reports_path = Path("../reports/sentiment_reports")

input_file = Path("../data/processed/generated_posts.csv")

output_file = Path("../data/sentiment_analyzed/sentiment_analyzed_posts.csv")
plot_file = reports_path / "sentiment_distribution.png"

df = pd.read_csv(input_file)
print("✅ Data loaded for sentiment analysis")  


def analyze_sentiment(text):
    blob = TextBlob(str(text))
    polarity = blob.sentiment.polarity  # range: -1 (neg) → +1 (pos)
    if polarity > 0.2:
        label = "Positive"
    elif polarity < -0.2:
        label = "Negative"
    else:
        label = "Neutral"
    return polarity, label

df["sentiment_score"], df["sentiment_label"] = zip(*df["generated_text"].apply(analyze_sentiment))


# --- Step 3: Assign dominant emotion (optional heuristic) ---
def get_emotion(score):
    if score > 0.5:
        return "Joy"
    elif 0.1 < score <= 0.5:
        return "Optimism"
    elif -0.1 <= score <= 0.1:
        return "Calm"
    elif -0.5 <= score < -0.1:
        return "Frustration"
    else:
        return "Anger"
    


df["dominant_emotion"] = df["sentiment_score"].apply(get_emotion)


# --- Step 4: Save detailed sentiment results ---
output_file.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_file, index=False, encoding="utf-8")
print(f"✅ Sentiment analysis results saved to {output_file}")

# --- Step 5: Plot and save sentiment distribution chart ---
sentiment_counts = df["sentiment_label"].value_counts()

plt.figure(figsize=(6, 4))
sentiment_counts.plot(kind="bar", color=["green", "gray", "red"])
plt.title("Sentiment Distribution of Generated Posts")
plt.xlabel("Sentiment")
plt.ylabel("Number of Posts")
plt.tight_layout()

reports_path.mkdir(parents=True, exist_ok=True)
plt.savefig(plot_file)
plt.close()

print(f"📊 Sentiment distribution chart saved to {plot_file}")