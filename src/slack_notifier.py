import os
import pandas as pd
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from dotenv import load_dotenv
from pathlib import Path

# --- Load environment variables ---
load_dotenv()

SLACK_TOKEN = os.getenv("SLACK_BOT_TOKEN")
CHANNEL_ID = os.getenv("SLACK_CHANNEL_ID")

# --- Define data path ---
metrics_file = Path("../data/metrics/Post_performance_metrics.csv")


# --- Initialize Slack client ---
client = WebClient(token=SLACK_TOKEN)

# --- Step 1: Load recent performance data ---
df = pd.read_csv(metrics_file)

# keep one row per topic + variation (highest sentiment score wins)
df = (
    df.sort_values("sentiment_score", ascending=False)
      .drop_duplicates(subset=["topic", "variation_no"])
)

# --- Step 2: Compute a simulated performance score ---
# Combines your content heuristic score + sentiment positivity
df["performance_score"] = df["score"] + (df["sentiment_score"] * 10)

# --- Step 3: Select top 3 posts ---
top_posts = df.sort_values("performance_score", ascending=False).head(3)

# --- Step 4: Prepare message ---
message = "*📢 AI Content Optimizer Summary*\n"
message += "Here are the top 3 high-potential posts:\n\n"

for _, row in top_posts.iterrows():
    topic = row.get("topic", "N/A")
    sent = row.get("sentiment_score", 0)
    tone = row.get("tone", "N/A")
    perf = round(row.get("performance_score", 0), 2)
    preview = row.get("generated_text", "")[:150] + "..."

    message += f"🚀 *{topic}*\n🧠 Tone: {tone} | ❤️ Sentiment: {sent:.2f}\n⭐ Predicted Score: {perf}\n💬 {preview}\n\n"

# --- Step 5: Send message to Slack ---
try:
    response = client.chat_postMessage(channel=CHANNEL_ID, text=message)
    print("✅ Slack summary sent successfully!")
except SlackApiError as e:
    print(f"⚠️ Error sending message to Slack: {e.response['error']}")

