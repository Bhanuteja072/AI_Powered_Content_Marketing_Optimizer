import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path

# --- Define data paths ---
st.set_page_config(page_title="AI Marketing Optimizer Dashboard", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
POSTS_FILE = DATA_DIR / "optimized_posts.csv"
SENTIMENT_FILE = DATA_DIR / "sentiment_analyzed" / "sentiment_analyzed_posts.csv"

@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)

df_posts = load_csv(POSTS_FILE)
df_sentiment = load_csv(SENTIMENT_FILE)

if df_posts.empty:
    st.error("optimized_posts.csv is missing or empty.")
    st.stop()

if not df_sentiment.empty and "topic" in df_sentiment.columns:
    df = df_posts.merge(
        df_sentiment[["topic", "sentiment_score", "sentiment_label"]],
        on="topic",
        how="left",
    )
else:
    df = df_posts.copy()

if "sentiment_score" not in df.columns:
    if "sentiment" in df.columns:
        df["sentiment_score"] = df["sentiment"]
    else:
        df["sentiment_score"] = 0.0

if "sentiment_label" not in df.columns:
    df["sentiment_label"] = "Unknown"

df["sentiment_score"] = pd.to_numeric(df["sentiment_score"], errors="coerce").fillna(0.0)

st.title("📊 AI-Powered Content Marketing Dashboard")
st.markdown("Live snapshot of generated content, sentiment, and performance trends.")

# --- KPIs ---
total_posts = len(df)
avg_sentiment = df["sentiment_score"].mean()
avg_score = df["score"].mean() if "score" in df.columns else 0.0

col1, col2, col3 = st.columns(3)
col1.metric("Total Posts Generated", total_posts)
col2.metric("Average Sentiment", f"{avg_sentiment:.3f}")
col3.metric("Average Score", f"{avg_score:.2f}")

# --- Sentiment Distribution ---
st.subheader("Sentiment Distribution")
sent_counts = df["sentiment_label"].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(5, 3))
sent_counts.plot(kind="bar", color=["green", "gray", "red"], ax=ax)
ax.set_title("Sentiment Distribution")
ax.set_xlabel("Sentiment")
ax.set_ylabel("Number of Posts")
st.pyplot(fig)

# --- Top Topics by Performance ---
if "score" in df.columns:
    st.subheader("Top Topics by Performance")
    top_topics = (
        df.groupby("topic")["score"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )

    fig2, ax2 = plt.subplots(figsize=(6, 3))
    top_topics.plot(kind="barh", color="skyblue", ax=ax2)
    ax2.set_title("Top 10 Topics by Score")
    ax2.set_xlabel("Average Score")
    ax2.set_ylabel("Topic")
    st.pyplot(fig2)
else:
    st.info("Score column missing—cannot plot top topics.")

st.markdown("---")
st.markdown("Dashboard powered by Streamlit + Matplotlib.")
