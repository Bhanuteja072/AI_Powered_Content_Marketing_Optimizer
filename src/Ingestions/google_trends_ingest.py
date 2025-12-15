import os
import pandas as pd
from pytrends.request import TrendReq
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUTPUT_FILE = DATA_DIR / "google_trends_selected.csv"


DATA_DIR.mkdir(parents=True, exist_ok=True)



pytrends = TrendReq(hl='en-US', tz=330)  # English language, India timezone (UTC+5:30)


KEYWORDS = [
    "content generation",
    "AI marketing",
    "social media campaigns"
]


print("\n🔍 Fetching Google Trends data for:", ", ".join(KEYWORDS))


pytrends.build_payload(KEYWORDS, cat=0, timeframe='today 3-m', geo='', gprop='')

df = pytrends.interest_over_time()


if 'isPartial' in df.columns:
    df = df.drop(columns=['isPartial'])

# Add a fetch timestamp
df["fetch_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# --- SAVE TO CSV ---
df.to_csv(OUTPUT_FILE, index=True, encoding="utf-8")

print(f"\n✅ Combined Google Trends data saved successfully!")
print(f"📁 File Location: {OUTPUT_FILE}")

# --- SHOW SAMPLE OUTPUT ---
print("\n📊 Sample Data:")
print(df.head())
