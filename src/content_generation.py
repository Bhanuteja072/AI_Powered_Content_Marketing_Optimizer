import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import pandas as pd
from src.llm.groq_generate import generate_content
from src.scorer import score_post
from src.scorer import optimize_post


eda_keywords = Path("../data/processed/eda_top_keywords.csv")
sentiment_file = Path("../data/processed/eda_sentiment_summary.csv")
hashtags_dir = Path("../data/processed/hashtags")

output_file = Path("../data/processed/generated_posts.csv")

df_kw = pd.read_csv(eda_keywords)
df_tone = pd.read_csv(sentiment_file)


top_keywords = (
    df_kw.sort_values("avg_engagement_rate", ascending=False)
    .head(25)["keyword"]
    .tolist()
)
# --- Best performing tone from sentiment analysis ---
best_tone = (
    df_tone.sort_values("engagement_rate", ascending=False)
    .iloc[0]["sentiment_label"]
)

hashtag_list: list[str] = []
for csv in hashtags_dir.glob("*_hashtags.csv"):
    try:
        df_h = pd.read_csv(csv)
        if "hashtag" in df_h.columns:
            hashtag_list.extend(df_h["hashtag"].dropna().astype(str).tolist())
    except Exception:
        pass
# Normalize and dedupe
hashtag_list = sorted({h.strip() for h in hashtag_list if h.strip()})
# Keep a manageable subset for the prompt
prompt_hashtags = hashtag_list[:30]


print("🎯 Top Topics:", top_keywords)
print("🗣️  Best Performing Tone:", best_tone)
print(f"🔖 Loaded {len(hashtag_list)} hashtags (using {len(prompt_hashtags)} in prompt)")


topic = input("Enter the main topic for content generation: ").strip()
if not topic:
    topic = "AI in Content Marketing"  # default fallback

print("\n🎯 Selected Topic:", topic)
print("🔥 Top Keywords:", top_keywords)
print("🗣️  Best Tone:", best_tone)




# --- Generate multiple variations for A/B testing ---
num_variations = 3
generated_posts = []

for i in range(num_variations):
    text = generate_content(
        topic=topic,
        tone=best_tone,
        max_words=150,
        keywords=top_keywords,
        hashtags=prompt_hashtags,

    )   # call your LLM wrapper
    generated_posts.append({
        "topic": topic,
        "tone": best_tone,
        "keywords_used": ", ".join(top_keywords),
        "hashtags_pool": ", ".join(prompt_hashtags),

        "variation_no": i + 1,
        "generated_text": text.strip()
    })

# --- Save results ---
df_gen = pd.DataFrame(generated_posts)
df_gen.to_csv(output_file, index=False, encoding="utf-8")

print(f"\n✅ Generated {len(df_gen)} optimized posts saved to {output_file}")




df_gen["score"] = df_gen["generated_text"].apply(
    lambda x: score_post(x, top_keywords)
)
df_gen = df_gen.sort_values("score", ascending=False)
df_gen.to_csv(output_file, index=False, encoding="utf-8")

print("✅ Scored & ranked posts saved!")
# print(df_gen[["topic","score","generated_text"]].head())

# Print highest raw score (score_post method)
# best_raw = df_gen.iloc[0]
# print("\nBest raw scored post (score_post):")
# print(f"Variation {best_raw['variation_no']} | Score {best_raw['score']:.3f}")
# print(best_raw["generated_text"])



feature_rows = []
for _, row in df_gen.iterrows():
    feats = optimize_post(row["generated_text"], top_keywords)
    feature_rows.append({
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
    })

df_opt = pd.DataFrame(feature_rows).sort_values("score", ascending=False)

# Save a scored table (STEP 5 deliverable)
optimized_out = Path("../data/processed/optimized_posts.csv")
df_opt.to_csv(optimized_out, index=False, encoding="utf-8")

print("Saved optimized comparison table to", optimized_out)
print(df_opt[["topic", "variation_no", "keyword_hits", "sentiment", "score"]].head())

# Print highest feature-based score (optimize_post method)
best_feature = df_opt.iloc[0]
print("\nBest feature-optimized post (optimize_post):")
print(f"Variation {best_feature['variation_no']} | Score {best_feature['score']:.3f}")
print(best_feature["generated_text"])