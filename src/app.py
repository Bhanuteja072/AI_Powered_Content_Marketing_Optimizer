import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from textblob import TextBlob

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.append(str(PROJECT_ROOT))

from src.content_generation import (
	GenerationContext,
	load_generation_context,
	run_generation,
)
from src.scorer import build_scoring_summary


DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
SENTIMENT_DIR = DATA_DIR / "sentiment_analyzed"
SENTIMENT_OUTPUT_PATH = DATA_DIR / "sentiment_analyzed" / "sentiment_analyzed_posts.csv"
METRICS_OUTPUT_PATH = DATA_DIR / "metrics" / "Post_performance_metrics.csv"


def _safe_read_csv(path: Path) -> pd.DataFrame:
	if not path.exists():
		raise FileNotFoundError(f"Missing required file: {path}")
	return pd.read_csv(path)


def _analyze_sentiment(text: str) -> tuple[float, str]:
	blob = TextBlob(str(text))
	polarity = float(blob.sentiment.polarity)
	if polarity > 0.2:
		label = "Positive"
	elif polarity < -0.2:
		label = "Negative"
	else:
		label = "Neutral"
	return polarity, label


def _dominant_emotion(score: float) -> str:
	if score > 0.5:
		return "Joy"
	if 0.1 < score <= 0.5:
		return "Optimism"
	if -0.1 <= score <= 0.1:
		return "Calm"
	if -0.5 <= score < -0.1:
		return "Frustration"
	return "Anger"


def _run_sentiment_pipeline(df):
	if df.empty:
		return df
	result = df.copy()
	result["sentiment_score"], result["sentiment_label"] = zip(
		*result["generated_text"].apply(_analyze_sentiment)
	)
	result["dominant_emotion"] = result["sentiment_score"].apply(_dominant_emotion)
	return result


def _run_performance_metrics():
	optimized_file = PROCESSED_DIR / "optimized_posts.csv"
	engagement_file = PROCESSED_DIR / "combined_engagement_data.csv"
	sentiment_file = SENTIMENT_DIR / "sentiment_analyzed_posts.csv"

	df_opt = _safe_read_csv(optimized_file)
	df_sent = _safe_read_csv(sentiment_file)
	df_opt = df_opt.merge(
		df_sent[["topic", "sentiment_score", "sentiment_label"]],
		on="topic",
		how="left",
	)

	df_eng = _safe_read_csv(engagement_file)
	df_eng["topic"] = df_eng["platform"]

	metrics = ["like_count", "comment_count", "share_count", "view_count"]
	for col in metrics:
		if col not in df_eng.columns:
			df_eng[col] = 0

	df = df_opt.merge(df_eng, on="topic", how="left")
	df["platform"] = df["platform"].fillna(df["topic"])
	df["total_engagement"] = (
		df["like_count"].fillna(0)
		+ df["comment_count"].fillna(0)
		+ df["share_count"].fillna(0)
	)
	df["engagement_rate"] = df["total_engagement"] / (df["view_count"].fillna(0) + 1)
	df["engagement_rate"] = df["engagement_rate"].round(4)

	avg_eng = (
		df.groupby("platform")["engagement_rate"].mean().sort_values(ascending=False)
	)
	sent = df["sentiment_score"].dropna()
	eng = df["engagement_rate"].dropna()
	if len(sent) < 2 or sent.std() == 0 or eng.std() == 0:
		corr_value = float("nan")
	else:
		corr_value = sent.corr(eng)

	METRICS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
	df_out = df[[
		"topic",
		"tone",
		"variation_no",
		"generated_text",
		"score",
		"sentiment_score",
		"engagement_rate",
		"total_engagement",
	]].copy()
	# Keep just the best-performing variation per topic (highest score).
	df_out = (
		df_out.sort_values(["topic", "score"], ascending=[True, False])
		.drop_duplicates(subset=["topic"], keep="first")
	)
	df_out.to_csv(METRICS_OUTPUT_PATH, index=False, encoding="utf-8")

	return {
		"df": df,
		"df_out": df_out,
		"avg_eng": avg_eng,
		"corr": corr_value,
	}


def _load_posting_recommendations():
	data_file = PROCESSED_DIR / "combined_engagement_data.csv"
	try:
		df = _safe_read_csv(data_file)
	except FileNotFoundError:
		return None
	if "posted_at" not in df.columns or "engagement_rate" not in df.columns:
		return None
	working = df[["posted_at", "engagement_rate"]].dropna()
	if working.empty:
		return None
	working["posted_at"] = pd.to_datetime(working["posted_at"], errors="coerce")
	working = working.dropna(subset=["posted_at", "engagement_rate"])
	if working.empty:
		return None
	working["day_of_week"] = working["posted_at"].dt.day_name()
	working["hour_of_day"] = working["posted_at"].dt.hour
	valid = working.dropna(subset=["day_of_week", "hour_of_day"])
	if valid.empty:
		return None
	week_order = [
		"Monday",
		"Tuesday",
		"Wednesday",
		"Thursday",
		"Friday",
		"Saturday",
		"Sunday",
	]
	day_stats = (
		valid.groupby("day_of_week")["engagement_rate"]
		.mean()
		.reindex(week_order)
		.dropna()
	)
	hour_stats = valid.groupby("hour_of_day")["engagement_rate"].mean().sort_values(ascending=False)
	best_day = day_stats.idxmax() if not day_stats.empty else None
	best_day_rate = float(day_stats.max()) if best_day else None
	best_hour = int(hour_stats.idxmax()) if not hour_stats.empty else None
	best_hour_rate = float(hour_stats.max()) if best_hour is not None else None
	return {
		"day": best_day,
		"day_rate": best_day_rate,
		"hour": best_hour,
		"hour_rate": best_hour_rate,
	}


def get_generation_context() -> GenerationContext:
	return load_generation_context(top_n_keywords=None, max_prompt_hashtags=None)


def main() -> None:
	st.set_page_config(page_title="Content Generation", layout="wide")
	st.title("AI-Powered Content Generator")
	st.caption("Generate, score, and optimize social posts using your latest trend signals.")
	st.info("This is a simulated campaign test based on historical trends.")

	context = get_generation_context()
	posting_recommendations = _load_posting_recommendations()

	with st.sidebar:
		st.subheader("Context snapshot")
		st.markdown(
			f"**Detected tone:** `{context.best_tone}`\n\n"
			f"**Keyword count:** {len(context.top_keywords)}\n"
			f"**Hashtags in pool:** {len(context.prompt_hashtags)}"
		)
		if context.prompt_hashtags:
			st.markdown(
				"<small>Sample hashtags:</small><br>" +
				", ".join(context.prompt_hashtags[:10]),
				unsafe_allow_html=True,
			)

		if "show_keywords" not in st.session_state:
			st.session_state["show_keywords"] = False
		label = "Hide top keywords" if st.session_state["show_keywords"] else "Show top keywords"
		if st.button(label, key="toggle_keywords"):
			st.session_state["show_keywords"] = not st.session_state["show_keywords"]
		if st.session_state["show_keywords"]:
			st.write(", ".join(context.top_keywords) or "No keywords available")

		if "show_hashtags" not in st.session_state:
			st.session_state["show_hashtags"] = False
		hashtag_label = "Hide hashtag pool" if st.session_state["show_hashtags"] else "Show hashtag pool"
		if st.button(hashtag_label, key="toggle_hashtags"):
			st.session_state["show_hashtags"] = not st.session_state["show_hashtags"]
		if st.session_state["show_hashtags"]:
			st.write(", ".join(context.prompt_hashtags) or "No hashtags available")

	with st.form("generation_form"):
		topic = st.text_input(
			"Topic / campaign focus",
			value="AI in Content Marketing",
			help="Describe the main idea the post should cover.",
		)
		tone = st.text_input(
			"Desired tone",
			value=context.best_tone,
			help="Fallbacks to the best-performing tone from sentiment analysis.",
		)
		available_keywords = max(1, len(context.top_keywords))
		max_keywords = st.slider(
			"Keywords to include",
			min_value=1,
			max_value=available_keywords,
			value=available_keywords,
			step=1,
		)
		max_words = st.slider("Max words", min_value=80, max_value=250, value=150)
		custom_keywords = st.text_input(
			"Must-use keywords (comma separated)",
			help="List any additional keywords that should appear in generated posts.",
		)
		available_hashtags = max(1, 10)
		max_hashtags = st.slider(
			"Hashtags to include",
			min_value=1,
			max_value=10,
			value=available_hashtags,
			step=1,
		)
		custom_hashtags = st.text_input(
			"Must-use hashtags (comma separated)",
			help="Add specific hashtags you want inserted into every variation.",
		)

		submitted = st.form_submit_button("Generate posts", width="stretch")

	if submitted:
		if not topic.strip():
			st.error("Please provide a topic before generating posts.")
			return

		selected_keywords = context.top_keywords[:max_keywords]
		user_keywords = [kw.strip() for kw in custom_keywords.split(",") if kw.strip()] if custom_keywords else []
		if user_keywords:
			selected_keywords = user_keywords + [kw for kw in selected_keywords if kw not in user_keywords]
		selected_hashtags = context.prompt_hashtags[:max_hashtags]
		user_hashtags = [ht.strip() for ht in custom_hashtags.split(",") if ht.strip()] if custom_hashtags else []
		if user_hashtags:
			selected_hashtags = user_hashtags + [ht for ht in selected_hashtags if ht not in user_hashtags]
		with st.spinner("Generating posts..."):
			scored_df, _ = run_generation(
				topic=topic.strip(),
				tone=tone.strip() or context.best_tone,
				context=context,
				keywords=selected_keywords,
				hashtags=selected_hashtags,
				num_variations=3,
				max_words=max_words,
			)

		st.success(f"Generated {len(scored_df)} variations.")
		if posting_recommendations:
			day_col, hour_col = st.columns(2)
			day_value = posting_recommendations.get("day") or "—"
			day_delta = None
			if posting_recommendations.get("day_rate") is not None:
				day_delta = f"Avg engagement {posting_recommendations['day_rate'] * 100:.1f}%"
			hour_value = "—"
			if posting_recommendations.get("hour") is not None:
				hour_value = f"{posting_recommendations['hour']:02d}:00 UTC"
			hour_delta = None
			if posting_recommendations.get("hour_rate") is not None:
				hour_delta = f"Avg engagement {posting_recommendations['hour_rate'] * 100:.1f}%"
			day_col.metric("Best day to publish", day_value, delta=day_delta)
			hour_col.metric("Best hour to publish", hour_value, delta=hour_delta)
		st.session_state["latest_posts"] = scored_df
		st.session_state["latest_keywords"] = selected_keywords
		if not scored_df.empty:
			winner = scored_df.iloc[0]
			st.success(f"Recommended Winner: Variation A (Control) (Score {winner['score']:.2f})")
			summary_table = build_scoring_summary(scored_df, selected_keywords)
			winner_features = summary_table.iloc[0]
			reasons = []
			kw_hits = int(winner_features.get("keyword_hits", 0))
			if kw_hits:
				reasons.append(f"Better keyword coverage: {kw_hits} target terms present")
			sent_val = float(winner_features.get("sentiment", 0.0))
			if sent_val > 0.15:
				reasons.append(f"Positive sentiment (+{sent_val:.2f})")
			elif sent_val < -0.15:
				reasons.append(f"Strong stance (sentiment {sent_val:.2f})")
			wc = int(winner_features.get("word_count", 0))
			if 20 <= wc <= 80:
				reasons.append(f"Optimal length for feed (≈{wc} words)")
			htg = int(winner_features.get("hashtags", 0))
			if 1 <= htg <= 3:
				reasons.append(f"Clean hashtag use ({htg} tags)")
			if reasons:
				st.markdown("**Why this wins:**\n- " + "\n- ".join(reasons))
		st.markdown("### A/B Test Results")
		for idx, row in scored_df.iterrows():
			letter = chr(65 + idx) if idx < 26 else str(idx + 1)
			label = f"{letter} (Control)" if idx == 0 else letter
			badge = " 🏅 Best Choice" if idx == 0 else ""
			score_val = float(row.get("score", 0.0))
			if score_val >= 5:
				prediction = "🔥 High Potential"
			elif score_val >= 3:
				prediction = "⚠️ Medium Potential"
			else:
				prediction = "❌ Low Potential"
			text_body = str(row.get("generated_text", ""))
			word_guess = len(text_body.split())
			hashtag_guess = text_body.count("#")
			if word_guess <= 80 and hashtag_guess <= 3:
				platform_hint = "Suited for Twitter"
			elif hashtag_guess >= 5:
				platform_hint = "Suited for Instagram"
			else:
				platform_hint = "Suited for LinkedIn"
			st.markdown(f"**Variation {label} – Score {row['score']:.2f}{badge}**  · {prediction}")
			st.write(row["generated_text"])
			st.caption(platform_hint)
			st.caption(row.get("generated_at", ""))
			st.divider()

	sentiment_disabled = "latest_posts" not in st.session_state or st.session_state["latest_posts"].empty
	if st.button("Analyze sentiment for latest posts", disabled=sentiment_disabled, type="primary"):
		latest_df = st.session_state.get("latest_posts")
		if latest_df is None or latest_df.empty:
			st.warning("Generate posts first to analyze sentiment.")
		else:
			with st.spinner("Running sentiment analysis..."):
				analysis_df = _run_sentiment_pipeline(latest_df)
				SENTIMENT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
				analysis_df.to_csv(SENTIMENT_OUTPUT_PATH, index=False, encoding="utf-8")
			st.dataframe(
				analysis_df[[
					"variation_no",
					"sentiment_label",
					"dominant_emotion",
					"sentiment_score",
					"generated_text",
				]],
				width="stretch",
			)
			st.bar_chart(analysis_df["sentiment_label"].value_counts())

	st.caption("Scoring weighs keyword hits, sentiment, readability, length, and hashtag usage to rank each variation.")
	if st.button("Show scoring details", disabled=sentiment_disabled):
		latest_df = st.session_state.get("latest_posts")
		if latest_df is None or latest_df.empty:
			st.warning("Generate posts first to see scoring details.")
		else:
			keywords = st.session_state.get("latest_keywords") or context.top_keywords
			with st.spinner("Scoring latest posts..."):
				scoring_table = build_scoring_summary(latest_df, keywords)
			st.dataframe(
				scoring_table[[
					"variation_no",
					"score",
					"final_score",
					"keyword_hits",
					"sentiment",
					"word_count",
					"hashtags",
					"generated_text",
				]],
				width="stretch",
			)

	st.markdown("---")
	st.subheader("Performance Metrics")
	if st.button("Compute performance metrics", type="secondary"):
		try:
			with st.spinner("Crunching metrics..."):
				metrics_result = _run_performance_metrics()
		except FileNotFoundError as exc:
			st.error(str(exc))
		else:
			corr_value = metrics_result["corr"]
			if pd.notna(corr_value):
				st.metric("Sentiment↔Engagement correlation", f"{corr_value:.3f}")
			latest_df = st.session_state.get("latest_posts")
			filtered = metrics_result["df_out"]
			if latest_df is not None and not latest_df.empty:
				latest_texts = set(latest_df["generated_text"].astype(str).str.strip())
				filtered = filtered[filtered["generated_text"].astype(str).str.strip().isin(latest_texts)]
				if filtered.empty:
					filtered = metrics_result["df_out"]
			else:
				st.info("No recent generation found; showing complete metrics table.")

			st.dataframe(filtered, width="stretch")


if __name__ == "__main__":
	main()
