# pages/2_Mentions.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import html
from datetime import datetime, date

# ---------------- CONFIG ----------------
CSV_URL = "https://docs.google.com/spreadsheets/d/10LcDId4y2vz5mk7BReXL303-OBa2QxsN3drUcefpdSQ/export?format=csv"

# ---------------- HELPERS & CACHES ----------------
@st.cache_data(ttl=3600)
def ensure_nltk_stopwords():
    """Download stopwords if missing (cached)."""
    try:
        nltk.data.find("corpora/stopwords")
    except Exception:
        nltk.download("stopwords", quiet=True)
    return set(nltk.corpus.stopwords.words("english"))

# Load stopwords once
NLTK_STOPWORDS = ensure_nltk_stopwords()

@st.cache_data(ttl=3600)
def load_data(csv_url: str) -> pd.DataFrame:
    """Load CSV from URL and normalize columns + parse dates."""
    try:
        df = pd.read_csv(csv_url)
    except Exception as e:
        st.error(f"❌ Failed to load CSV: {e}")
        return pd.DataFrame()

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Parse and localize dates into published_parsed (UTC -> Africa/Nairobi)
    if "published" in df.columns:
        df["published_parsed"] = pd.to_datetime(df["published"], errors="coerce", utc=True)
        # Convert to Africa/Nairobi if tz-aware
        try:
            df["published_parsed"] = df["published_parsed"].dt.tz_convert("Africa/Nairobi")
        except Exception:
            # If conversion fails (e.g., all NaT), leave as-is
            pass
        # Create DATE (YYYY-MM-DD for grouping) and display date
        df["DATE"] = df["published_parsed"].dt.date.astype("object").fillna("")  # date objects or empty string
        df["DATE_STR"] = df["published_parsed"].dt.strftime("%d-%b-%Y").fillna("")
        df["TIME"] = df["published_parsed"].dt.strftime("%H:%M").fillna("")
    else:
        df["published_parsed"] = pd.NaT
        df["DATE"] = ""
        df["DATE_STR"] = ""
        df["TIME"] = ""

    # Ensure text columns exist and fillna
    for col in ["title", "summary", "source", "tonality", "link"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)

    # Normalize tonality values to Title case and map common variants
    def normalize_tonality(x):
        s = str(x).strip().lower()
        if s in ("pos", "positive", "p"):
            return "Positive"
        if s in ("neg", "negative", "n"):
            return "Negative"
        if s in ("neu", "neutral", "neutrality"):
            return "Neutral"
        return x.title() if x else ""

    df["tonality"] = df["tonality"].apply(normalize_tonality)

    # Rename for display consistency (uppercase column names)
    rename_map = {
        "title": "TITLE",
        "summary": "SUMMARY",
        "source": "SOURCE",
        "tonality": "TONALITY",
        "link": "LINK",
    }
    df = df.rename(columns=rename_map)

    return df

# Utility: build HTML table with conditional row colors (safe escaping)
def make_html_table(df: pd.DataFrame, cols, max_rows=200):
    COLORS = {
        "Positive": "#dff7df",
        "Neutral": "#f3f3f3",
        "Negative": "#ffd6d6"
    }

    html_table = """
    <div style="overflow:auto; max-height:650px;">
    <table style="border-collapse:collapse; width:100%; font-family:Arial, sans-serif;">
      <thead>
        <tr>
    """
    for c in cols:
        html_table += f'<th style="text-align:left; padding:8px; border-bottom:1px solid #ddd; background:#f8f8f8;">{html.escape(c)}</th>'
    html_table += "</tr></thead><tbody>"

    n = min(len(df), max_rows)
    for i in range(n):
        row = df.iloc[i]
        ton = str(row.get("TONALITY", "")).strip()
        bg = COLORS.get(ton, "#ffffff")
        html_table += f'<tr style="background:{bg};">'
        for c in cols:
            v = row.get(c, "")
            if c == "LINK" and isinstance(v, str) and v.startswith("http"):
                cell = f'<a href="{html.escape(v)}" target="_blank" rel="noopener noreferrer">Open Article</a>'
            else:
                cell = html.escape(str(v))
                # Keep line breaks readable
                cell = cell.replace("\n", "<br/>")
            html_table += f'<td style="padding:8px; vertical-align:top; border-bottom:1px solid #eee;">{cell}</td>'
        html_table += "</tr>"

    html_table += "</tbody></table></div>"
    return html_table

# N-gram utilities
def top_unigrams(texts, top_n=20, extra_stopwords=None):
    stop_words = set(NLTK_STOPWORDS) | set(STOPWORDS)
    if extra_stopwords:
        stop_words |= set(extra_stopwords)
    vectorizer = CountVectorizer(ngram_range=(1, 1), stop_words=stop_words)
    X = vectorizer.fit_transform(texts)
    counts = np.asarray(X.sum(axis=0)).ravel()
    vocab = np.array(vectorizer.get_feature_names_out())
    top_idx = counts.argsort()[::-1][:top_n]
    return list(zip(vocab[top_idx], counts[top_idx]))

def top_ngrams(texts, ngram=(2,2), top_n=20, extra_stopwords=None):
    stop_words = set(NLTK_STOPWORDS) | set(STOPWORDS)
    if extra_stopwords:
        stop_words |= set(extra_stopwords)
    vectorizer = CountVectorizer(ngram_range=ngram, stop_words=stop_words, min_df=1)
    X = vectorizer.fit_transform(texts)
    counts = np.asarray(X.sum(axis=0)).ravel()
    vocab = np.array(vectorizer.get_feature_names_out())
    top_idx = counts.argsort()[::-1][:top_n]
    return list(zip(vocab[top_idx], counts[top_idx]))

def make_wordcloud_figure(text_blob, width=10, height=5):
    fig = plt.figure(figsize=(width, height))
    if not text_blob.strip():
        plt.text(0.5, 0.5, "No text available", horizontalalignment='center', verticalalignment='center')
        plt.axis("off")
        return fig
    wc = WordCloud(width=int(width*100), height=int(height*100), background_color="white",
                   stopwords=set(NLTK_STOPWORDS) | set(STOPWORDS))
    wc.generate(text_blob)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.tight_layout(pad=0)
    return fig

# ---------------- MAIN APP ----------------
st.set_page_config(page_title="Mentions — Enhanced", layout="wide")
st.title("📋 Mentions — Enhanced View")

df = load_data(CSV_URL)
if df.empty:
    st.info("No data available. Check the CSV URL and sharing settings.")
    st.stop()

# Sidebar filters
st.sidebar.header("Filters / Keyword Trends")

# Date range filter using published_parsed dates if available
has_dates = df["DATE"].dtype == object and df["DATE"].nunique() > 0 and df["published_parsed"].notna().any()
if has_dates:
    # derive min/max as date objects
    valid_dates = df.loc[df["published_parsed"].notna(), "published_parsed"].dt.tz_convert("Africa/Nairobi").dt.date
    min_dt = valid_dates.min()
    max_dt = valid_dates.max()
    date_range = st.sidebar.date_input("Date range (Nairobi)", value=(min_dt, max_dt), min_value=min_dt, max_value=max_dt)
    # Ensure tuple
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = date_range
        end_date = date_range
else:
    start_date = None
    end_date = None

# Keyword search input + dynamic multiselect of top terms
search = st.sidebar.text_input("Search (TITLE / SUMMARY)")
# Build texts for overall top terms (use full df to show choices)
all_texts = (df.get("TITLE", "").astype(str) + " " + df.get("SUMMARY", "").astype(str)).tolist()
# Compute top unigrams for selection (catch if not enough text)
try:
    top_terms = [t for t, _ in top_unigrams(all_texts, top_n=40)]
except Exception:
    top_terms = []
keyword_select = st.sidebar.multiselect("Filter by keyword (top terms)", options=top_terms, default=[])

# Tonality filter
tonality_options = ["Positive", "Neutral", "Negative"]
selected_tonalities = st.sidebar.multiselect("Tonality", options=tonality_options, default=tonality_options)

# Apply filters
filtered = df.copy()

# Date filtering
if start_date and end_date:
    mask_date = filtered["published_parsed"].notna() & \
                (filtered["published_parsed"].dt.tz_convert("Africa/Nairobi").dt.date >= start_date) & \
                (filtered["published_parsed"].dt.tz_convert("Africa/Nairobi").dt.date <= end_date)
    filtered = filtered.loc[mask_date].reset_index(drop=True)

# Text search filter
if search:
    mask_search = filtered["TITLE"].str.contains(search, case=False, na=False) | filtered["SUMMARY"].str.contains(search, case=False, na=False)
    filtered = filtered.loc[mask_search].reset_index(drop=True)

# Keyword multiselect filter (require ALL selected keywords to appear - user-friendly)
for kw in keyword_select:
    if kw:
        mask_kw = filtered["TITLE"].str.contains(rf"\b{kw}\b", case=False, na=False) | filtered["SUMMARY"].str.contains(rf"\b{kw}\b", case=False, na=False)
        filtered = filtered.loc[mask_kw].reset_index(drop=True)

# Tonality filter
if selected_tonalities:
    filtered = filtered[filtered["TONALITY"].isin(selected_tonalities)].reset_index(drop=True)

# Results info
st.markdown(f"**Results:** {len(filtered):,} articles (showing latest first)")

# Sort by date if available
if "published_parsed" in filtered.columns:
    filtered = filtered.sort_values(by="published_parsed", ascending=False).reset_index(drop=True)

# Columns to show in table
cols_to_show = ["DATE_STR", "TIME", "SOURCE", "TONALITY", "TITLE", "SUMMARY", "LINK"]
cols_to_show = [c for c in cols_to_show if c in filtered.columns]

# Render HTML table
html_table = make_html_table(filtered, cols_to_show, max_rows=500)
st.markdown(html_table, unsafe_allow_html=True)

# CSV download (all filtered rows)
csv_bytes = filtered[cols_to_show].to_csv(index=False).encode("utf-8")
st.download_button("📥 Download Filtered Mentions (CSV)", data=csv_bytes, file_name="helb_mentions_filtered.csv", mime="text/csv")

# ---------------- Analytics / Visuals ----------------
st.markdown("---")
st.header("🔎 Keyword & Sentiment Trends")

# Combine text for analysis (from filtered set)
texts = (filtered.get("TITLE", "").astype(str) + " " + filtered.get("SUMMARY", "").astype(str)).tolist()
big_text = " ".join(texts)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Top Unigrams")
    top_uni = top_unigrams(texts, top_n=20)
    if top_uni:
        uni_df = pd.DataFrame(top_uni, columns=["term", "count"])
        st.table(uni_df)
    else:
        st.info("Not enough text to extract unigrams.")

    st.subheader("Top Bigrams")
    top_bi = top_ngrams(texts, ngram=(2,2), top_n=15)
    if top_bi:
        bi_df = pd.DataFrame(top_bi, columns=["bigram", "count"])
        st.table(bi_df)
    else:
        st.info("Not enough text for bigrams.")

with col2:
    st.subheader("Word Cloud")
    fig_wc = make_wordcloud_figure(big_text, width=10, height=5)
    st.pyplot(fig_wc)

# Top Trigrams
st.subheader("Top Trigrams")
top_tri = top_ngrams(texts, ngram=(3,3), top_n=12)
if top_tri:
    tri_df = pd.DataFrame(top_tri, columns=["trigram", "count"])
    st.table(tri_df)
else:
    st.info("No trigrams available.")

# ---------------- Tonality Distribution & Trends ----------------
st.markdown("---")
st.header("📈 Tonality & Activity")

col_a, col_b = st.columns([1, 1])

with col_a:
    st.subheader("Tonality Distribution")
    ton_counts = filtered["TONALITY"].value_counts().reindex(tonality_options).fillna(0)
    fig1, ax1 = plt.subplots()
    ax1.bar(ton_counts.index, ton_counts.values)
    ax1.set_xlabel("Tonality")
    ax1.set_ylabel("Count")
    ax1.set_title("Mentions by Tonality")
    for i, v in enumerate(ton_counts.values):
        ax1.text(i, v + max(1, ton_counts.values.max()*0.01), str(int(v)), ha='center')
    st.pyplot(fig1)

with col_b:
    st.subheader("Top Sources")
    top_sources = filtered["SOURCE"].value_counts().head(10)
    if not top_sources.empty:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.barh(top_sources.index[::-1], top_sources.values[::-1])
        ax2.set_xlabel("Count")
        ax2.set_title("Top 10 Sources")
        st.pyplot(fig2)
    else:
        st.info("No source data available.")

# Mentions over time (daily)
st.subheader("Mentions Over Time (daily)")
if filtered["published_parsed"].notna().any():
    time_series = filtered.groupby(filtered["published_parsed"].dt.tz_convert("Africa/Nairobi").dt.date).size()
    time_series = time_series.sort_index()
    if not time_series.empty:
        fig3, ax3 = plt.subplots(figsize=(10, 3))
        ax3.plot(time_series.index, time_series.values, marker='o', linestyle='-')
        ax3.set_xlabel("Date")
        ax3.set_ylabel("Mentions")
        ax3.set_title("Mentions per Day")
        plt.xticks(rotation=45)
        st.pyplot(fig3)
    else:
        st.info("Not enough date data to plot mentions over time.")
else:
    st.info("No publish dates available to plot mentions over time.")

# Sentiment trend (stacked)
st.subheader("Sentiment Trend Over Time (stacked)")
if filtered["published_parsed"].notna().any():
    df_trend = filtered.copy()
    df_trend["date_day"] = df_trend["published_parsed"].dt.tz_convert("Africa/Nairobi").dt.date
    pivot = df_trend.pivot_table(index="date_day", columns="TONALITY", values="TITLE", aggfunc="count").fillna(0)
    # Ensure columns order
    for col in tonality_options:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot[tonality_options].sort_index()
    if not pivot.empty:
        fig4, ax4 = plt.subplots(figsize=(10, 4))
        pivot.plot(kind="bar", stacked=True, ax=ax4)
        ax4.set_xlabel("Date")
        ax4.set_ylabel("Count")
        ax4.set_title("Tonality counts per day (stacked)")
        plt.xticks(rotation=45)
        st.pyplot(fig4)
    else:
        st.info("Not enough data to build sentiment trend.")
else:
    st.info("No publish dates available for sentiment trend.")

st.markdown("---")
st.caption("Tip: Adjust filters in the sidebar (date range, keywords, and tonality) to update all tables and charts.")
