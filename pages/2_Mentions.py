import streamlit as st
import pandas as pd
import numpy as np
import re
from io import BytesIO
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords as nltk_stopwords
from datetime import datetime
import pytz
import html

# Download NLTK stopwords only once
nltk.download('stopwords', quiet=True)

# Define stopwords properly
NLTK_STOPWORDS = nltk_stopwords.words("english")
STOPWORDS_SET = list(set(NLTK_STOPWORDS) | set(STOPWORDS))

st.set_page_config(page_title="Mentions Monitor", layout="wide")

st.title("📢 Mentions Monitor")

# ------------------------
# Load Data
# ------------------------
@st.cache_data(ttl=3600)
def load_data(url):
    df = pd.read_csv(url)
    return df

sheet_url = st.secrets["public_gsheets_url"]
df = load_data(sheet_url)

if df.empty:
    st.warning("No data available.")
    st.stop()

# ------------------------
# Data Cleaning
# ------------------------
def clean_html(text):
    return html.escape(str(text))

df["TITLE"] = df["TITLE"].fillna("").apply(clean_html)
df["DESCRIPTION"] = df["DESCRIPTION"].fillna("").apply(clean_html)
df["SOURCE"] = df["SOURCE"].fillna("Unknown")

# Parse dates
def parse_date(d):
    try:
        return pd.to_datetime(d)
    except:
        return pd.NaT

df["DATE"] = df["DATE"].apply(parse_date)
df["DATE"] = df["DATE"].dt.tz_localize("UTC").dt.tz_convert("Africa/Nairobi")

# ------------------------
# Sidebar Filters
# ------------------------
st.sidebar.header("🔎 Filters")

sources = st.sidebar.multiselect("Filter by Source", options=df["SOURCE"].unique(), default=df["SOURCE"].unique())
tones = st.sidebar.multiselect("Filter by Tonality", options=df["TONALITY"].unique(), default=df["TONALITY"].unique())

min_date, max_date = df["DATE"].min(), df["DATE"].max()
date_range = st.sidebar.date_input("Date range", [min_date, max_date])

keyword_filter = st.sidebar.text_input("Search keyword")

filtered = df[
    df["SOURCE"].isin(sources) &
    df["TONALITY"].isin(tones) &
    (df["DATE"].dt.date.between(date_range[0], date_range[1]))
]

if keyword_filter:
    filtered = filtered[
        filtered["TITLE"].str.contains(keyword_filter, case=False, na=False) |
        filtered["DESCRIPTION"].str.contains(keyword_filter, case=False, na=False)
    ]

st.subheader("Filtered Mentions")
st.write(f"Total results: {len(filtered)}")

# ------------------------
# Display Table
# ------------------------
def make_clickable(link):
    return f'<a href="{link}" target="_blank">🔗 Link</a>'

def color_row(tonality):
    if tonality == "Positive":
        return "background-color: #d4edda;"
    elif tonality == "Negative":
        return "background-color: #f8d7da;"
    else:
        return "background-color: #fff3cd;"

styled_df = filtered.copy()
styled_df["URL"] = styled_df["URL"].apply(make_clickable)

st.write(
    styled_df[["DATE", "SOURCE", "TONALITY", "TITLE", "DESCRIPTION", "URL"]]
    .style.applymap(color_row, subset=["TONALITY"])
    .to_html(escape=False),
    unsafe_allow_html=True,
)

# ------------------------
# Download CSV
# ------------------------
csv = filtered.to_csv(index=False).encode("utf-8")
st.download_button("📥 Download CSV", data=csv, file_name="mentions.csv", mime="text/csv")

# ------------------------
# Helper functions for n-grams
# ------------------------
def top_unigrams(texts, top_n=20, extra_stopwords=None):
    stop_words = STOPWORDS_SET.copy()
    if extra_stopwords:
        stop_words = list(set(stop_words) | set(extra_stopwords))
    vectorizer = CountVectorizer(ngram_range=(1, 1), stop_words=stop_words)
    X = vectorizer.fit_transform(texts)
    freqs = zip(vectorizer.get_feature_names_out(), X.sum(axis=0).tolist()[0])
    return sorted(freqs, key=lambda x: -x[1])[:top_n]

def top_ngrams(texts, n=2, top_n=20, extra_stopwords=None):
    stop_words = STOPWORDS_SET.copy()
    if extra_stopwords:
        stop_words = list(set(stop_words) | set(extra_stopwords))
    vectorizer = CountVectorizer(ngram_range=(n, n), stop_words=stop_words)
    X = vectorizer.fit_transform(texts)
    freqs = zip(vectorizer.get_feature_names_out(), X.sum(axis=0).tolist()[0])
    return sorted(freqs, key=lambda x: -x[1])[:top_n]

# ------------------------
# Keyword Analysis
# ------------------------
st.subheader("🔠 Keyword Analysis")

all_text = " ".join(filtered["TITLE"].fillna("") + " " + filtered["DESCRIPTION"].fillna(""))

if all_text.strip():
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Top Unigrams**")
        for word, freq in top_unigrams([all_text]):
            st.write(f"{word}: {freq}")

        st.markdown("**Top Bigrams**")
        for word, freq in top_ngrams([all_text], n=2):
            st.write(f"{word}: {freq}")

    with col2:
        st.markdown("**Top Trigrams**")
        for word, freq in top_ngrams([all_text], n=3):
            st.write(f"{word}: {freq}")

        # Wordcloud
        st.markdown("**Word Cloud**")
        wordcloud = WordCloud(width=800, height=400, background_color="white", stopwords=STOPWORDS_SET).generate(all_text)
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)
else:
    st.info("No text available for keyword analysis.")

# ------------------------
# Charts
# ------------------------
st.subheader("📊 Insights")

# Tonality distribution
tonality_counts = filtered["TONALITY"].value_counts()
st.bar_chart(tonality_counts)

# Mentions over time
trend = filtered.groupby(filtered["DATE"].dt.date).size()
st.line_chart(trend)

# Source analysis
source_counts = filtered["SOURCE"].value_counts().head(10)
st.bar_chart(source_counts)

