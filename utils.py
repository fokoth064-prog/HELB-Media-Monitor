# utils.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer
import nltk

# Ensure NLTK stopwords are available
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords as nltk_stopwords


# ---------------- CONFIG ----------------
CSV_URL = "https://docs.google.com/spreadsheets/d/10LcDId4y2vz5mk7BReXL303-OBa2QxsN3drUcefpdSQ/export?format=csv"


# ---------------- LOAD DATA ----------------
def load_data(csv_url: str = CSV_URL) -> pd.DataFrame:
    """Load and preprocess mentions dataset from Google Sheets CSV."""
    try:
        df = pd.read_csv(csv_url)
    except Exception as e:
        print(f"❌ Failed to load CSV: {e}")
        return pd.DataFrame()

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Parse and localize dates
    if "published" in df.columns:
        df["published_parsed"] = pd.to_datetime(df["published"], errors="coerce", utc=True)
        try:
            df["published_parsed"] = df["published_parsed"].dt.tz_convert("Africa/Nairobi")
        except Exception:
            pass
        df["DATE"] = df["published_parsed"].dt.strftime("%d-%b-%Y")
        df["TIME"] = df["published_parsed"].dt.strftime("%H:%M")
    else:
        df["published_parsed"] = pd.NaT
        df["DATE"] = ""
        df["TIME"] = ""

    # Ensure text columns exist
    for col in ["title", "summary", "source", "tonality", "link"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("")

    # Rename to uppercase for display consistency
    rename_map = {
        "title": "TITLE",
        "summary": "SUMMARY",
        "source": "SOURCE",
        "tonality": "TONALITY",
        "link": "LINK",
    }
    df = df.rename(columns=rename_map)

    return df


# ---------------- HTML TABLE BUILDER ----------------
def make_html_table(df: pd.DataFrame, cols: list, max_rows: int = 200) -> str:
    """
    Create an HTML table with inline styles:
    - Positive -> darker green background
    - Neutral  -> light grey
    - Negative -> darker red background
    """
    COLORS = {
        "Positive": "#a8e6a3",  # darker green
        "Neutral": "#f3f3f3",   # light grey
        "Negative": "#f5a8a8"   # darker red
    }

    html = """
    <div style="overflow:auto; max-height:650px;">
    <table style="border-collapse:collapse; width:100%; font-family:Arial, sans-serif; font-size:14px; color:#000;">
      <thead>
        <tr>
    """
    # Header row
    for c in cols:
        html += f'<th style="text-align:left; padding:8px; border-bottom:1px solid #ddd; background:#f8f8f8; color:#000;">{c}</th>'
    html += "</tr></thead><tbody>"

    # Rows
    n = min(len(df), max_rows)
    for i in range(n):
        row = df.iloc[i]
        ton = str(row.get("TONALITY", "")).strip()
        bg = COLORS.get(ton, "#ffffff")
        html += f'<tr style="background:{bg};">'
        for c in cols:
            v = row.get(c, "")
            if c == "LINK" and isinstance(v, str) and v.startswith("http"):
                cell = f'<a href="{v}" target="_blank" rel="noopener noreferrer">Open Article</a>'
            else:
                cell = str(v).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            html += f'<td style="padding:8px; vertical-align:top; border-bottom:1px solid #eee; color:#000;">{cell}</td>'
        html += "</tr>"
    html += "</tbody></table></div>"
    return html


# ---------------- KEYWORD UTILITIES ----------------
def clean_stopwords(extra_stopwords=None):
    """Combine NLTK + wordcloud stopwords and clean them for sklearn."""
    stop_words = set(nltk_stopwords.words("english")) | set(STOPWORDS)
    if extra_stopwords:
        stop_words |= set(extra_stopwords)
    stop_words = {w for w in stop_words if isinstance(w, str) and w.strip()}
    return stop_words


def top_unigrams(texts, top_n=20, extra_stopwords=None):
    stop_words = clean_stopwords(extra_stopwords)
    vectorizer = CountVectorizer(ngram_range=(1, 1), stop_words=list(stop_words))
    X = vectorizer.fit_transform(texts)
    counts = np.asarray(X.sum(axis=0)).ravel()
    vocab = np.array(vectorizer.get_feature_names_out())
    top_idx = counts.argsort()[::-1][:top_n]
    return list(zip(vocab[top_idx], counts[top_idx]))


def top_ngrams(texts, ngram=(2, 2), top_n=20, extra_stopwords=None):
    stop_words = clean_stopwords(extra_stopwords)
    vectorizer = CountVectorizer(ngram_range=ngram, stop_words=list(stop_words), min_df=1)
    X = vectorizer.fit_transform(texts)
    counts = np.asarray(X.sum(axis=0)).ravel()
    vocab = np.array(vectorizer.get_feature_names_out())
    top_idx = counts.argsort()[::-1][:top_n]
    return list(zip(vocab[top_idx], counts[top_idx]))


def make_wordcloud_image(text_blob, width=800, height=400):
    """Generate a word cloud image buffer from text."""
    if not text_blob.strip():
        return None
    wc = WordCloud(
        width=width,
        height=height,
        background_color="white",
        stopwords=clean_stopwords()
    )
    wc.generate(text_blob)
    fig = plt.figure(figsize=(12, 6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    buf = BytesIO()
    fig.tight
