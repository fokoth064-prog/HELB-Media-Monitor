# pages/2_Mentions.py
import streamlit as st
import pandas as pd
from utils import load_data, make_html_table

# ---------------- STREAMLIT PAGE CONFIG ----------------
st.set_page_config(page_title="Mentions", layout="wide")
st.title("📋 Mentions — Enhanced View")

# ---------------- LOAD DATA ----------------
df = load_data()
if df.empty:
    st.info("No data available. Check the CSV URL and sharing settings.")
    st.stop()

# ---------------- FILTERS ----------------
st.sidebar.header("Filters")
search = st.sidebar.text_input("Search (TITLE / SUMMARY)")

# Apply search filter
if search:
    mask = (
        df["TITLE"].str.contains(search, case=False, na=False)
        | df["SUMMARY"].str.contains(search, case=False, na=False)
    )
    filtered = df[mask].copy()
else:
    filtered = df.copy()

st.markdown(f"**Results:** {len(filtered):,} articles (showing latest first)")

# ---------------- SORT & DISPLAY ----------------
# Ensure sorted by published_parsed if present
if "published_parsed" in filtered.columns:
    filtered = filtered.sort_values(by="published_parsed", ascending=False).reset_index(
        drop=True
    )

# Columns to show
cols_to_show = ["DATE", "TIME", "SOURCE", "TONALITY", "TITLE", "SUMMARY", "LINK"]
cols_to_show = [c for c in cols_to_show if c in filtered.columns]

# Build HTML table
html_table = make_html_table(filtered, cols_to_show, max_rows=500)
st.markdown(html_table, unsafe_allow_html=True)

# ---------------- CSV DOWNLOAD ----------------
csv_bytes = filtered[cols_to_show].to_csv(index=False).encode("utf-8")
st.download_button(
    "📥 Download Filtered Mentions (CSV)",
    data=csv_bytes,
    file_name="helb_mentions_filtered.csv",
    mime="text/csv",
)
