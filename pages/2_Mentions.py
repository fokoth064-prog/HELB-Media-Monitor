# pages/2_Mentions.py
import streamlit as st
import pandas as pd
from utils import load_data, make_html_table

CSV_URL = "https://docs.google.com/spreadsheets/d/10LcDId4y2vz5mk7BReXL303-OBa2QxsN3drUcefpdSQ/export?format=csv"

st.title("📋 Mentions — Enhanced View")

df = load_data(CSV_URL)
if df.empty:
    st.info("No data available. Check the CSV URL and sharing settings.")
    st.stop()

# Search filter
st.sidebar.header("Filters")
search = st.sidebar.text_input("Search (TITLE / SUMMARY)")

if search:
    mask = df["TITLE"].str.contains(search, case=False, na=False) | df["SUMMARY"].str.contains(search, case=False, na=False)
    filtered = df[mask].copy()
else:
    filtered = df.copy()

st.markdown(f"**Results:** {len(filtered):,} articles (showing latest first)")

cols_to_show = ["DATE", "TIME", "SOURCE", "TONALITY", "TITLE", "SUMMARY", "LINK"]
if "published_parsed" in filtered.columns:
    filtered = filtered.sort_values(by="published_parsed", ascending=False).reset_index(drop=True)

html_table = make_html_table(filtered, cols_to_show, max_rows=500)
st.markdown(html_table, unsafe_allow_html=True)

# Download
csv_bytes = filtered[cols_to_show].to_csv(index=False).encode("utf-8")
st.download_button("📥 Download Filtered Mentions (CSV)", data=csv_bytes, file_name="helb_mentions_filtered.csv", mime="text/csv")

