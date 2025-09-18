# pages/2_Mentions.py
import streamlit as st
import pandas as pd

# ---------------- CONFIG ----------------
CSV_URL = "https://docs.google.com/spreadsheets/d/10LcDId4y2vz5mk7BReXL303-OBa2QxsN3drUcefpdSQ/export?format=csv"
# Replace above with your own sheet export link if different.

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data(csv_url):
    try:
        df = pd.read_csv(csv_url)
    except Exception as e:
        # Return an empty DataFrame but keep the error visible to the app
        st.error(f"Failed to load CSV from URL: {e}")
        return pd.DataFrame()

    # Show available columns for debugging
    st.write("Columns available in CSV:", df.columns.tolist())

    # Normalize column names (trim whitespace)
    df.columns = [c.strip() for c in df.columns]

    # Ensure required columns exist; create fallback columns if missing
    if 'published' not in df.columns:
        # try common alternate names
        for alt in ['published date', 'published_date', 'publish_date', 'date']:
            if alt in df.columns:
                df['published'] = df[alt]
                break
    if 'published' not in df.columns:
        # create empty published column to avoid KeyErrors later
        df['published'] = pd.NaT

    # Parse published into datetime
    df['published_parsed'] = pd.to_datetime(df['published'], errors='coerce')

    # Ensure textual columns exist to avoid KeyErrors
    for col in ['title', 'summary', 'source', 'tonality', 'link']:
        if col not in df.columns:
            df[col] = ""  # empty fallback

    # Fill NaNs in searchable text columns
    df['title'] = df['title'].fillna('')
    df['summary'] = df['summary'].fillna('')

    return df

# Load
df = load_data(CSV_URL)

st.title("📋 Mentions List")

# If load failed, df will be empty
if df.empty:
    st.info("No data available. Check that the CSV URL is correct and the sheet is shared publicly.")
else:
    # --- Search/filter ---
    search = st.text_input("Search by keyword (title/summary)")
    if search:
        mask = df['title'].str.contains(search, case=False, na=False) | df['summary'].str.contains(search, case=False, na=False)
        filtered = df[mask].copy()
    else:
        filtered = df.copy()

    # Show the number of results
    st.markdown(f"**Results:** {len(filtered):,} articles")

    # Columns to display (only those present)
    wanted = ['published', 'source', 'tonality', 'title', 'summary', 'link']
    cols_to_show = [c for c in wanted if c in filtered.columns]

    # Sort by published date if present
    if 'published_parsed' in filtered.columns:
        filtered = filtered.sort_values(by='published_parsed', ascending=False)

    # Display
    st.dataframe(filtered[cols_to_show].reset_index(drop=True), height=500)

    # Download filtered CSV
    csv_bytes = filtered[cols_to_show].to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Filtered Mentions", data=csv_bytes, file_name="helb_mentions_filtered.csv", mime="text/csv")
