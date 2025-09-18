# pages/2_Mentions.py
import streamlit as st
import pandas as pd

# ---------------- CONFIG ----------------
CSV_URL = "https://docs.google.com/spreadsheets/d/10LcDId4y2vz5mk7BReXL303-OBa2QxsN3drUcefpdSQ/export?format=csv"

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data(csv_url):
    try:
        df = pd.read_csv(csv_url)
    except Exception as e:
        st.error(f"❌ Failed to load CSV: {e}")
        return pd.DataFrame()

    # Normalize column names
    df.columns = [c.strip() for c in df.columns]

    # Ensure published_parsed column exists
    if "published" in df.columns:
        # Parse and localize to Nairobi (EAT, UTC+3)
        df["published_parsed"] = pd.to_datetime(df["published"], errors="coerce", utc=True)
        df["published_parsed"] = df["published_parsed"].dt.tz_convert("Africa/Nairobi")

        # Create clean date and time columns
        df["date_only"] = df["published_parsed"].dt.date
        df["time_only"] = df["published_parsed"].dt.strftime("%H:%M")
    else:
        df["published_parsed"] = pd.NaT
        df["date_only"] = ""
        df["time_only"] = ""

    # Fill NaNs for text columns
    for col in ["title", "summary", "source", "tonality", "link"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("")

    return df

# Load
df = load_data(CSV_URL)

st.title("📋 Mentions List")

if df.empty:
    st.info("No data available. Check that the CSV URL is correct and the sheet is shared publicly.")
else:
    # --- Search filter ---
    search = st.text_input("Search by keyword (title/summary)")
    if search:
        mask = (
            df["title"].str.contains(search, case=False, na=False)
            | df["summary"].str.contains(search, case=False, na=False)
        )
        filtered = df[mask].copy()
    else:
        filtered = df.copy()

    # --- Display results ---
    st.markdown(f"**Results:** {len(filtered):,} articles")

    cols_to_show = ["DATE", "TIME", "SOURCE", "TONALITY", "TITLE", "SUMMARY", "LINK"]
    cols_to_show = [c for c in cols_to_show if c in filtered.columns]

    if "published_parsed" in filtered.columns:
        filtered = filtered.sort_values(by="published_parsed", ascending=False)

    st.dataframe(filtered[cols_to_show].reset_index(drop=True), height=500)

    # --- Download filtered ---
    csv_bytes = filtered[cols_to_show].to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Filtered Mentions",
        data=csv_bytes,
        file_name="helb_mentions_filtered.csv",
        mime="text/csv",
    )
