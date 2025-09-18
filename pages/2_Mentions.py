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
        df["published_parsed"] = pd.to_datetime(df["published"], errors="coerce", utc=True)
        df["published_parsed"] = df["published_parsed"].dt.tz_convert("Africa/Nairobi")

        # Create clean DATE and TIME columns
        df["DATE"] = df["published_parsed"].dt.date
        df["TIME"] = df["published_parsed"].dt.strftime("%H:%M")
    else:
        df["published_parsed"] = pd.NaT
        df["DATE"] = ""
        df["TIME"] = ""

    # Fill NaNs for text columns
    for col in ["title", "summary", "source", "tonality", "link"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("")

    # Rename columns to uppercase for consistency
    rename_map = {
        "title": "TITLE",
        "summary": "SUMMARY",
        "source": "SOURCE",
        "tonality": "TONALITY",
        "link": "LINK",
    }
    df = df.rename(columns=rename_map)

    return df

# Load
df = load_data(CSV_URL)

st.title("📋 Mentions List")

if df.empty:
    st.info("No data available. Check that the CSV URL is correct and the sheet is shared publicly.")
else:
    # --- Search filter ---
    search = st.text_input_
