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
    df.columns = [c.strip().lower() for c in df.columns]

    # Ensure published_parsed column exists
    if "published" in df.columns:
        df["published_parsed"] = pd.to_datetime(df["published"], errors="coerce", utc=True)
        df["published_parsed"] = df["published_parsed"].dt.tz_convert("Africa/Nairobi")

        # Create clean DATE and TIME columns
        df["DATE"] = df["published_parsed"].dt.strftime("%d-%b-%Y")
        df["TIME"] = df["published_parsed"].dt.strftime("%H:%M")
    else:
        df["published_parsed"] = pd.NaT
        df["DATE"] =
