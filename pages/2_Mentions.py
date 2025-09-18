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
        df["DATE"] = ""
        df["TIME"] = ""

    # Ensure text columns exist
    for col in ["title", "summary", "source", "tonality", "link"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("")

    # Rename all to UPPERCASE
    rename_map = {
        "title": "TITLE",
        "summary": "SUMMARY",
        "source": "SOURCE",
        "tonality": "TONALITY",
        "link": "LINK",
    }
    df = df.rename(columns=rename_map)

    return df

# ---------------- MAIN APP ----------------
df = load_data(CSV_URL)

st.title("📋 Mentions List")

if df.empty:
    st.info("No data available. Check that the CSV URL is correct and the sheet is shared publicly.")
else:
    # --- Search filter ---
    search = st.text_input("Search by keyword (TITLE/SUMMARY)")
    if search:
        mask = (
            df["TITLE"].str.contains(search, case=False, na=False)
            | df["SUMMARY"].str.contains(search, case=False, na=False)
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

    # Make LINK column clickable
    if "LINK" in filtered.columns:
        filtered["LINK"] = filtered["LINK"].apply(
            lambda x: f"[Open Article]({x})" if isinstance(x, str) and x.startswith("http") else ""
        )

    st.dataframe(filtered[cols_to_show].reset_index(drop=True), height=500)

    # --- Download filtered ---
    csv_bytes = filtered[cols_to_show].to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Filtered Mentions",
        data=csv_bytes,
        file_name="helb_mentions_filtered.csv",
        mime="text/csv",
    )
