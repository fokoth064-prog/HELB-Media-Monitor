import streamlit as st
import pandas as pd

CSV_URL = "https://docs.google.com/spreadsheets/d/10LcDId4y2vz5mk7BReXL303-OBa2QxsN3drUcefpdSQ/export?format=csv"

@st.cache_data
def load_data():
    df = pd.read_csv(CSV_URL)
    df['published_parsed'] = pd.to_datetime(df['published'], errors='coerce')
    return df

df = load_data()

st.title("📋 Mentions List")
st.write("Columns available:", df.columns.tolist())

# Search filter
search = st.text_input("Search by keyword (title/summary)")

if search:
    filtered = df[df['title'].str.contains(search, case=False, na=False) |
                  df['summary'].str.contains(search, case=False, na=False)]
else:
    filtered = df.copy()

cols_to_show = [c for c in ['published','source','tonality','title','summary','link'] if c in filtered.columns]
st.dataframe(
    filtered[cols_to_show].sort_values('published_parsed', ascending=False),
    height=500
)

)

# Download filtered
csv_bytes = filtered.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download Filtered Mentions", data=csv_bytes, file_name="helb_mentions.csv", mime="text/csv")
