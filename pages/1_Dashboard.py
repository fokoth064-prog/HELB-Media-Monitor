import streamlit as st
import pandas as pd
import plotly.express as px

# ---------- CONFIG ----------
CSV_URL = "https://docs.google.com/spreadsheets/d/10LcDId4y2vz5mk7BReXL303-OBa2QxsN3drUcefpdSQ/export?format=csv"

# ---------- LOAD DATA ----------
@st.cache_data
def load_data():
    df = pd.read_csv(CSV_URL)
    df['published_parsed'] = pd.to_datetime(df['published'], errors='coerce')
    df['year'] = df['published_parsed'].dt.year

    # Financial Year
    df['fin_year'] = df['published_parsed'].apply(
        lambda x: f"{x.year}/{x.year+1}" if x.month >= 7 else f"{x.year-1}/{x.year}"
    )

    # Financial Quarter
    def get_fin_quarter(date):
        m = date.month
        if 7 <= m <= 9:
            return "Q1 (Jul–Sep)"
        elif 10 <= m <= 12:
            return "Q2 (Oct–Dec)"
        elif 1 <= m <= 3:
            return "Q3 (Jan–Mar)"
        else:
            return "Q4 (Apr–Jun)"
    df['fin_quarter'] = df['published_parsed'].apply(get_fin_quarter)
    return df

df = load_data()

st.title("📊 Dashboard")

# ---------- FILTERS ----------
st.sidebar.header("Filters")

years = sorted(df['year'].dropna().unique())
year_filter = st.sidebar.multiselect("Year", options=years, default=years)

fin_years = sorted(df['fin_year'].dropna().unique())
fin_year_filter = st.sidebar.multiselect("Financial Year", options=fin_years, default=fin_years)

quarters = df['fin_quarter'].unique()
quarter_filter = st.sidebar.multiselect("Quarter", options=quarters, default=quarters)

filtered = df[
    df['year'].isin(year_filter) &
    df['fin_year'].isin(fin_year_filter) &
    df['fin_quarter'].isin(quarter_filter)
]

# ---------- METRIC TILES ----------
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Mentions", len(filtered))
with col2:
    pos = (filtered['tonality'] == "Positive").mean() * 100 if not filtered.empty else 0
    st.metric("Positive %", f"{pos:.1f}%")
with col3:
    neg = (filtered['tonality'] == "Negative").mean() * 100 if not filtered.empty else 0
    st.metric("Negative %", f"{neg:.1f}%")
with col4:
    if not filtered.empty:
        st.metric("Date Range", f"{filtered['published_parsed'].min().date()} → {filtered['published_parsed'].max().date()}")

# ---------- VISUALS ----------
if not filtered.empty:
    col1, col2 = st.columns(2)
    with col1:
        fig_tonality = px.pie(filtered, names="tonality", title="Tonality Breakdown")
        st.plotly_chart(fig_tonality, use_container_width=True)

    with col2:
        fig_sources = px.bar(filtered['source'].value_counts().head(5),
                             title="Top 5 Sources")
        st.plotly_chart(fig_sources, use_container_width=True)

    st.subheader("Mentions Trend Over Time")
    timeseries = filtered.groupby(filtered['published_parsed'].dt.date).size().reset_index(name="count")
    fig_time = px.line(timeseries, x="published_parsed", y="count", title="Mentions Trend")
    st.plotly_chart(fig_time, use_container_width=True)
else:
    st.warning("No data available for selected filters.")
