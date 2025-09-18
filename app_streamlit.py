import streamlit as st

st.set_page_config(
    page_title="HELB Mentions Monitor",
    layout="wide"
)

st.title("📊 HELB Mentions Monitor")

st.sidebar.success("Select a page above ☝️")

st.markdown("""
Welcome to the **HELB Mentions Dashboard**.  
Use the sidebar to navigate:

- **Dashboard** → Key metrics, charts, and breakdowns  
- **Mentions** → Full searchable list of news mentions
""")
