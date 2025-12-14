import streamlit as st

# SIDEBAR
product_detection_page = st.Page("product_detection.py", title="Product Detection", icon="🔎")
about_page = st.Page("about.py", title="About", icon="ℹ️")
tutorial_page = st.Page("tutorial.py", title="Tutorial", icon="❓")
pg=st.navigation([ product_detection_page, tutorial_page, about_page])
pg.run()