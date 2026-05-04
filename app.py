import streamlit as st
import pandas as pd
st.set_page_config(page_title="Churn Risk Engine", layout="wide")
st.title("🏦 Bank Churn Risk & Retention Engine")
st.write("App is running successfully ✅")
st.sidebar.header("Customer Inputs")
credit_score = st.sidebar.slider("Credit Score", 350, 850, 650)
age = st.sidebar.slider("Age", 18, 92, 40)
st.write("### Inputs")
st.write(f"Credit Score: {credit_score}")
st.write(f"Age: {age}")