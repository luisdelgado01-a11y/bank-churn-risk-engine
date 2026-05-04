import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Bank Churn Risk Engine", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load("models/model.pkl")

model = load_model()

st.title("🏦 Bank Churn Risk & Retention Engine")
st.write("Predict whether a banking customer is at risk of churn and recommend a retention action.")
st.write("Built by Luis Delgado • FP&A | Data Analytics | Decision Intelligence")

st.sidebar.header("Customer Inputs")

credit_score = st.sidebar.slider("Credit Score", 350, 850, 650)
geography = st.sidebar.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
age = st.sidebar.slider("Age", 18, 92, 40)
tenure = st.sidebar.slider("Tenure", 0, 10, 5)
balance = st.sidebar.number_input("Balance", min_value=0.0, max_value=300000.0, value=50000.0, step=1000.0)
num_products = st.sidebar.selectbox("Number of Products", [1, 2, 3, 4])
has_cr_card = st.sidebar.selectbox("Has Credit Card?", ["Yes", "No"])
is_active = st.sidebar.selectbox("Active Member?", ["Yes", "No"])
estimated_salary = st.sidebar.number_input("Estimated Salary", min_value=0.0, max_value=250000.0, value=75000.0, step=1000.0)

input_df = pd.DataFrame([{
    "CreditScore": credit_score,
    "Geography": geography,
    "Gender": gender,
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": num_products,
    "HasCrCard": 1 if has_cr_card == "Yes" else 0,
    "IsActiveMember": 1 if is_active == "Yes" else 0,
    "EstimatedSalary": estimated_salary
}])

probability = model.predict_proba(input_df)[0][1]
prediction = model.predict(input_df)[0]

st.subheader("Customer Profile")
st.dataframe(input_df, use_container_width=True)

st.subheader("Prediction Result")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Churn Probability", f"{probability:.1%}")

with col2:
    if probability >= 0.60:
        st.metric("Risk Tier", "High")
    elif probability >= 0.30:
        st.metric("Risk Tier", "Medium")
    else:
        st.metric("Risk Tier", "Low")

with col3:
    estimated_value_at_risk = (balance * 0.02) + (estimated_salary * 0.01)
    st.metric("Estimated Annual Value at Risk", f"${estimated_value_at_risk:,.0f}")

if probability >= 0.60:
    st.error("HIGH RISK: Immediate retention action recommended.")
    st.write("Recommended action: relationship manager outreach, fee review, personalized offer, and follow-up within 7 days.")
elif probability >= 0.30:
    st.warning("MEDIUM RISK: Customer should be monitored and targeted for engagement.")
    st.write("Recommended action: personalized email, product usage review, loyalty incentive, or service check-in.")
else:
    st.success("LOW RISK: Customer appears stable.")
    st.write("Recommended action: continue standard relationship management.")

if probability >= 0.60:
    st.error("🚨 HIGH RISK CUSTOMER")
    st.write("💸 Estimated Loss Risk: Customer likely to churn without intervention.")
elif probability >= 0.30:
    st.warning("⚠️ MEDIUM RISK CUSTOMER")
    st.write("📉 Early signs of disengagement detected.")
else:
    st.success("✅ LOW RISK CUSTOMER")
    st.write("📈 Stable customer profile.")

st.divider()

st.subheader("Business Interpretation")

if probability >= 0.60:
    st.write(
        "This customer shows strong churn indicators. The bank should prioritize intervention because the cost of losing the customer may exceed the cost of a targeted retention offer."
    )
elif probability >= 0.30:
    st.write(
        "This customer has moderate churn risk. A low-cost engagement campaign may reduce attrition before the customer becomes high risk."
    )
else:
    st.write(
        "This customer has low predicted churn risk. Retention budget should be focused on higher-risk customers."
    )

with st.expander("About this model"):
    st.write(
        "This app uses a machine learning classification model trained on banking customer data to estimate churn probability."
    )
    st.write(
        "The model considers credit score, geography, gender, age, tenure, balance, number of products, credit card ownership, activity status, and estimated salary."
    )
    st.divider()

st.subheader("About the Creator")

st.write(
    "This project was built by Luis Delgado, a Senior Data Analyst specializing in data-driven decision systems, "
    "operational analytics, and predictive modeling. The goal of this tool is to translate customer data into "
    "clear business decisions and financial impact."
)