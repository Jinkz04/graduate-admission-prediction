import streamlit as st
import joblib
import pandas as pd

# Load models
linear_model = joblib.load('models/linear_reg.pkl')
ridge_model = joblib.load('models/ridge_reg.pkl')
lasso_model = joblib.load('models/lasso_reg.pkl')
elastic_net_model = joblib.load('models/elastic_net.pkl')

# Set page config
st.set_page_config(page_title="Graduate Admission Predictor", layout="centered")

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0px 0px 12px rgba(0,0,0,0.1);
        font-family: 'Segoe UI', sans-serif;
    }
    h1 {
        color: #004466;
        text-align: center;
    }
    .stSlider > div > div {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 10px;
    }
    .prediction {
        font-size: 1.5rem;
        color: #006400;
        text-align: center;
        margin-top: 1.5rem;
    }
    </style>
""", unsafe_allow_html=True)

with st.container():
    st.title("🎓 Graduate Admission Predictor")

    with st.sidebar:
        st.header("Input Features")

        gre_score = st.slider("GRE Score", 280, 340, 310)
        toefl_score = st.slider("TOEFL Score", 80, 120, 100)
        university_rating = st.slider("University Rating", 1, 5, 3)
        sop = st.slider("SOP Strength", 1.0, 5.0, 3.0)
        lor = st.slider("LOR Strength", 1.0, 5.0, 3.0)
        cgpa = st.slider("CGPA", 6.0, 10.0, 8.0)
        research = st.radio("Research Experience", ["No", "Yes"])

        model_choice = st.selectbox("Choose Model", [
            "Linear Regression", "Ridge Regression", "Lasso Regression", "Elastic Net"
        ])

    research = 1 if research == "Yes" else 0

    # Use exact column names from training
    input_data = pd.DataFrame([[
        gre_score, toefl_score, university_rating, sop, lor, cgpa, research
    ]], columns=['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA', 'Research'])

    model_map = {
        "Linear Regression": linear_model,
        "Ridge Regression": ridge_model,
        "Lasso Regression": lasso_model,
        "Elastic Net": elastic_net_model
    }

    model = model_map[model_choice]
    prediction = model.predict(input_data)[0]

    st.markdown(f"<div class='prediction'>🎯 Predicted Chance of Admission: <strong>{prediction:.2f}</strong></div>", unsafe_allow_html=True)
