# src/app.py

# -*- coding: utf-8 -*-
"""
CardioPredictNet AI: Advanced Cardiovascular Risk Assessment
=========================================================

A sophisticated, interactive web application for cardiovascular disease risk
assessment using a deep learning model.

This Streamlit application provides an elegant and intuitive interface for
healthcare professionals and individuals to evaluate heart disease risk.

Technology Stack: Streamlit, TensorFlow/Keras, Plotly, Scikit-learn
Design Philosophy: Professional, data-driven, and visually compelling.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# --- CORRECT IMPORTS FOR A MODULAR PROJECT ---
# This is the key fix: app.py now imports the logic instead of defining it.
from config import ModelConfig
from model import HeartDiseasePredictor
# ---------------------------------------------


# ============================================================================
# Page Configuration and Professional Styling
# ============================================================================

def set_page_config():
    """Configure Streamlit page settings and inject custom CSS for a stunning look."""
    st.set_page_config(
        page_title="CardioPredictNet AI",
        page_icon="❤️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for a sophisticated, dark-themed UI
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
        .stApp {
            background-color: #0f172a;
            font-family: 'Roboto', sans-serif;
        }
        .block-container {
            background: rgba(30, 41, 59, 0.6);
            border-radius: 20px;
            padding: 2rem 3rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
            margin-bottom: 2rem;
        }
        h1, h2, h3 {
            font-weight: 700;
            text-align: center;
        }
        h1 {
            background: linear-gradient(90deg, #5e72e4, #9f58e4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3rem;
        }
        h2 { color: #e2e8f0; font-size: 2.25rem; }
        h3 { color: #94a3b8; font-size: 1.5rem; }
        p, .stMarkdown, .stNumberInput, .stSelectbox { color: #cbd5e0; }
        .css-1d391kg {
            background-color: #1e293b;
            border-right: 1px solid rgba(255, 255, 255, 0.1);
        }
        .stButton > button {
            background: linear-gradient(90deg, #5e72e4, #9f58e4);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 0.75rem 2rem;
            font-size: 1.1rem;
            font-weight: 700;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(94, 114, 228, 0.4);
            width: 100%;
        }
        .stButton > button:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(94, 114, 228, 0.5);
        }
        .metric-card {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            padding: 1.5rem;
            text-align: center;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .metric-card h3 { font-size: 1.2rem; color: #94a3b8; }
        .metric-card h2 { font-size: 2.5rem; color: #e2e8f0; }
        </style>
    """, unsafe_allow_html=True)


# ============================================================================
# Visualization Functions
# ============================================================================

def create_gauge_chart(probability: float) -> go.Figure:
    """Creates an elegant gauge chart for risk probability."""
    if probability < 0.3:
        risk_level, color = "Low Risk", "#2dce89"
    elif probability < 0.7:
        risk_level, color = "Moderate Risk", "#fb6340"
    else:
        risk_level, color = "High Risk", "#f5365c"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"<b>{risk_level}</b>", 'font': {'size': 24, 'color': '#e2e8f0'}},
        number={'suffix': "%", 'font': {'size': 48, 'color': color}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': color, 'thickness': 0.3},
            'bgcolor': "rgba(255, 255, 255, 0.1)",
            'steps': [
                {'range': [0, 30], 'color': 'rgba(45, 206, 137, 0.3)'},
                {'range': [30, 70], 'color': 'rgba(251, 99, 64, 0.3)'},
                {'range': [70, 100], 'color': 'rgba(245, 54, 92, 0.3)'}],
        }))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': '#e2e8f0', 'family': 'Roboto'},
        height=350
    )
    return fig


# ============================================================================
# Main Application Logic
# ============================================================================

@st.cache_resource
def load_model_and_config():
    """Loads the model and config, cached for performance."""
    config = ModelConfig()
    predictor = HeartDiseasePredictor(config)
    predictor.load_trained_model()
    return predictor

def main():
    """The main function that runs the Streamlit application."""
    set_page_config()

    # --- Sidebar ---
    with st.sidebar:
        st.markdown("<h1>CardioPredictNet AI</h1>", unsafe_allow_html=True)
        st.markdown("### A Deep Learning Approach to Cardiovascular Risk Assessment")
        st.info("**Disclaimer:** This tool is for informational purposes only and does not constitute medical advice.")
        st.markdown("---")
        st.markdown("#### **Instructions:**\n1. Input patient data.\n2. Click 'Analyze Risk'.\n3. Review the report.")
        st.markdown("---")
        st.markdown("#### **Model Details:**\n- **Model:** TensorFlow/Keras\n- **Architecture:** Neural Network")

    # --- Main Panel ---
    st.markdown("<h1>CardioPredictNet AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#94a3b8;'>Enter patient metrics below for an AI-powered risk assessment.</p>", unsafe_allow_html=True)

    # Load the model using the cached function
    predictor = load_model_and_config()

    with st.container():
        st.markdown('<div class="block-container">', unsafe_allow_html=True)
        st.markdown("<h3>Patient Health Metrics</h3>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age (years)", 18, 100, 55)
            height = st.number_input("Height (cm)", 100, 250, 170)
            weight = st.number_input("Weight (kg)", 30, 300, 75)
        with col2:
            gender = st.selectbox("Gender", [1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
            ap_hi = st.number_input("Systolic BP (mmHg)", 80, 250, 120)
            ap_lo = st.number_input("Diastolic BP (mmHg)", 40, 150, 80)
        with col3:
            cholesterol = st.selectbox("Cholesterol", [1, 2, 3], format_func=lambda x: {1: "Normal", 2: "Above", 3: "Well Above"}[x])
            gluc = st.selectbox("Glucose", [1, 2, 3], format_func=lambda x: {1: "Normal", 2: "Above", 3: "Well Above"}[x])
            active = st.selectbox("Physical Activity", [1, 0], format_func=lambda x: "Active" if x == 1 else "Inactive")

        lifestyle_col1, lifestyle_col2 = st.columns(2)
        with lifestyle_col1:
             smoke = st.selectbox("Smoker", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        with lifestyle_col2:
             alco = st.selectbox("Consumes Alcohol", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")

        st.markdown('</div>', unsafe_allow_html=True)

    # --- Analysis Button ---
    analyze_button = st.button("🔍 Analyze Cardiovascular Risk")

    # --- Results Section ---
    if analyze_button:
        features = np.array([[
            age, gender, height, weight, ap_hi, ap_lo,
            cholesterol, gluc, smoke, alco, active
        ]], dtype=np.float32)

        with st.spinner("Performing deep learning analysis..."):
            predictions, probabilities = predictor.predict(features)
            prediction = predictions.flatten()[0]
            probability = probabilities.flatten()[0]

        with st.container():
            st.markdown('<div class="block-container">', unsafe_allow_html=True)
            st.markdown("<h2>Risk Analysis Report</h2>", unsafe_allow_html=True)

            res_col1, res_col2 = st.columns([2, 1])
            with res_col1:
                st.plotly_chart(create_gauge_chart(probability), use_container_width=True)
            with res_col2:
                st.markdown("<h3>Key Metrics</h3>", unsafe_allow_html=True)
                result_text = "High Risk" if prediction == 1 else "Low Risk"
                st.markdown(f"""
                    <div class='metric-card'>
                        <h3>Predicted Risk</h3>
                        <h2 style='color: {"#f5365c" if prediction == 1 else "#2dce89"};'>{result_text}</h2>
                    </div>
                """, unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"""
                    <div class='metric-card'>
                        <h3>Prediction Confidence</h3>
                        <h2>{probability*100:.1f}%</h2>
                    </div>
                """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("<h3>Interpretation & Recommendations</h3>", unsafe_allow_html=True)
            if prediction == 1:
                st.warning("""
                **High Risk Detected:** The model predicts a significant probability of cardiovascular disease.
                **Recommendations:**
                - **Immediate Consultation:** It is strongly advised to consult a healthcare professional.
                - **Lifestyle Review:** Discuss diet, exercise, and lifestyle choices with your doctor.
                """)
            else:
                st.success("""
                **Low Risk Detected:** The model predicts a low probability of cardiovascular disease.
                **Recommendations:**
                - **Maintain Healthy Habits:** Continue with a balanced diet and regular physical activity.
                - **Regular Check-ups:** Continue with routine health check-ups.
                """)
            st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()