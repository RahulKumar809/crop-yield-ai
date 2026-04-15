import streamlit as st
import pandas as pd
import os
import numpy as np
import sklearn.compose._column_transformer as sklearn_column_transformer
from langchain_groq import ChatGroq
from langchain_community.chat_message_histories import ChatMessageHistory
import joblib
from scipy import sparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize LLM (optional)
llm = ChatGroq(model="llama-3.1-8b-instant", api_key=groq_api_key) if groq_api_key else None

# Page config
st.set_page_config(page_title='Agriculture Yield Advisor', page_icon='🌾')
st.title('KrishiMitra AI 🌾')

# Session states
if "history" not in st.session_state:
    st.session_state.history = ChatMessageHistory()

if "predicted_yield" not in st.session_state:
    st.session_state.predicted_yield = None

if "llm_explanation" not in st.session_state:
    st.session_state.llm_explanation = None

if "yield_pred" not in st.session_state:
    st.session_state.yield_pred = None


# Compatibility patch
def apply_sklearn_pickle_compatibility_patch():
    if not hasattr(sklearn_column_transformer, "_RemainderColsList"):
        class _RemainderColsList(list):
            pass
        sklearn_column_transformer._RemainderColsList = _RemainderColsList


# Load model (FIXED ✅)
@st.cache_resource
def load_pipeline_model():
    apply_sklearn_pickle_compatibility_patch()
    return joblib.load("crop_yield.joblib")


# Load dataset
@st.cache_data
def load_dataset():
    return pd.read_csv("new_df.csv")


# Load model safely
try:
    pipeline = load_pipeline_model()
except Exception as exc:
    st.error(f"Model could not be loaded: {exc}")
    st.info("Install: pip install scikit-learn==1.6.1")
    st.stop()


# Load data
df = load_dataset()

crop_list = sorted(df["Crop"].dropna().unique())
state_list = sorted(df["State"].dropna().unique())
season_list = sorted(df["Season"].dropna().unique())

# UI Inputs
crop = st.selectbox("Select Crop", crop_list)
season = st.selectbox("Select Season", season_list)
state = st.selectbox("Select State", state_list)

area = st.number_input("Enter Cultivated Area (in hectares)")
rainfall = st.number_input("Enter Annual Rainfall (mm)")
fertilizer = st.number_input("Fertilizer Used (kg)")
pesticide = st.number_input("Pesticide Used (kg)")


# Prediction function
def predict_yield():
    input_data = pd.DataFrame({
        "Season": [season],
        "State": [state],
        "Annual_Rainfall": [rainfall],
        "Fertilizer": [fertilizer],
        "Pesticide": [pesticide],
        "Crop": [crop],
        "Area": [area],
    })

    try:
        return pipeline.predict(input_data)[0]
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None


# Predict button
if st.button("Predict"):
    pred = predict_yield()

    if pred is not None:
        st.success(f"🌾 Predicted Yield: {pred:.2f} tonnes")

        if llm:
            prompt = f"""
            You are an agriculture expert.

            Yield predicted: {pred:.2f} tonnes

            Explain:
            1. Why this yield?
            2. How to improve?
            3. Govt schemes in India

            Keep it simple and farmer-friendly.
            """

            try:
                response = llm.invoke(prompt)
                st.write("### 🤖 Advisor:")
                st.write(response.content)
            except Exception as e:
                st.error(f"LLM error: {e}")
        else:
            st.warning("Add GROQ_API_KEY for AI explanation")