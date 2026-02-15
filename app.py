import streamlit as st
import pandas as pd

st.set_page_config(page_title="Bank Marketing ML", layout="wide")

st.markdown("""
<style>
    [data-testid="stSidebar"] {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

st.title("Bank Marketing Prediction")
st.markdown("---")
st.write("ML Classification System - Step by Step Analysis")

if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None


st.header("Step 1: Upload Dataset")
if st.session_state.uploaded_file is None:
    uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
    
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
        st.rerun()
else:
    df = pd.read_csv(st.session_state.uploaded_file)
    st.success(f"✓ Dataset loaded: {len(df):,} records, {df.shape[1]} features")
    
    with st.expander("View Data Preview"):
        st.dataframe(df.head())
    
    if st.button("Change File"):
        st.session_state.uploaded_file = None
        st.session_state.selected_model = None
        st.rerun()

if st.session_state.uploaded_file is not None:
    st.markdown("---")
    st.header("Step 2: Select Model")
    
    model_options = [
        'Logistic Regression',
        'Decision Tree',
        'K-Nearest Neighbors',
        'Naive Bayes',
        'Random Forest',
        'XGBoost'
    ]
    
    selected_model = st.radio(
        "Choose a machine learning algorithm:",
        model_options,
        index=0 if st.session_state.selected_model is None else model_options.index(st.session_state.selected_model)
    )
    
    if selected_model != st.session_state.selected_model:
        st.session_state.selected_model = selected_model
    
    if st.session_state.selected_model:
        st.success(f"✓ Model selected: {st.session_state.selected_model}")

if st.session_state.uploaded_file is not None and st.session_state.selected_model is not None:
    st.markdown("---")
    st.header("Step 3: Get Predictions")
    st.button("Run Analysis", disabled=True)

st.markdown("---")
st.caption("ML Assignment 2 • Bank Marketing Dataset")