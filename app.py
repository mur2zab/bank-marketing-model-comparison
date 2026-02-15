import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Bank Marketing ML", layout="wide")

st.markdown("""
<style>
    [data-testid="stSidebar"] {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model(model_name):
    filename = f"model_{model_name.replace(' ', '_').lower()}.pkl"
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except:
        return None

@st.cache_resource
def load_scaler():
    try:
        with open('scaler.pkl', 'rb') as f:
            return pickle.load(f)
    except:
        return None

st.title("Bank Marketing Prediction")
st.markdown("---")

if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None
if 'results' not in st.session_state:
    st.session_state.results = None

st.header("Step 1: Upload Dataset")

if st.session_state.uploaded_file is None:
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
        st.rerun()
else:
    df = pd.read_csv(st.session_state.uploaded_file)
    st.success(f"✓ Dataset: {len(df):,} records")
    
    if st.button("Change File"):
        st.session_state.uploaded_file = None
        st.session_state.selected_model = None
        st.session_state.results = None
        st.rerun()

if st.session_state.uploaded_file is not None:
    st.markdown("---")
    st.header("Step 2: Select Model")
    
    model_options = ['Logistic Regression', 'Decision Tree', 'K-Nearest Neighbors', 
                     'Naive Bayes', 'Random Forest', 'XGBoost']
    
    selected_model = st.radio("Choose algorithm:", model_options)
    st.session_state.selected_model = selected_model

if st.session_state.uploaded_file is not None and st.session_state.selected_model is not None:
    st.markdown("---")
    st.header("Step 3: Get Predictions")
    
    if st.button("Run Analysis"):
        df = pd.read_csv(st.session_state.uploaded_file)
        
        if 'y' in df.columns:
            X_test = df.drop('y', axis=1)
            y_test = df['y']
        else:
            X_test = df
            y_test = None
        
        model = load_model(st.session_state.selected_model)
        scaler = load_scaler()
        
        if model:
            if st.session_state.selected_model in ['Logistic Regression', 'K-Nearest Neighbors', 'Naive Bayes']:
                X_test_processed = scaler.transform(X_test) if scaler else X_test
            else:
                X_test_processed = X_test
            
            with st.spinner('Running analysis...'):
                y_pred = model.predict(X_test_processed)
                
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_test_processed)[:, 1]
                else:
                    y_pred_proba = y_pred
                
                st.session_state.results = {
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba,
                    'y_test': y_test
                }
                
                st.success("✓ Analysis complete!")
                st.rerun()

if st.session_state.results is not None:
    results = st.session_state.results
    y_test = results['y_test']
    y_pred = results['y_pred']
    y_pred_proba = results['y_pred_proba']
    
    st.markdown("---")
    st.header("Results")
    
    if y_test is not None:
        # Metrics
        st.subheader("Performance Metrics")
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
        with col2:
            st.metric("AUC", f"{roc_auc_score(y_test, y_pred_proba):.4f}")
        with col3:
            st.metric("Precision", f"{precision_score(y_test, y_pred):.4f}")
        with col4:
            st.metric("Recall", f"{recall_score(y_test, y_pred):.4f}")
        with col5:
            st.metric("F1", f"{f1_score(y_test, y_pred):.4f}")
        with col6:
            st.metric("MCC", f"{matthews_corrcoef(y_test, y_pred):.4f}")
        
        # Confusion Matrix
        st.subheader("Confusion Matrix")
        
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No', 'Yes'],
                   yticklabels=['No', 'Yes'],
                   ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)
        plt.close()

# Footer
st.markdown("---")
st.caption("ML Assignment 2 • Bank Marketing Dataset")