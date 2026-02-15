import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Bank Marketing ML", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;600&display=swap');
              
    /* Dark theme */
    body {
        color: #ffffff;
    }
    .stApp {
        background: #0a0a0a;
        font-family: 'IBM Plex Sans', sans-serif;
    }
    
    /* Hide sidebar */
    [data-testid="stSidebar"] {
        display: none;
    }
    
    /* Typography */
    h1, h2, h3, h4 {
        color: #ffffff;
        font-weight: 300;
        letter-spacing: -0.02em;
    }
    
    p, div, span, label {
        color: #b5b5b5;
    }
    
    /* Buttons */
    .stButton > button {
        background: #ffffff;
        color: #000000;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: 600;
        text-transform: uppercase;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: #e5e5e5;
        box-shadow: 0 4px 12px rgba(255, 255, 255, 0.2);
    }
    
    /* Download button styling */
    .stDownloadButton > button {
        background: #0a0a0a;
        color: #ffffff;
        border: 2px solid #ffffff;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        text-transform: uppercase;
        transition: all 0.3s ease;
    }
    
    .stDownloadButton > button:hover {
        background: #ffffff;
        color: #0a0a0a;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 300;
        color: #ffffff;
        font-family: 'IBM Plex Mono', monospace;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.75rem;
        text-transform: uppercase;
        color: #8a8a8a;
        font-weight: 600;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        border: 2px dashed #3a3a3a;
        background: #0f0f0f;
        padding: 2rem;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

if 'df' not in st.session_state:
    st.session_state.df = None
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None
if 'comparison_results' not in st.session_state:
    st.session_state.comparison_results = []
if 'current_results' not in st.session_state:
    st.session_state.current_results = None

@st.cache_resource
def load_model(model_name):
    filename = f"./model/model_{model_name.replace(' ', '_').lower()}.pkl"
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

@st.cache_resource
def load_scaler():
    try:
        with open('scaler.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None
    
def plot_styled_cm(cm):
    fig, ax = plt.subplots(figsize=(2.5, 1.8))
    
    fig.patch.set_facecolor('#121212')
    ax.set_facecolor('#121212')
    
    cmap = sns.dark_palette("#4a9eff", as_cmap=True)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, cbar=False,
                annot_kws={"size": 9, "family": "IBM Plex Mono", "weight": "bold"},
                linewidths=0.5, linecolor='#121212')
    
    ax.set_xticklabels(['NO', 'YES'], color='#888', family='IBM Plex Mono', fontsize=7)
    ax.set_yticklabels(['NO', 'YES'], color='#888', family='IBM Plex Mono', fontsize=7)
    ax.set_xlabel('PREDICTED', color='#4a9eff', family='IBM Plex Mono', fontsize=6)
    ax.set_ylabel('ACTUAL', color='#4a9eff', family='IBM Plex Mono', fontsize=6)
    
    for _, spine in ax.spines.items():
        spine.set_visible(False)
        
    plt.tight_layout(pad=0)
    return fig

def get_predictions(model, X_test, y_test, model_name, scaler):
    needs_scaling = ['Logistic Regression', 'K-Nearest Neighbors', 'Naive Bayes']
    if model_name in needs_scaling and scaler is not None:
        X_test_processed = scaler.transform(X_test)
    else:
        X_test_processed = X_test
    
    y_pred = model.predict(X_test_processed)
    y_pred_proba = model.predict_proba(X_test_processed)[:, 1] if hasattr(model, 'predict_proba') else y_pred
    
    results = {
        'model_name': model_name,
        'accuracy': accuracy_score(y_test, y_pred) if y_test is not None else None,
        'auc': roc_auc_score(y_test, y_pred_proba) if y_test is not None else None,
        'precision': precision_score(y_test, y_pred) if y_test is not None else None,
        'recall': recall_score(y_test, y_pred) if y_test is not None else None,
        'f1': f1_score(y_test, y_pred) if y_test is not None else None,
        'mcc': matthews_corrcoef(y_test, y_pred) if y_test is not None else None,
        'y_pred': y_pred,
        'confusion_matrix': confusion_matrix(y_test, y_pred) if y_test is not None else None
    }
    return results

st.markdown('<div class="app-header"><h1>Bank Marketing Prediction</h1><div class="accent-line"></div><p class="subtitle">ML Classification System</p></div>', unsafe_allow_html=True)

step1_active = st.session_state.df is None
st.markdown(f'<div class="step-header"><div class="step-number {"active" if step1_active else "completed"}">{"Step 1" if step1_active else "✓"}</div><h2 class="step-title">Upload Dataset</h2></div>', unsafe_allow_html=True)

if st.session_state.df is None:
    st.markdown("**Need a sample file to test?**")
    try:
        with open('./model/test_data.csv', 'r') as f:
            sample_data = f.read()
        st.download_button(
            label="Download Sample Test Data",
            data=sample_data,
            file_name='sample_test_data.csv',
            mime='text/csv',
            help="Download a sample CSV file to test the application",
            use_container_width=True
        )
    except FileNotFoundError:
        try:
            with open('./model/test_data_500.csv', 'r') as f:
                sample_data = f.read()
            st.download_button(
                label="Download Sample Test Data (500 records)",
                data=sample_data,
                file_name='sample_test_data.csv',
                mime='text/csv',
                help="Download a sample CSV file to test the application",
                use_container_width=True
            )
        except FileNotFoundError:
            pass
    
    st.markdown("---")
    st.markdown("**Upload your dataset:**")
    
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'], label_visibility="collapsed")
    if uploaded_file:
        st.session_state.df = pd.read_csv(uploaded_file)
        st.rerun()
else:
    st.markdown(f'<div class="status-badge">✓ {len(st.session_state.df)} Records Loaded</div>', unsafe_allow_html=True)
    if st.button("Reset / Upload New File"):
        st.session_state.df = None
        st.session_state.current_results = None
        st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

if st.session_state.df is not None:
    step2_active = st.session_state.selected_model is None
    st.markdown(f'<div class="step-header"><div class="step-number {"active" if step2_active else "completed"}">{"Step 2" if step2_active else "✓"}</div><h2 class="step-title">Select Model</h2></div>', unsafe_allow_html=True)
    
    model_list = ['Logistic Regression', 'Decision Tree', 'K-Nearest Neighbors', 'Naive Bayes', 'Random Forest', 'XGBoost']
    selected = st.selectbox("Choose algorithm", model_list, index=None, placeholder="Select a model...")
    
    if selected and selected != st.session_state.selected_model:
        st.session_state.selected_model = selected
        st.session_state.current_results = None
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

if st.session_state.selected_model and st.session_state.df is not None:
    st.markdown('<div class="step-header"><div class="step-number active">Step 3</div><h2 class="step-title">Run Analysis</h2></div>', unsafe_allow_html=True)
    
    if st.button("Execute Prediction", use_container_width=True):
        df = st.session_state.df
        X_test = df.drop('y', axis=1) if 'y' in df.columns else df
        y_test = df['y'] if 'y' in df.columns else None
        
        model = load_model(st.session_state.selected_model)
        scaler = load_scaler()
        
        if model:
            st.session_state.current_results = get_predictions(model, X_test, y_test, st.session_state.selected_model, scaler)
            st.rerun()
        else:
            st.error(f"Model file for {st.session_state.selected_model} not found.")
    st.markdown('</div>', unsafe_allow_html=True)

if st.session_state.current_results:
    res = st.session_state.current_results
    st.markdown("Metrics: " + res['model_name'])
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Accuracy", f"{res['accuracy']:.3f}")
    m2.metric("AUC", f"{res['auc']:.3f}")
    m3.metric("Precision", f"{res['precision']:.3f}")
    m4.metric("Recall", f"{res['recall']:.3f}")
    m5.metric("F1", f"{res['f1']:.3f}")
    m6.metric("MCC", f"{res['mcc']:.3f}")

    if res['confusion_matrix'] is not None:
        st.pyplot(plot_styled_cm(res['confusion_matrix']))
    
    # Download predictions button
    st.markdown("---")
    st.markdown("Export Predictions")
    
    df = st.session_state.df
    results_df = df.copy()
    results_df['Prediction'] = res['y_pred']
    if 'y' in df.columns:
        results_df['Actual'] = df['y']
    
    csv = results_df.to_csv(index=False)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.download_button(
            label="Download Predictions CSV",
            data=csv,
            file_name=f"predictions_{res['model_name'].replace(' ', '_').lower()}.csv",
            mime='text/csv',
            use_container_width=True,
            help="Download the complete dataset with predictions"
        )
    
    with col2:
        if st.button("Add to Comparison", use_container_width=True):
            if not any(r['model_name'] == res['model_name'] for r in st.session_state.comparison_results):
                st.session_state.comparison_results.append(res)
                st.toast("✓ Added to comparison!")
            else:
                st.toast("Already in comparison!")

if st.session_state.comparison_results:
    st.markdown("---")
    st.markdown("Model Comparison Matrix")
    
    comp_df = pd.DataFrame(st.session_state.comparison_results).drop(['y_pred', 'confusion_matrix'], axis=1)
    st.table(comp_df)
    
    if st.button("Clear Comparison", use_container_width=False):
        st.session_state.comparison_results = []
        st.rerun()

# Footer
st.markdown("---")
st.caption("ML Assignment 2 • Bank Marketing Dataset")