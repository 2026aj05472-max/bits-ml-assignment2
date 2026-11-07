import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

st.set_page_config(page_title="ML Classifier Evaluation Suite", page_icon="🔬", layout="wide")

st.markdown("""
    <div style="background-color:#1E3A8A;padding:20px;border-radius:10px;margin-bottom:25px;">
        <h1 style="color:white;text-align:center;margin:0;">Interactive ML Classifier Evaluation Dashboard</h1>
        <p style="color:#E0E7FF;text-align:center;margin:5px 0 0 0;">
            BITS Pilani M.Tech AIML - Machine Learning Assignment 2
        </p>
    </div>
""", unsafe_allow_html=True)

st.sidebar.header("Navigation & Configuration")

model_mapping = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/svm.joblib"
}

selected_model_name = st.sidebar.selectbox("Choose Classification Model:", list(model_mapping.keys()))
uploaded_file = st.sidebar.file_uploader("Upload Test CSV Dataset", type=["csv"])

st.sidebar.markdown("---")
st.sidebar.info("**Dataset Requirements:**\n- Features: 30 numerical measurements\n- Target: 'target' column (0=Malignant, 1=Benign)")

@st.cache_resource
def load_model(path):
    return joblib.load(path)

# --- Automatic Data Source Selection ---
default_csv_path = "test_data.csv"
df_test = None

if uploaded_file is not None:
    try:
        df_test = pd.read_csv(uploaded_file)
        st.sidebar.success("Using uploaded CSV file.")
    except Exception as e:
        st.error(f"Error reading uploaded file: {e}")
elif os.path.exists(default_csv_path):
    try:
        df_test = pd.read_csv(default_csv_path)
        st.sidebar.info("Using default repository dataset (`test_data.csv`).")
    except Exception as e:
        st.error(f"Error reading repository dataset: {e}")
else:
    st.info("👈 Upload test_data.csv from the sidebar to evaluate the model.")

# --- Evaluation Dashboard Execution ---
if df_test is not None:
    try:
        st.subheader("1. Test Dataset Preview")
        c1, c2 = st.columns([3, 1])
        c1.dataframe(df_test.head(5), use_container_width=True)
        c2.metric("Total Test Samples", df_test.shape[0])
        c2.metric("Total Columns", df_test.shape[1])
        
        if 'target' not in df_test.columns:
            st.error("Error: The CSV dataset must contain a 'target' column.")
        else:
            X_eval = df_test.drop(columns=['target'])
            y_eval = df_test['target']
            
            model_path = model_mapping[selected_model_name]
            if not os.path.exists(model_path):
                st.error(f"Model file not found at {model_path}")
            else:
                model = load_model(model_path)
                y_pred = model.predict(X_eval)
                try:
                    y_proba = model.predict_proba(X_eval)[:, 1]
                    auc = roc_auc_score(y_eval, y_proba)
                except Exception:
                    auc = float('nan')
                
                acc = accuracy_score(y_eval, y_pred)
                prec = precision_score(y_eval, y_pred, zero_division=0)
                rec = recall_score(y_eval, y_pred, zero_division=0)
                f1 = f1_score(y_eval, y_pred, zero_division=0)
                mcc = matthews_corrcoef(y_eval, y_pred)
                
                st.markdown("---")
                st.subheader(f"2. Evaluation Metrics: {selected_model_name}")
                m1, m2, m3, m4, m5, m6 = st.columns(6)
                m1.metric("Accuracy", f"{acc:.4f}")
                m2.metric("AUC Score", f"{auc:.4f}")
                m3.metric("Precision", f"{prec:.4f}")
                m4.metric("Recall", f"{rec:.4f}")
                m5.metric("F1 Score", f"{f1:.4f}")
                m6.metric("MCC Score", f"{mcc:.4f}")
                
                st.markdown("---")
                st.subheader("3. Diagnostic Visualizations & Detailed Report")
                v1, v2 = st.columns([1, 1])
                with v1:
                    st.write("**Confusion Matrix**")
                    cm = confusion_matrix(y_eval, y_pred)
                    fig, ax = plt.subplots(figsize=(5, 4))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                                xticklabels=['Malignant (0)', 'Benign (1)'],
                                yticklabels=['Malignant (0)', 'Benign (1)'], ax=ax)
                    ax.set_xlabel('Predicted Label', fontweight='bold')
                    ax.set_ylabel('True Label', fontweight='bold')
                    st.pyplot(fig)
                    
                with v2:
                    st.write("**Classification Report**")
                    cr_dict = classification_report(y_eval, y_pred, target_names=['Malignant (0)', 'Benign (1)'], output_dict=True)
                    st.dataframe(pd.DataFrame(cr_dict).transpose().style.format(precision=4), use_container_width=True)
    except Exception as e:
        st.error(f"Error evaluating model on test data: {e}")
