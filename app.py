import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, matthews_corrcoef, confusion_matrix
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
import seaborn as sns

# Page title
st.title("ML Assignment 2 - Classification Models Dashboard")

st.write("Upload dataset and select model to evaluate")

# Model selection dropdown
model_name = st.selectbox(
    "Select Model",
    (
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    )
)

# File uploader
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

# Load model function
def load_model(name):

    model_path = f"models/{name}.pkl"

    return joblib.load(model_path)


# When file uploaded
if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.write("Dataset Preview:")
    st.write(df.head())

    # Assume last column is target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Load selected model
    model = load_model(model_name)

    # Predictions
    y_pred = model.predict(X)

    # For AUC
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, y_prob)
    else:
        auc = 0

    # Metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='weighted')
    recall = recall_score(y, y_pred, average='weighted')
    f1 = f1_score(y, y_pred, average='weighted')
    mcc = matthews_corrcoef(y, y_pred)

    st.subheader("Evaluation Metrics")

    st.write(f"Accuracy: {accuracy:.4f}")
    st.write(f"AUC Score: {auc:.4f}")
    st.write(f"Precision: {precision:.4f}")
    st.write(f"Recall: {recall:.4f}")
    st.write(f"F1 Score: {f1:.4f}")
    st.write(f"MCC Score: {mcc:.4f}")

    # Confusion Matrix
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots()

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)

    st.pyplot(fig)

    # Classification Report
    st.subheader("Classification Report")

    report = classification_report(y, y_pred)

    st.text(report)

else:

    st.write("Please upload a dataset to continue")

