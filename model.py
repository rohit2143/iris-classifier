import json
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODELS_DIR = "models"
AVAILABLE_MODELS = ["KNN", "LogisticRegression", "RandomForest", "SVM"]

@st.cache_resource
def _load_joblib(path):
    return joblib.load(path)

@st.cache_resource
def _load_json(path):
    with open(path, "r") as f:
        return json.load(f)

@st.cache_resource
def load_dataset():
    return _load_joblib(os.path.join(MODELS_DIR, "iris_dataset.pkl"))

@st.cache_data
def load_dataframe():
    return pd.read_csv(os.path.join(MODELS_DIR, "iris_df.csv"))

@st.cache_resource
def load_model(model_name: str):
    return _load_joblib(os.path.join(MODELS_DIR, f"{model_name}.pkl"))

@st.cache_resource
def load_scaler():
    return _load_joblib(os.path.join(MODELS_DIR, "scaler.pkl"))

@st.cache_resource
def load_metrics(model_name: str):
    return _load_json(os.path.join(MODELS_DIR, f"{model_name}_metrics.json"))

def predict_species(features, model_name="KNN", return_proba=False):
    model = load_model(model_name)
    scaler = load_scaler()
    iris = load_dataset()

    x_scaled = scaler.transform(np.array(features).reshape(1, -1))
    pred_idx = model.predict(x_scaled)[0]
    species = iris.target_names[pred_idx]

    if return_proba and hasattr(model, "predict_proba"):
        return species, model.predict_proba(x_scaled)[0]
    return species
