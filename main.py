import streamlit as st
import pandas as pd
import numpy as np
from model import AVAILABLE_MODELS, load_dataset, load_dataframe, load_metrics, predict_species
from visuals import pairplot, confusion_matrix_plot
import plotly.express as px

st.set_page_config(page_title="Iris Classifier", layout="wide")
st.title("🌸 Iris Flower Classifier — Pro Dashboard")

# Sidebar
st.sidebar.header("⚙️ Controls")
model_name = st.sidebar.selectbox("Model", AVAILABLE_MODELS)
show_probs = st.sidebar.checkbox("Show prediction probabilities", True)
live_pred = st.sidebar.checkbox("Live prediction", True)
normalize_cm = st.sidebar.checkbox("Normalize confusion matrix", True)

iris = load_dataset()
df = load_dataframe()
metrics = load_metrics(model_name)

# Navigation
page = st.sidebar.radio("Go to:", ["Dataset", "Statistics", "Visualizations", "Classifier"])

if page == "Dataset":
    st.subheader("Iris Dataset")
    st.dataframe(df, use_container_width=True)

elif page == "Statistics":
    st.subheader(f"Model Performance — {model_name}")
    left, right = st.columns([1, 2])
    with left:
        st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    with right:
        st.write(pd.DataFrame(metrics["report"]).T)

    st.markdown("---")
    st.subheader("Confusion Matrix")
    confusion_matrix_plot(np.array(metrics["cm"]), labels=list(iris.target_names), normalize=normalize_cm)

elif page == "Visualizations":
    st.subheader("Interactive Pairplot")
    pairplot(df)

    st.markdown("---")
    st.subheader("Class Distribution")
    st.bar_chart(df["species"].value_counts())

elif page == "Classifier":
    st.subheader(f"Classify a New Flower — using {model_name}")

    # Sliders
    sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 5.1, 0.1)
    sepal_width  = st.slider("Sepal width (cm)",  2.0, 4.5, 3.5, 0.1)
    petal_length = st.slider("Petal length (cm)", 1.0, 7.0, 1.4, 0.1)
    petal_width  = st.slider("Petal width (cm)",  0.1, 2.5, 0.2, 0.1)
    inputs = [sepal_length, sepal_width, petal_length, petal_width]

    def render_prediction():
        if show_probs:
            species, probs = predict_species(inputs, model_name, return_proba=True)
        else:
            species = predict_species(inputs, model_name, return_proba=False)
            probs = None

        st.success(f"Predicted Species: {species}")
        if probs is not None:
            prob_df = pd.DataFrame({"Species": iris.target_names, "Probability": probs})
            st.bar_chart(prob_df.set_index("Species"))

    if live_pred:
        render_prediction()
    else:
        if st.button("Predict"):
            render_prediction()

    # Show input point on pairplot
    st.markdown("### 📊 Your Input in Context")
    df_plot = df.copy()
    df_plot["point_type"] = "dataset"
    df_plot = pd.concat(
        [df_plot, pd.DataFrame([inputs + ["your input"]],
         columns=["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)", "point_type"])],
        ignore_index=True
    )
    fig = px.scatter_matrix(df_plot,
                            dimensions=df.columns[:-1],
                            color="point_type",
                            symbol="point_type",
                            title="Your Input vs Dataset",
                            height=700)
    st.plotly_chart(fig, use_container_width=True)
