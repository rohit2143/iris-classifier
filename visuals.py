import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st

def pairplot(df):
    dims = [c for c in df.columns if c != "species"]
    fig = px.scatter_matrix(
        df,
        dimensions=dims,
        color="species",
        symbol="species",
        title="Pairplot of Iris Dataset",
        height=700,
    )
    st.plotly_chart(fig, use_container_width=True)

def confusion_matrix_plot(cm: np.ndarray, labels, normalize=True):
    cm = np.array(cm)
    if normalize:
        cm_display = (cm.astype("float") / cm.sum(axis=1, keepdims=True) * 100).round(1)
        ann = [[f"{int(cm[i,j])}<br>({cm_display[i,j]}%)" for j in range(cm.shape[1])] for i in range(cm.shape[0])]
        z = cm_display
        colorbar_title = "%"
    else:
        ann = [[str(cm[i,j]) for j in range(cm.shape[1])] for i in range(cm.shape[0])]
        z = cm
        colorbar_title = "count"

    fig = ff.create_annotated_heatmap(
        z=z, x=list(labels), y=list(labels), annotation_text=ann,
        colorscale="Blues", showscale=True
    )
    fig.update_layout(title="Confusion Matrix", height=500)
    st.plotly_chart(fig, use_container_width=True)
