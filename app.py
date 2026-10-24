import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import make_moons, make_circles, make_blobs, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, log_loss
)

st.set_page_config(
    page_title="ML Studio Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Machine Learning Subject Project Dashboard")
st.markdown("### Interactive Dataset Synthesis, Ingestion, Model Training & Visual Diagnostics")

# Sidebar Configuration
st.sidebar.header("1. Dataset Configuration")
data_source = st.sidebar.radio("Data Source Mode", ["Generate Synthetic Dataset", "Upload Custom CSV"])

df = None
feature_names = []
target_name = "target"

if data_source == "Generate Synthetic Dataset":
    preset = st.sidebar.selectbox("Synthetic Preset", ["Two Moons", "Concentric Circles", "Linearly Separable Blobs", "Multi-Feature Classification"])
    n_samples = st.sidebar.slider("Number of Samples", 50, 1000, 300, step=50)
    noise = st.sidebar.slider("Noise Level (sigma)", 0.0, 0.5, 0.15, step=0.01)
    
    if preset == "Two Moons":
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
        feature_names = ["Curvature_X", "Curvature_Y"]
    elif preset == "Concentric Circles":
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.4, random_state=42)
        feature_names = ["Radius_X", "Radius_Y"]
    elif preset == "Linearly Separable Blobs":
        X, y = make_blobs(n_samples=n_samples, centers=2, cluster_std=1.0 + noise*3, random_state=42)
        feature_names = ["Feature_1", "Feature_2"]
    else:
        X, y = make_classification(n_samples=n_samples, n_features=4, n_informative=3, n_redundant=1, random_state=42)
        feature_names = ["Feature_1", "Feature_2", "Feature_3", "Feature_4"]
    
    df = pd.DataFrame(X, columns=feature_names)
    df[target_name] = y

else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv", "txt"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        target_col = st.sidebar.selectbox("Select Target Column (Label)", df.columns, index=len(df.columns)-1)
        avail_features = [c for c in df.columns if c != target_col]
        selected_features = st.sidebar.multiselect("Select Feature Columns", avail_features, default=avail_features[:2])
        if len(selected_features) < 1:
            st.error("Please select at least 1 feature.")
            st.stop()
        
        feature_names = selected_features
        target_name = target_col
        # Binary target mapping
        y_raw = df[target_name].values
        unique_labels = np.unique(y_raw)
        if len(unique_labels) >= 2:
            y = np.where(y_raw == unique_labels[1], 1, 0)
        else:
            y = y_raw.astype(int)
        X = df[feature_names].values
    else:
        st.info("Please upload a CSV file or switch to 'Generate Synthetic Dataset'.")
        st.stop()

# Model Hyperparameters
st.sidebar.header("2. Model & Training Settings")
model_choice = st.sidebar.selectbox("Machine Learning Model", ["Logistic Regression", "K-Nearest Neighbors", "Decision Tree"])

test_size = st.sidebar.slider("Test Split Ratio", 0.1, 0.4, 0.2, step=0.05)
standardize = st.sidebar.checkbox("Standardize Features (Z-Score)", value=True)

if model_choice == "Logistic Regression":
    C_reg = st.sidebar.slider("Inverse Regularization Strength (C)", 0.01, 10.0, 1.0, step=0.1)
    max_iter = st.sidebar.slider("Max Iterations", 50, 500, 200, step=25)
    model = LogisticRegression(C=C_reg, max_iter=max_iter, random_state=42)
elif model_choice == "K-Nearest Neighbors":
    n_neighbors = st.sidebar.slider("Number of Neighbors (K)", 1, 25, 5, step=2)
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
else:
    max_depth = st.sidebar.slider("Max Tree Depth", 1, 10, 4)
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)

# Data Splitting & Scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y if len(np.unique(y)) > 1 else None)

scaler = None
if standardize:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
else:
    X_train_scaled = X_train
    X_test_scaled = X_test

# Train Model
model.fit(X_train_scaled, y_train)

# Predictions & Metrics
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else y_pred

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
cm = confusion_matrix(y_test, y_pred)

# Main UI Display
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Test Accuracy", f"{acc*100:.1f}%")
col2.metric("Precision", f"{prec:.3f}")
col3.metric("Recall", f"{rec:.3f}")
col4.metric("F1-Score", f"{f1:.3f}")
col5.metric("ROC AUC", f"{roc_auc:.3f}")

st.divider()

# Visualizations Row 1
vcol1, vcol2 = st.columns([1.2, 1])

with vcol1:
    st.subheader("2D Feature Space & Decision Boundary")
    if len(feature_names) >= 2:
        fx = 0
        fy = 1
        x_min, x_max = X[:, fx].min() - 0.5, X[:, fx].max() + 0.5
        y_min, y_max = X[:, fy].min() - 0.5, X[:, fy].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        
        # Grid input
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        if len(feature_names) > 2:
            padding = np.tile(X[:, 2:].mean(axis=0), (mesh_points.shape[0], 1))
            mesh_points = np.hstack([mesh_points, padding])
        
        mesh_scaled = scaler.transform(mesh_points) if scaler else mesh_points
        Z = model.predict_proba(mesh_scaled)[:, 1] if hasattr(model, "predict_proba") else model.predict(mesh_scaled)
        Z = Z.reshape(xx.shape)
        
        fig_boundary = go.Figure()
        fig_boundary.add_trace(go.Contour(
            x=np.linspace(x_min, x_max, 100),
            y=np.linspace(y_min, y_max, 100),
            z=Z,
            colorscale="RdBu_r",
            showscale=True,
            colorbar=dict(title="P(Class 1)"),
            opacity=0.65
        ))
        fig_boundary.add_trace(go.Scatter(
            x=X_test[y_test == 0, fx], y=X_test[y_test == 0, fy],
            mode="markers", marker=dict(color="#0ea5e9", size=7, line=dict(color="white", width=1)),
            name="Class 0 (Test)"
        ))
        fig_boundary.add_trace(go.Scatter(
            x=X_test[y_test == 1, fx], y=X_test[y_test == 1, fy],
            mode="markers", marker=dict(color="#f43f5e", size=7, line=dict(color="white", width=1)),
            name="Class 1 (Test)"
        ))
        fig_boundary.update_layout(
            xaxis_title=feature_names[fx],
            yaxis_title=feature_names[fy],
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_boundary, use_container_width=True)
    else:
        st.write("2D Boundary requires at least 2 features.")

with vcol2:
    st.subheader("Confusion Matrix")
    fig_cm = px.imshow(
        cm,
        text_auto=True,
        labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
        x=["Class 0", "Class 1"],
        y=["Class 0", "Class 1"],
        color_continuous_scale="Blues"
    )
    st.plotly_chart(fig_cm, use_container_width=True)

# Visualizations Row 2
vcol3, vcol4 = st.columns(2)

with vcol3:
    st.subheader("ROC Curve (Receiver Operating Characteristic)")
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC = {roc_auc:.3f})", line=dict(color="#8b5cf6", width=3)))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random Guess", line=dict(dash="dash", color="gray")))
    fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_roc, use_container_width=True)

with vcol4:
    st.subheader("Feature Importance / Model Weights")
    if hasattr(model, "coef_"):
        weights = model.coef_[0]
        fig_weights = px.bar(x=weights, y=feature_names, orientation="h", labels={"x": "Coefficient Weight", "y": "Feature"}, color=weights, color_continuous_scale="RdBu_r")
        st.plotly_chart(fig_weights, use_container_width=True)
    elif hasattr(model, "feature_importances_"):
        fig_weights = px.bar(x=model.feature_importances_, y=feature_names, orientation="h", labels={"x": "Importance", "y": "Feature"})
        st.plotly_chart(fig_weights, use_container_width=True)
    else:
        st.info("K-Nearest Neighbors uses distance-based instance voting without explicit linear coefficients.")

st.divider()
st.subheader("Dataset Preview & Export")
st.dataframe(df.head(10), use_container_width=True)

csv_data = df.to_csv(index=False).encode("utf-8")
st.download_button("📥 Download Generated/Processed Dataset (CSV)", csv_data, "ml_dataset.csv", "text/csv")
