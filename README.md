# Machine Learning Studio & Dashboard Project

A complete, end-to-end Machine Learning Dashboard and educational laboratory for dataset synthesis, CSV ingestion, model training (Logistic Regression, KNN, Decision Tree), and rich visual diagnostics.

---

## 🌟 Key Features

1. **Synthetic Dataset Generator**:
   - Generates 6 benchmark distributions: Two Moons, Concentric Circles, Linearly Separable Blobs, Spiral XOR, Customer Churn, and Linear Trend.
   - User-controlled sample sizes and Gaussian noise levels ($\sigma$).
   - CSV export for created datasets.

2. **Custom Dataset Ingestion (Upload CSV)**:
   - Drag & drop or browse CSV files.
   - Built-in delimiter auto-detection and quote parsing.
   - Interactive column selector for target labels and feature vectors.
   - 4 built-in standard demo datasets: Iris, Titanic, Breast Cancer Biopsy, Study Hours vs Pass.

3. **Machine Learning Model Training**:
   - **Logistic Regression**: Implemented with Sigmoid activation $\sigma(z) = \frac{1}{1 + e^{-z}}$, Batch Gradient Descent, Binary Cross-Entropy Loss, and L2 Regularization penalty.
   - **K-Nearest Neighbors (KNN)**: Non-parametric instance-based classification with Euclidean metric.
   - **Decision Tree**: CART-based recursive binary tree with Gini impurity minimization.
   - **Preprocessing**: Automatic Z-score standardization $((x - \mu)/\sigma)$ and 80/20 train/test split.

4. **Visual Analytics & Diagnostics**:
   - **2D Decision Boundary Contour**: High-resolution probability surface with color-coded confidence levels and decision boundary threshold ($P=0.5$).
   - **Training Loss Convergence Curve**: Epoch-by-epoch loss reduction.
   - **Confusion Matrix**: True Positives, True Negatives, False Positives (Type I), False Negatives (Type II).
   - **ROC Curve & AUC Score**: Sensitivity vs False Positive Rate curve with trapezoidal integration.
   - **Feature Weights Bar Chart**: Learned coefficient importances.

5. **Live Inference Playground**:
   - Real-time feature sliders with instant predicted class badge and probability gauge bar.

---
