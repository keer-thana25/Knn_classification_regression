# app.py


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="KNN Explorer",
    page_icon="📘",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0b1120;
        color: white;
    }

    .main-title {
        font-size: 42px;
        font-weight: bold;
        color: #60a5fa;
        text-align: center;
        margin-bottom: 5px;
    }

    .sub-title {
        text-align: center;
        color: #94a3b8;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .card {
        background-color: #111827;
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #1e293b;
        margin-bottom: 20px;
    }

    .section-title {
        font-size: 24px;
        font-weight: bold;
        color: #38bdf8;
        margin-bottom: 15px;
    }

    .metric-box {
        background-color: #1e293b;
        padding: 15px;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #334155;
    }

    .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: #60a5fa;
    }

    .metric-label {
        color: #cbd5e1;
        font-size: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.title("KNN Explorer")

menu = st.sidebar.radio(
    "Select Algorithm",
    ["KNN Classification", "KNN Regression"]
)

k_value = st.sidebar.slider("Select K Value", 1, 20, 5)

test_size = st.sidebar.slider("Test Size (%)", 10, 40, 20)

scale_data = st.sidebar.checkbox("Apply Standard Scaling", value=True)

# --------------------------------------------------
# TITLE
# --------------------------------------------------
st.markdown('<div class="main-title">KNN Machine Learning Explorer</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="sub-title">Interactive KNN Classification and Regression using Streamlit</div>',
    unsafe_allow_html=True
)

# ==================================================
# KNN CLASSIFICATION
# ==================================================
if menu == "KNN Classification":

    # ----------------------------------------------
    # INTRODUCTION
    # ----------------------------------------------
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown('<div class="section-title">What is KNN Classification?</div>', unsafe_allow_html=True)

    st.write(
        """
        K-Nearest Neighbors (KNN) Classification is a supervised machine learning algorithm.

        It classifies data points based on the nearest neighboring points.

        The algorithm finds the K nearest neighbors and assigns the majority class.
        """
    )

    st.markdown('<div class="section-title">Why do we use KNN Classification?</div>', unsafe_allow_html=True)

    st.write(
        """
        • Simple and easy to understand
        • Works well for small datasets
        • Used in image recognition and recommendation systems
        • No training phase complexity
        """
    )

    st.markdown('</div>', unsafe_allow_html=True)

    # ----------------------------------------------
    # LOAD DATASET
    # ----------------------------------------------
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown('<div class="section-title">1. Load Iris Dataset</div>', unsafe_allow_html=True)

    iris = load_iris()

    df = pd.DataFrame(iris.data, columns=iris.feature_names)

    df['target'] = iris.target

    st.write("### First 5 Rows")
    st.dataframe(df.head(), use_container_width=True)

    st.write("### Column Names")
    st.write(df.columns.tolist())

    st.markdown('</div>', unsafe_allow_html=True)

    # ----------------------------------------------
    # PREPROCESSING
    # ----------------------------------------------
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown('<div class="section-title">2. Data Preprocessing</div>', unsafe_allow_html=True)

    X = df.drop('target', axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size / 100,
        random_state=42
    )

    if scale_data:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        X_train = pd.DataFrame(X_train, columns=X.columns)
        X_test = pd.DataFrame(X_test, columns=X.columns)

    col1, col2 = st.columns(2)

    with col1:
        st.write("### X Train")
        st.dataframe(X_train.head())

        st.write("### Y Train")
        st.dataframe(pd.DataFrame(y_train).head())

    with col2:
        st.write("### X Test")
        st.dataframe(X_test.head())

        st.write("### Y Test")
        st.dataframe(pd.DataFrame(y_test).head())

    st.markdown('</div>', unsafe_allow_html=True)

    # ----------------------------------------------
    # MODEL CREATION
    # ----------------------------------------------
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown('<div class="section-title">3. Create KNN Classifier</div>', unsafe_allow_html=True)

    st.code(
        f"""
model = KNeighborsClassifier(
    n_neighbors={k_value},
    metric='minkowski',
    p=2
)
        """,
        language='python'
    )

    model = KNeighborsClassifier(
        n_neighbors=k_value,
        metric='minkowski',
        p=2
    )

    st.markdown('</div>', unsafe_allow_html=True)

    # ----------------------------------------------
    # TRAIN MODEL
    # ----------------------------------------------
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown('<div class="section-title">4. Train the Model</div>', unsafe_allow_html=True)

    model.fit(X_train, y_train)

    st.success("Model trained successfully")

    st.markdown('</div>', unsafe_allow_html=True)

    # ----------------------------------------------
    # PREDICTION
    # ----------------------------------------------
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown('<div class="section-title">5. Prediction</div>', unsafe_allow_html=True)

    y_pred = model.predict(X_test)

    result = pd.DataFrame({
        'Actual': y_test.values,
        'Predicted': y_pred
    })

    st.dataframe(result.head(15), use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ----------------------------------------------
    # EVALUATION
    # ----------------------------------------------
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown('<div class="section-title">6. Model Evaluation</div>', unsafe_allow_html=True)

    accuracy = accuracy_score(y_test, y_pred)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f'''<div class="metric-box"><div class="metric-value">{accuracy:.2f}</div><div class="metric-label">Accuracy</div></div>''', unsafe_allow_html=True)

    with col2:
        st.markdown(f'''<div class="metric-box"><div class="metric-value">{k_value}</div><div class="metric-label">Neighbors</div></div>''', unsafe_allow_html=True)

    with col3:
        st.markdown(f'''<div class="metric-box"><div class="metric-value">3</div><div class="metric-label">Classes</div></div>''', unsafe_allow_html=True)

    st.write("### Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    ax.imshow(cm)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    for i in range(len(cm)):
        for j in range(len(cm[0])):
            ax.text(j, i, cm[i, j], ha='center', va='center')

    st.pyplot(fig)

    st.write("### Classification Report")

    report = classification_report(y_test, y_pred)

    st.text(report)

    st.markdown('</div>', unsafe_allow_html=True)

# ==================================================
# KNN REGRESSION
# ==================================================
else:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown('<div class="section-title">What is KNN Regression?</div>', unsafe_allow_html=True)

    st.write(
        """
        KNN Regression predicts continuous numerical values.

        It works by finding the nearest neighbors and calculating the average of those nearby values.
        """
    )

    st.markdown('<div class="section-title">Why do we use KNN Regression?</div>', unsafe_allow_html=True)

    st.write(
        """
        • Predict house prices
        • Predict sales and demand
        • Easy to implement
        • Works well for small datasets
        """
    )

    st.markdown('</div>', unsafe_allow_html=True)

    # ----------------------------------------------
    # LOAD DATASET
    # ----------------------------------------------
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown('<div class="section-title">1. Load California Housing Dataset</div>', unsafe_allow_html=True)

    housing = fetch_california_housing()

    df = pd.DataFrame(housing.data, columns=housing.feature_names)

    df['PRICE'] = housing.target

    st.write("### First 5 Rows")
    st.dataframe(df.head(), use_container_width=True)

    st.write("### Column Names")
    st.write(df.columns.tolist())

    st.markdown('</div>', unsafe_allow_html=True)

    # ----------------------------------------------
    # PREPROCESSING
    # ----------------------------------------------
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown('<div class="section-title">2. Data Preprocessing</div>', unsafe_allow_html=True)

    X = df.drop('PRICE', axis=1)
    y = df['PRICE']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size / 100,
        random_state=42
    )

    if scale_data:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        X_train = pd.DataFrame(X_train, columns=X.columns)
        X_test = pd.DataFrame(X_test, columns=X.columns)

    col1, col2 = st.columns(2)

    with col1:
        st.write("### X Train")
        st.dataframe(X_train.head())

        st.write("### Y Train")
        st.dataframe(pd.DataFrame(y_train).head())

    with col2:
        st.write("### X Test")
        st.dataframe(X_test.head())

        st.write("### Y Test")
        st.dataframe(pd.DataFrame(y_test).head())

    st.markdown('</div>', unsafe_allow_html=True)

    # ----------------------------------------------
    # MODEL CREATION
    # ----------------------------------------------
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown('<div class="section-title">3. Create KNN Regressor</div>', unsafe_allow_html=True)

    st.code(
        f"model = KNeighborsRegressor(n_neighbors={k_value})",
        language='python'
    )

    model = KNeighborsRegressor(n_neighbors=k_value)

    st.markdown('</div>', unsafe_allow_html=True)

    # ----------------------------------------------
    # TRAIN MODEL
    # ----------------------------------------------
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown('<div class="section-title">4. Train the Model</div>', unsafe_allow_html=True)

    model.fit(X_train, y_train)

    st.success("Model trained successfully")

    st.markdown('</div>', unsafe_allow_html=True)

    # ----------------------------------------------
    # PREDICTION
    # ----------------------------------------------
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown('<div class="section-title">5. Prediction</div>', unsafe_allow_html=True)

    y_pred = model.predict(X_test)

    result = pd.DataFrame({
        'Actual Price': y_test.values[:20],
        'Predicted Price': y_pred[:20]
    })

    st.dataframe(result, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ----------------------------------------------
    # EVALUATION
    # ----------------------------------------------
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown('<div class="section-title">6. Model Evaluation</div>', unsafe_allow_html=True)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f'''<div class="metric-box"><div class="metric-value">{mse:.2f}</div><div class="metric-label">MSE</div></div>''', unsafe_allow_html=True)

    with col2:
        st.markdown(f'''<div class="metric-box"><div class="metric-value">{mae:.2f}</div><div class="metric-label">MAE</div></div>''', unsafe_allow_html=True)

    with col3:
        st.markdown(f'''<div class="metric-box"><div class="metric-value">{r2:.2f}</div><div class="metric-label">R² Score</div></div>''', unsafe_allow_html=True)

    fig, ax = plt.subplots()
    ax.scatter(y_test[:100], y_pred[:100])
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title("Actual vs Predicted")

    st.pyplot(fig)

    st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")

st.markdown(
    "<center style='color:gray;'>KNN Machine Learning Explorer using Streamlit</center>",
    unsafe_allow_html=True
)
