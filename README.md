# KNN Classification and Regression using Streamlit

An interactive Machine Learning web application built using Streamlit for demonstrating:

- KNN Classification using Iris Dataset
- KNN Regression using California Housing Dataset

This project contains:
- Jupyter Notebooks
- Streamlit Web Application
- Data Preprocessing
- Model Training
- Prediction
- Evaluation Metrics
- Professional UI Design

---

# Live Demo

🚀 Streamlit Live App:

https://knnclassificationregression.streamlit.app/

---

# GitHub Repository

🔗 GitHub Repository:

https://github.com/keer-thana25/Knn_classification_regression

---

# Project Structure

```text
Knn_classification_regression/
│
├── README.md
├── app.py
├── requirements.txt
│
├── classification/
│   └── knn.ipynb
│
└── regression/
    └── knn_regression.ipynb
```

# Files Included

## 1. README.md
Project documentation and setup instructions.

## 2. app.py
Main Streamlit web application for:
- KNN Classification
- KNN Regression

## 3. knn.ipynb
Jupyter Notebook for:
- KNN Classification
- Iris Dataset
- Model Training
- Prediction
- Evaluation

## 4. knn_regression.ipynb
Jupyter Notebook for:
- KNN Regression
- California Housing Dataset
- Model Training
- Prediction
- Evaluation

## 5. requirements.txt
Contains all required libraries for the project.

---

# KNN Classification

## What is KNN Classification?

K-Nearest Neighbors (KNN) Classification is a supervised machine learning algorithm used to classify data into different categories.

The algorithm:
1. Finds the nearest neighbors
2. Checks the majority class
3. Predicts the output class

---

## Why do we use KNN Classification?

- Simple and easy to understand
- No complex training process
- Works well for small datasets
- Used in:
  - Image Recognition
  - Recommendation Systems
  - Medical Diagnosis
  - Pattern Recognition

---

## Dataset Used

### Iris Dataset

The Iris dataset contains:
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

Target Classes:
- Setosa
- Versicolor
- Virginica

---

## Steps Performed

- Import libraries
- Load dataset
- Display first 5 rows
- Display column names
- Data preprocessing
- Train test split
- Feature scaling
- Create KNN Classifier
- Train model
- Predict output
- Evaluate model
- Display confusion matrix
- Display classification report
- Display accuracy score

---

## Algorithm Used

```python
KNeighborsClassifier(
    n_neighbors=5,
    metric='minkowski',
    p=2
)
```

---

# KNN Regression

## What is KNN Regression?

KNN Regression is a supervised machine learning algorithm used to predict continuous numerical values.

The algorithm:
1. Finds nearest neighbors
2. Calculates average nearby values
3. Predicts numerical output

---

## Why do we use KNN Regression?

- Predict numerical values
- Easy to implement
- Works well with smaller datasets
- Used in:
  - House Price Prediction
  - Sales Prediction
  - Demand Forecasting
  - Financial Analysis

---

## Dataset Used

### California Housing Dataset

The dataset contains:
- Median Income
- House Age
- Average Rooms
- Population
- Latitude
- Longitude

Target:
- House Price

---

## Steps Performed

- Import libraries
- Load dataset
- Display first 5 rows
- Display column names
- Data preprocessing
- Train test split
- Feature scaling
- Create KNN Regressor
- Train model
- Predict output
- Calculate MSE
- Calculate MAE
- Calculate R2 Score
- Display Actual vs Predicted values

---

## Algorithm Used

```python
KNeighborsRegressor(
    n_neighbors=5
)
```

---

# Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

---

# Installation

## Clone Repository

```bash
git clone https://github.com/keer-thana25/Knn_classification_regression.git
```

---

## Move into Project Folder

```bash
cd Knn_classification_regression
```

---

## Install Required Libraries

```bash
pip install -r requirements.txt
```

---

## Run Streamlit Application

```bash
streamlit run app.py
```

---

# requirements.txt

```txt
streamlit
pandas
numpy
matplotlib
scikit-learn
```

---

# Output

The application provides:

✅ Interactive Machine Learning Interface  
✅ KNN Classification  
✅ KNN Regression  
✅ Data Preprocessing  
✅ Evaluation Metrics  
✅ Professional UI Design  
✅ Interactive Sidebar Controls  

---

# Author

Keerthana

GitHub:
https://github.com/keer-thana25
