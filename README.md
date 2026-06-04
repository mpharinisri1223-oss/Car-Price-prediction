# 🚗 Car Price Prediction Dashboard

## 📌 Project Overview

The Car Price Prediction Dashboard is a machine learning web application developed using Streamlit. The application predicts the price of a car based on features such as fuel type, engine location, engine type, horsepower, peak RPM, city MPG, and highway MPG.

The project demonstrates the complete machine learning workflow including data cleaning, regression modeling, feature importance visualization, user interaction, and prediction generation.

---

## 🎯 Objectives

* Load and clean car dataset
* Perform regression modeling
* Visualize feature importance
* Create an interactive user interface
* Predict car prices based on user inputs
* Display prediction results with charts

---

## 🛠️ Technologies Used

* Python
* Streamlit
* Pandas
* NumPy
* Scikit-Learn
* Plotly
* Matplotlib

---

## 📂 Dataset Features

The dataset contains the following attributes:

* fuel-type
* engine-location
* engine-type
* horsepower
* peak-rpm
* city-mpg
* highway-mpg
* price

---

## ⚙️ Machine Learning Model

The project uses:

**Random Forest Regressor**

Steps performed:

1. Data Loading
2. Data Cleaning
3. Handling Missing Values
4. Encoding Categorical Features
5. Train-Test Split
6. Model Training
7. Performance Evaluation
8. Prediction Generation

---

## 📊 Dashboard Features

### Dataset Preview

Displays the loaded dataset.

### Model Performance

Shows the R² Score of the trained model.

### Feature Importance

Visualizes the most influential features affecting car price.

### User Input Interface

Allows users to enter car specifications.

### Price Prediction

Predicts the estimated car price based on user inputs.

### Interactive Charts

Displays prediction results using Plotly visualizations.

---

## 🚀 Installation

### Clone Repository

```bash
git clone https://github.com/mpharinisri1223-oss/Car-Price-prediction.git
```

### Move into Project Folder

```bash
cd Car-Price-prediction
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```text
Car-Price-prediction/
│
├── app.py
├── carprice.csv
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📈 Output

The dashboard provides:

* Dataset Preview
* Feature Importance Graph
* Model Accuracy Score
* Car Price Prediction
* Interactive Charts

---

## 👩‍💻 Developer

**Priyadharshini J**

Applied AI Project

---

## 📜 License

This project is developed for educational and academic purposes.
