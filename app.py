import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Car Price Prediction Dashboard",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Car Price Prediction Dashboard")
st.write("Predict used car prices using Machine Learning")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("carprice.csv")
    return df

try:
    df = load_data()
except:
    st.error("Place carprice.csv in the same folder as app.py")
    st.stop()

# -----------------------------
# DATA CLEANING
# -----------------------------
df["price"] = df["price"].replace("?", np.nan)
df["price"] = pd.to_numeric(df["price"], errors="coerce")

df = df.dropna()

# -----------------------------
# DATA PREVIEW
# -----------------------------
st.subheader("Dataset Preview")
st.dataframe(df.head())

st.subheader("Dataset Information")
st.write("Rows:", df.shape[0])
st.write("Columns:", df.shape[1])

# -----------------------------
# ENCODING
# -----------------------------
df_model = df.copy()

encoders = {}

for col in df_model.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])
    encoders[col] = le

# -----------------------------
# FEATURES & TARGET
# -----------------------------
X = df_model.drop("price", axis=1)
y = df_model["price"]

# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# MODEL TRAINING
# -----------------------------
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# MODEL EVALUATION
# -----------------------------
predictions = model.predict(X_test)

r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)

st.subheader("Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.metric("R² Score", f"{r2:.2f}")

with col2:
    st.metric("MAE", f"{mae:.2f}")

# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------
st.subheader("Feature Importance")

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(
    importance["Feature"],
    importance["Importance"]
)
ax.set_xlabel("Importance")
ax.set_ylabel("Features")
ax.set_title("Feature Importance")
ax.invert_yaxis()

st.pyplot(fig)

# -----------------------------
# USER INPUT SECTION
# -----------------------------
st.subheader("Predict Car Price")

col1, col2 = st.columns(2)

with col1:

    fuel_type = st.selectbox(
        "Fuel Type",
        encoders["fuel-type"].classes_
    )

    engine_location = st.selectbox(
        "Engine Location",
        encoders["engine-location"].classes_
    )

    horsepower = st.number_input(
        "Horsepower",
        min_value=40,
        max_value=500,
        value=100
    )

    peak_rpm = st.number_input(
        "Peak RPM",
        min_value=3000,
        max_value=7000,
        value=5000
    )

with col2:

    engine_type = st.selectbox(
        "Engine Type",
        encoders["engine-type"].classes_
    )

    city_mpg = st.number_input(
        "City MPG",
        min_value=5,
        max_value=60,
        value=25
    )

    highway_mpg = st.number_input(
        "Highway MPG",
        min_value=5,
        max_value=70,
        value=30
    )

# -----------------------------
# ENCODE INPUT
# -----------------------------
fuel_encoded = encoders["fuel-type"].transform([fuel_type])[0]
location_encoded = encoders["engine-location"].transform([engine_location])[0]
engine_encoded = encoders["engine-type"].transform([engine_type])[0]

input_data = pd.DataFrame({
    "fuel-type": [fuel_encoded],
    "engine-location": [location_encoded],
    "engine-type": [engine_encoded],
    "horsepower": [horsepower],
    "peak-rpm": [peak_rpm],
    "city-mpg": [city_mpg],
    "highway-mpg": [highway_mpg]
})

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("Predict Price"):

    predicted_price = model.predict(input_data)[0]

    st.success(
        f"Estimated Car Price: ${predicted_price:,.2f}"
    )

    st.subheader("Prediction Visualization")

    chart_df = pd.DataFrame({
        "Category": ["Predicted Price"],
        "Value": [predicted_price]
    })

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.bar(
        chart_df["Category"],
        chart_df["Value"]
    )
    ax2.set_ylabel("Price ($)")
    ax2.set_title("Predicted Car Price")

    st.pyplot(fig2)

# -----------------------------
# DATA VISUALIZATION
# -----------------------------
st.subheader("Price Distribution")

fig3, ax3 = plt.subplots(figsize=(8, 4))
ax3.hist(df_model["price"], bins=20)
ax3.set_title("Car Price Distribution")
ax3.set_xlabel("Price")
ax3.set_ylabel("Count")

st.pyplot(fig3)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.write("Built using Streamlit + Random Forest Regression")