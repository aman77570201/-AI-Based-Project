import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Weather Data Analysis and Prediction",
    page_icon="🌦️",
    layout="wide"
)

st.title("🌦️ Weather Data Analysis and Prediction")
st.markdown("Analyze historical weather data and predict future temperature trends.")

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("weather.csv")

# Convert date column
df["date"] = pd.to_datetime(df["date"])

# -----------------------------
# Sidebar Filters
# -----------------------------
st.sidebar.header("📌 Filters")

city = st.sidebar.selectbox(
    "Select City",
    sorted(df["city"].unique())
)

city_df = df[df["city"] == city]

min_date = city_df["date"].min().date()
max_date = city_df["date"].max().date()

from_date = st.sidebar.date_input(
    "From Date",
    value=min_date,
    min_value=min_date,
    max_value=max_date
)

to_date = st.sidebar.date_input(
    "To Date",
    value=max_date,
    min_value=min_date,
    max_value=max_date
)

filtered_df = city_df[
    (city_df["date"] >= pd.to_datetime(from_date)) &
    (city_df["date"] <= pd.to_datetime(to_date))
]

st.success(f"✅ Dataset Loaded Successfully ({city})")

# -----------------------------
# Dataset Preview
# -----------------------------
st.subheader("📋 Dataset Preview")
st.dataframe(filtered_df.head())

# -----------------------------
# Dataset Information
# -----------------------------
st.subheader("📊 Dataset Information")

col1, col2 = st.columns(2)

with col1:
    st.metric("Total Records", len(filtered_df))

with col2:
    st.metric("Total Columns", len(filtered_df.columns))

st.write("### Column Names")
st.write(list(filtered_df.columns))

# -----------------------------
# Missing Values
# -----------------------------
st.subheader("❌ Missing Values")

missing = filtered_df.isnull().sum()

st.dataframe(missing)

# -----------------------------
# Dataset Statistics
# -----------------------------
st.subheader("📈 Dataset Statistics")

st.dataframe(filtered_df.describe(include="all"))

# -----------------------------
# Temperature Trend
# -----------------------------
st.subheader("🌡 Temperature Trend")

fig, ax = plt.subplots(figsize=(12,5))

ax.plot(
    filtered_df["date"],
    filtered_df["temperature_2m_max"],
    color="red",
    label="Maximum Temperature"
)

ax.set_xlabel("Date")
ax.set_ylabel("Temperature (°C)")
ax.set_title(f"{city} Temperature Trend")

plt.xticks(rotation=45)

ax.legend()

st.pyplot(fig)
# -----------------------------
# Machine Learning Model
# -----------------------------

model_df = filtered_df.copy()

# Day Number (Date → Number)
model_df["Day"] = (
    model_df["date"] - model_df["date"].min()
).dt.days

X = model_df[["Day"]]
y = model_df["temperature_2m_max"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

# Prediction on Test Data
y_pred = model.predict(X_test)

# Model Performance
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

st.subheader("🤖 Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.metric("R² Score", f"{r2:.3f}")

with col2:
    st.metric("MAE", f"{mae:.3f}")

# -----------------------------
# Future Temperature Prediction
# -----------------------------

st.subheader("🔮 Future Temperature Prediction")

future_date = st.date_input(
    "Select Future Date",
    value=max_date
)

if st.button("Predict Future Temperature"):

    future_day = (
        pd.to_datetime(future_date)
        - model_df["date"].min()
    ).days

    predicted_temp = model.predict([[future_day]])[0]

    st.success(
        f"""
📍 City : {city}

📅 Prediction Date : {future_date}

🌡️ Predicted Maximum Temperature : {predicted_temp:.2f} °C
"""
    )
    # -----------------------------
# Actual vs Predicted Graph
# -----------------------------

st.subheader("📈 Actual vs Predicted Temperature")

compare = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

fig, ax = plt.subplots(figsize=(10,5))

ax.plot(compare["Actual"].reset_index(drop=True),
        label="Actual Temperature")

ax.plot(compare["Predicted"].reset_index(drop=True),
        label="Predicted Temperature")

ax.set_xlabel("Samples")
ax.set_ylabel("Temperature (°C)")
ax.set_title("Actual vs Predicted Temperature")

ax.legend()

st.pyplot(fig)

# -----------------------------
# Future Trend Graph
# -----------------------------

st.subheader("📊 Historical Temperature Trend")

fig2, ax2 = plt.subplots(figsize=(12,5))

ax2.plot(
    model_df["date"],
    model_df["temperature_2m_max"],
    linewidth=2,
    label="Historical Temperature"
)

if st.button("Show Future Trend"):

    future_dates = pd.date_range(
        model_df["date"].max(),
        periods=30,
        freq="D"
    )

    future_days = (
        future_dates - model_df["date"].min()
    ).days

    future_prediction = model.predict(
        pd.DataFrame({"Day": future_days})
    )

    ax2.plot(
        future_dates,
        future_prediction,
        linestyle="--",
        linewidth=2,
        label="Predicted Trend"
    )

ax2.set_xlabel("Date")
ax2.set_ylabel("Temperature (°C)")
ax2.set_title(f"{city} Historical & Future Temperature Trend")
ax2.legend()

st.pyplot(fig2)

# -----------------------------
# Project Conclusion
# -----------------------------

st.subheader("📌 Project Conclusion")

st.success("""
✅ Historical weather data analyzed successfully.

✅ Linear Regression model trained successfully.

✅ Future maximum temperature predicted using historical trend.

✅ Dynamic city selection available.

✅ Dynamic date filtering available.

✅ Historical graphs generated successfully.

✅ Future prediction completed.
""")