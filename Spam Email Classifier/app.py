import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Load Dataset
df = pd.read_csv("spam.csv", sep="\t", names=["label", "message"])

# Features and Labels
X = df["message"]
y = df["label"]

# Convert text into numbers
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = MultinomialNB()
model.fit(X_train, y_train)

# Streamlit UI
st.sidebar.title("Project Information")
st.sidebar.write("📧 Smart Spam Email Classifier")
st.sidebar.write("👨‍💻 Developed by: Aman Kumar Singh")
st.sidebar.write("🤖 Model: Multinomial Naive Bayes")
st.sidebar.write("📊 Dataset: SMS Spam Collection")
st.title("📧 Smart Spam Email Classifier")
st.write(
    "This application uses a machine learning model to classify emails as Spam or Safe."
)

user_input = st.text_area(
    "Enter your Email or SMS",
    placeholder="Type your message here..."
)

if st.button("🔍 Analyze Message"):
    data = vectorizer.transform([user_input])
    prediction = model.predict(data)[0]
    confidence = max(model.predict_proba(vectorizer.transform([user_input]))[0]) * 100

    if prediction == "spam":
        st.error("🚨 SPAM Message ")
    else:
        st.success("✅ Safe Message ")
        st.info(f"Confidence: {confidence:.2f}%")