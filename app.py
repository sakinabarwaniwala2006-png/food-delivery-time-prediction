import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("food_delivery_model.pkl")
feature_columns = joblib.load("model_features.pkl")

st.set_page_config(
    page_title="Food Delivery Predictor",
    page_icon="🍔",
    layout="centered"
)

st.title("🍔 Food Delivery Time Predictor")
st.write("Predict the estimated delivery time of your food order.")

st.info("Enter the order details below.")

distance = st.slider("Delivery Distance (km)", 1.0, 15.0, 6.0)
items = st.number_input("Number of Items", 1, 20, 3)

weather = st.selectbox(
    "Weather Condition", ["Clear", "Cloudy", "Rainy"]
)

traffic = st.selectbox(
    "Traffic Level", ["Low", "Medium", "High"]
)

order_time = st.selectbox(
    "Order Time", ["Morning", "Afternoon", "Evening", "Night"]
)

prep_time = st.slider(
    "Restaurant Preparation Time (minutes)", 10, 40, 20
)

experience = st.slider(
    "Delivery Partner Experience (years)", 1, 10, 5
)

if st.button("Predict Delivery Time", type="primary"):

    new_order = pd.DataFrame([{
        "Distance_km": distance,
        "Items": items,
        "Weather": weather,
        "Traffic": traffic,
        "Order_Time": order_time,
        "Restaurant_Preparation_Time": prep_time,
        "Delivery_Partner_Experience": experience
    }])

    # Match the training-time encoding
    new_order = pd.get_dummies(
        new_order, drop_first=True, dtype=int
    )
    new_order = new_order.reindex(
        columns=feature_columns, fill_value=0
    )

    prediction = model.predict(new_order)[0]

    st.success(
        f"Estimated Delivery Time: {prediction:.0f} minutes"
    )

    st.caption(
        "This estimate comes from a machine learning model "
        "trained on a synthetic dataset for demonstration."
    )

st.divider()
st.caption("Minor Project | Food Delivery Time Prediction")
