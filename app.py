
import streamlit as st
import pandas as pd
import joblib
import os

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Food Delivery AI | Smart Predictor",
    page_icon="🍔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# CUSTOM CSS: PREMIUM PURPLE & BLUE THEME
# --------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(ellipse at top left,
        rgba(105, 70, 220, 0.16), transparent 45%),
        linear-gradient(135deg, #0b1020 0%, #11182d 55%, #17112b 100%);
    color: #f1f5f9;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1250px;
}

.hero {
    padding: 30px;
    border-radius: 24px;
    background: linear-gradient(120deg,
        rgba(91, 63, 211, 0.30),
        rgba(30, 115, 210, 0.20),
        rgba(167, 81, 218, 0.20));
    border: 1px solid rgba(167, 139, 250, 0.30);
    margin-bottom: 25px;
    transition: transform 0.3s ease, border-color 0.3s ease;
}

.hero:hover {
    transform: translateY(-3px);
    border-color: rgba(167, 139, 250, 0.65);
}

.hero h1 {
    font-size: clamp(1.8rem, 4vw, 2.8rem);
    font-weight: 800;
    line-height: 1.2;
    color: #ffffff;
    margin-bottom: 12px;
}

.hero p {
    color: #cbd5e1;
    font-size: 1rem;
    line-height: 1.7;
}

.eyebrow {
    color: #c4b5fd;
    font-size: 0.78rem;
    font-weight: 800;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 12px;
}

.section-heading {
    font-size: 1.35rem;
    font-weight: 750;
    color: #f8fafc;
    margin: 12px 0 5px 0;
}

.section-subtitle {
    color: #94a3b8;
    font-size: 0.92rem;
    margin-bottom: 18px;
}

div[data-testid="stMetric"] {
    background: linear-gradient(145deg,
        rgba(31, 41, 70, 0.96),
        rgba(26, 29, 56, 0.96));
    border: 1px solid rgba(148, 163, 184, 0.18);
    border-radius: 18px;
    padding: 20px;
    transition: transform 0.25s ease, border-color 0.25s ease;
}

div[data-testid="stMetric"]:hover {
    transform: translateY(-4px);
    border-color: #8b5cf6;
}

div[data-testid="stMetricLabel"] {
    color: #cbd5e1 !important;
}

div[data-testid="stMetricValue"] {
    color: #c4b5fd !important;
}

div.stButton > button {
    background: linear-gradient(100deg, #7c3aed, #4f46e5, #2563eb);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.8rem 1.2rem;
    font-weight: 700;
    min-height: 3rem;
    transition: transform 0.2s ease, filter 0.2s ease;
}

div.stButton > button:hover {
    transform: translateY(-2px);
    filter: brightness(1.15);
    color: white;
    border: none;
}

div[data-testid="stForm"] {
    background: rgba(22, 30, 53, 0.68);
    border: 1px solid rgba(148, 163, 184, 0.20);
    border-radius: 20px;
    padding: 22px;
}

div[data-testid="stTabs"] button {
    font-weight: 650;
}

div[data-testid="stAlert"] {
    border-radius: 14px;
}

footer {
    visibility: hidden;
}

#MainMenu {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD MODEL AND FEATURES
# --------------------------------------------------
@st.cache_resource
def load_model_files():
    model = joblib.load("food_delivery_model.pkl")
    features = joblib.load("model_features.pkl")
    return model, list(features)


try:
    model, feature_columns = load_model_files()
except Exception as e:
    st.error("The model files could not be loaded.")
    st.write(
        "Check that food_delivery_model.pkl and "
        "model_features.pkl are present in your repository."
    )
    st.stop()


# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown("""
<div class="hero">
    <div class="eyebrow">SMART DELIVERY ANALYTICS · MACHINE LEARNING</div>
    <h1>🍔 Food Delivery AI</h1>
    <p>
        Predict estimated delivery time using your order details
        and a trained machine learning model.
        Explore the prediction, order summary and model insights
        in one place.
    </p>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.markdown("## 🍽️ Delivery AI")
    st.caption("Your smart delivery prediction dashboard")

    page = st.radio(
        "Navigate",
        [
            "🚀 Delivery Predictor",
            "📊 Model Insights",
            "🗂️ Dataset Explorer"
        ]
    )

    st.divider()
    st.markdown("### About this project")
    st.write(
        "A machine learning minor project for estimating "
        "food delivery time from order and delivery features."
    )
    st.caption("Model: saved trained model")
    st.caption("Dataset: demonstration dataset")


# --------------------------------------------------
# DELIVERY PREDICTOR
# --------------------------------------------------
if page == "🚀 Delivery Predictor":

    st.markdown(
        '<div class="section-heading">🚀 Configure your order</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="section-subtitle">'
        'Adjust the details below and generate a delivery estimate.'
        '</div>',
        unsafe_allow_html=True
    )

    with st.form("delivery_prediction_form"):

        left, right = st.columns(2, gap="large")

        with left:
            st.markdown("### 📍 Delivery details")

            distance = st.slider(
                "Delivery distance (km)",
                min_value=1.0,
                max_value=15.0,
                value=6.0,
                step=0.5
            )

            items = st.number_input(
                "Number of food items",
                min_value=1,
                max_value=20,
                value=3,
                step=1
            )

            traffic = st.selectbox(
                "Traffic level",
                ["Low", "Medium", "High"]
            )

            weather = st.selectbox(
                "Weather condition",
                ["Clear", "Cloudy", "Rainy"]
            )

        with right:
            st.markdown("### 🛵 Restaurant & delivery partner")

            order_time = st.selectbox(
                "Order time",
                ["Morning", "Afternoon", "Evening", "Night"]
            )

            prep_time = st.slider(
                "Restaurant preparation time (minutes)",
                min_value=10,
                max_value=40,
                value=20,
                step=1
            )

            experience = st.slider(
                "Delivery partner experience (years)",
                min_value=1,
                max_value=10,
                value=5,
                step=1
            )

        submitted = st.form_submit_button(
            "✨ Predict Delivery Time",
            use_container_width=True
        )

    if submitted:
        new_order = pd.DataFrame([{
            "Distance_km": distance,
            "Items": items,
            "Weather": weather,
            "Traffic": traffic,
            "Order_Time": order_time,
            "Restaurant_Preparation_Time": prep_time,
            "Delivery_Partner_Experience": experience
        }])

        # Match the categorical encoding used in training
        new_order = pd.get_dummies(
            new_order,
            drop_first=True,
            dtype=int
        )

        # Ensure exactly the feature columns expected by the model
        new_order = new_order.reindex(
            columns=feature_columns,
            fill_value=0
        )

        try:
            prediction = float(model.predict(new_order)[0])

            st.divider()
            st.markdown("### 🎯 Your delivery estimate")

            result_col, distance_col, prep_col = st.columns(3)

            with result_col:
                st.metric(
                    "Estimated delivery time",
                    f"{prediction:.0f} min"
                )

            with distance_col:
                st.metric(
                    "Delivery distance",
                    f"{distance:.1f} km"
                )

            with prep_col:
                st.metric(
                    "Preparation time",
                    f"{prep_time} min"
                )

            st.markdown(
                f"""
                <div class="hero">
                    <div class="eyebrow">YOUR PREDICTION</div>
                    <h1>⏱️ {prediction:.0f} minutes</h1>
                    <p>
                        Your model's estimated delivery duration
                        for this order.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("### 🧾 Order summary")

            summary_col1, summary_col2, summary_col3 = st.columns(3)

            with summary_col1:
                st.write(f"📍 **Distance:** {distance:.1f} km")
                st.write(f"🍱 **Items:** {items}")

            with summary_col2:
                st.write(f"🌦️ **Weather:** {weather}")
                st.write(f"🚦 **Traffic:** {traffic}")

            with summary_col3:
                st.write(f"🕒 **Order time:** {order_time}")
                st.write(f"🛵 **Experience:** {experience} years")

            st.info(
                "This is a machine learning estimate, not a live "
                "delivery-company ETA. Actual delivery time can vary."
            )

        except Exception as e:
            st.error("Prediction failed. Please check the model inputs.")
            st.exception(e)

    else:
        st.info(
            "👆 Set your order details above and click "
            "'Predict Delivery Time' to see your estimate."
        )


# --------------------------------------------------
# MODEL INSIGHTS
# --------------------------------------------------
elif page == "📊 Model Insights":

    st.markdown("## 📊 Model Insights")
    st.write(
        "Explore which input features contribute most to "
        "the trained model's predictions."
    )

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_

        importance_df = pd.DataFrame({
            "Feature": feature_columns,
            "Importance": importances
        }).sort_values("Importance", ascending=False)

        top_features = importance_df.head(10).copy()

        st.markdown("### 🏆 Top 10 important features")

        st.bar_chart(
            top_features.set_index("Feature")["Importance"],
            horizontal=True
        )

        st.markdown("### Detailed feature importance")

        display_df = importance_df.copy()
        display_df["Importance"] = display_df["Importance"].round(4)

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Feature importance measures the model's relative "
            "reliance on features. It does not prove causation."
        )

    else:
        st.info(
            "This saved model does not expose feature_importances_. "
            "The prediction app can still work normally."
        )


# --------------------------------------------------
# DATASET EXPLORER
# --------------------------------------------------
elif page == "🗂️ Dataset Explorer":

    st.markdown("## 🗂️ Dataset Explorer")
    st.write(
        "Explore the dataset used in this project, if the CSV "
        "file is available in the deployed repository."
    )

    dataset_path = "food_delivery_dataset.csv"

    if os.path.exists(dataset_path):
        try:
            df = pd.read_csv(dataset_path)

            metric1, metric2, metric3 = st.columns(3)

            with metric1:
                st.metric("Total records", f"{len(df):,}")

            with metric2:
                st.metric("Total columns", len(df.columns))

            with metric3:
                st.metric("Missing values", int(df.isna().sum().sum()))

            st.markdown("### 🔎 Preview data")

            st.dataframe(
                df.head(100),
                use_container_width=True,
                hide_index=True
            )

            numeric_columns = df.select_dtypes(
                include="number"
            ).columns.tolist()

            if numeric_columns:
                st.markdown("### 📈 Explore a numeric feature")

                selected_column = st.selectbox(
                    "Choose a column",
                    numeric_columns
                )

                st.bar_chart(df[selected_column].dropna())

                st.markdown("### Statistical summary")
                st.dataframe(
                    df[numeric_columns].describe().round(2),
                    use_container_width=True
                )

        except Exception as e:
            st.error("The dataset could not be read.")
            st.exception(e)

    else:
        st.warning(
            "food_delivery_dataset.csv was not found. "
            "Add the dataset to the repository to use this page."
        )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()

st.markdown(
    """
    <div style="text-align:center; padding: 12px; color:#94a3b8;">
        <b style="color:#c4b5fd;">Food Delivery AI</b><br>
        Minor Project · Data Analysis & Machine Learning<br>
        <small>Predictions are estimates generated by a trained model.</small>
    </div>
    """,
    unsafe_allow_html=True
)
