import os
import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Food Delivery AI | Smart Predictor",
    page_icon="🍔",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# CUSTOM CSS: BLACK + NEON PINK + PURPLE THEME
# Text is white for strong readability.
# --------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --black: #09060D;
        --black-card: #120B18;
        --purple: #351044;
        --purple-light: #54206A;
        --pink: #FF2E93;
        --pink-light: #FF70B8;
        --white: #FFFFFF;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"] {
        background: var(--black) !important;
        color: var(--white) !important;
    }

    [data-testid="stHeader"] {
        background: rgba(9, 6, 13, 0.98) !important;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1250px;
    }

    /* White text across the application */
    .stApp p, .stApp li, .stApp label, .stApp span,
    .stApp .stMarkdown, .stApp [data-testid="stCaptionContainer"],
    .stApp [data-testid="stWidgetLabel"] {
        color: var(--white);
    }

    h1, h2, h3, h4, h5, h6 {
        color: var(--white) !important;
    }

    a { color: var(--pink-light) !important; }

    /* Hero panel */
    .hero {
        padding: 30px;
        border-radius: 24px;
        background: linear-gradient(120deg, #260D25 0%, #160D20 55%, #251039 100%);
        border: 1px solid var(--pink);
        box-shadow: 0 8px 28px rgba(255, 46, 147, 0.10);
        margin-bottom: 25px;
        transition: transform 0.25s ease, border-color 0.25s ease;
    }

    .hero:hover {
        transform: translateY(-2px);
        border-color: var(--pink-light);
    }

    .hero h1 {
        font-size: clamp(1.8rem, 4vw, 2.8rem);
        font-weight: 800;
        line-height: 1.2;
        color: var(--white) !important;
        margin-bottom: 12px;
    }

    .hero p {
        color: var(--white) !important;
        font-size: 1rem;
        line-height: 1.7;
    }

    .eyebrow {
        color: var(--pink-light) !important;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    .section-heading {
        font-size: 1.35rem;
        font-weight: 750;
        color: var(--white) !important;
        margin: 12px 0 5px 0;
    }

    .section-subtitle {
        color: var(--white) !important;
        font-size: 0.92rem;
        margin-bottom: 18px;
    }

    /* Metrics */
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #1D0E27, #120B18) !important;
        border: 1px solid var(--purple-light) !important;
        border-radius: 18px;
        padding: 20px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    div[data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        border-color: var(--pink) !important;
    }

    [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] *,
    [data-testid="stMetricValue"], [data-testid="stMetricValue"] * {
        color: var(--white) !important;
    }

    div[data-testid="stMetricValue"] {
        color: var(--pink-light) !important;
    }

    /* Buttons */
    .stButton > button, .stFormSubmitButton > button,
    [data-testid="stDownloadButton"] > button {
        background: linear-gradient(100deg, var(--pink), #8F24C7) !important;
        color: var(--white) !important;
        border: 1px solid var(--pink-light) !important;
        border-radius: 12px !important;
        padding: 0.75rem 1.2rem;
        font-weight: 750 !important;
        min-height: 3rem;
        transition: transform 0.2s ease, filter 0.2s ease;
    }

    .stButton > button *, .stFormSubmitButton > button *,
    [data-testid="stDownloadButton"] > button * {
        color: var(--white) !important;
    }

    .stButton > button:hover, .stFormSubmitButton > button:hover,
    [data-testid="stDownloadButton"] > button:hover {
        transform: translateY(-2px);
        filter: brightness(1.12);
    }

    /* Form panels */
    div[data-testid="stForm"] {
        background: #100A16 !important;
        border: 1px solid var(--purple-light) !important;
        border-radius: 20px;
        padding: 22px;
    }

    /* Input fields */
    .stTextInput input, .stNumberInput input, .stDateInput input,
    .stTimeInput input, [data-baseweb="input"] input,
    .stTextArea textarea {
        background: #1B0E27 !important;
        color: var(--white) !important;
        -webkit-text-fill-color: var(--white) !important;
        border-color: var(--purple-light) !important;
        caret-color: var(--pink);
    }

    input::placeholder, textarea::placeholder {
        color: var(--white) !important;
        -webkit-text-fill-color: var(--white) !important;
        opacity: 1;
    }

    [data-baseweb="select"] > div,
    [data-baseweb="popover"] > div,
    [data-baseweb="menu"] {
        background-color: #1B0E27 !important;
        color: var(--white) !important;
        border-color: var(--purple-light) !important;
    }

    [data-baseweb="select"] *, [data-baseweb="menu"] *, [role="option"] {
        color: var(--white) !important;
    }

    [data-baseweb="menu"] [aria-selected="true"],
    [role="option"]:hover {
        background-color: #54206A !important;
    }

    .stApp label, [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] label {
        color: var(--white) !important;
        font-weight: 600;
    }

    /* Slider, radio, checkbox and tabs */
    [data-testid="stSlider"] [role="slider"] {
        background-color: var(--pink) !important;
        border-color: var(--pink) !important;
    }

    [data-testid="stSlider"] *, .stCheckbox *, .stRadio * {
        color: var(--white) !important;
    }

    .stTabs [role="tab"] {
        color: var(--white) !important;
        font-weight: 650;
    }

    .stTabs [role="tab"][aria-selected="true"] {
        color: var(--pink-light) !important;
        border-bottom-color: var(--pink) !important;
    }

    .stTabs [data-baseweb="tab-highlight"] {
        background-color: var(--pink) !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #100815 !important;
        border-right: 1px solid var(--purple-light);
    }

    section[data-testid="stSidebar"] * {
        color: var(--white) !important;
    }

    section[data-testid="stSidebar"] [aria-checked="true"] {
        background-color: #351044 !important;
        border-radius: 10px;
    }

    /* Tables and dataframe */
    [data-testid="stDataFrame"], [data-testid="stTable"] {
        border: 1px solid var(--purple-light);
        border-radius: 12px;
    }

    [data-testid="stDataFrame"] *, [data-testid="stTable"] * {
        color: var(--white) !important;
    }

    /* Alerts and separators */
    [data-testid="stAlert"] {
        background: #170D20 !important;
        border: 1px solid var(--purple-light) !important;
        border-radius: 14px;
    }

    [data-testid="stAlert"] * { color: var(--white) !important; }
    hr, [data-testid="stDivider"] { border-color: var(--purple-light) !important; }

    footer, #MainMenu { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# LOAD MODEL AND FEATURES
# --------------------------------------------------
@st.cache_resource
def load_model_files():
    """Load the trained model and the exact feature order used in training."""
    model = joblib.load("food_delivery_model.pkl")
    features = joblib.load("model_features.pkl")
    return model, list(features)


try:
    model, feature_columns = load_model_files()
except Exception as exc:
    st.error("The model files could not be loaded.")
    st.write(
        "Make sure `food_delivery_model.pkl` and `model_features.pkl` "
        "are present in the same GitHub repository as `app.py`."
    )
    st.exception(exc)
    st.stop()

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">SMART DELIVERY ANALYTICS · MACHINE LEARNING</div>
        <h1>🍔 Food Delivery AI</h1>
        <p>
            Predict estimated delivery time using your order details
            and a trained machine learning model. Explore the prediction,
            order summary and model insights in one place.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------
with st.sidebar:
    st.markdown("## 🍽️ Delivery AI")
    st.caption("Your smart delivery prediction dashboard")

    page = st.radio(
        "Navigate",
        [
            "🚀 Delivery Predictor",
            "📊 Model Insights",
            "🗂️ Dataset Explorer",
        ],
    )

    st.divider()
    st.markdown("### About this project")
    st.write(
        "A machine learning minor project for estimating food delivery time "
        "from order and delivery features."
    )
    st.caption("Model: saved trained model")
    st.caption("Dataset: demonstration dataset")

# --------------------------------------------------
# DELIVERY PREDICTOR
# --------------------------------------------------
if page == "🚀 Delivery Predictor":
    st.markdown(
        '<div class="section-heading">🚀 Configure your order</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-subtitle">Adjust the details below and generate a delivery estimate.</div>',
        unsafe_allow_html=True,
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
                step=0.5,
            )
            items = st.number_input(
                "Number of food items",
                min_value=1,
                max_value=20,
                value=3,
                step=1,
            )
            traffic = st.selectbox("Traffic level", ["Low", "Medium", "High"])
            weather = st.selectbox("Weather condition", ["Clear", "Cloudy", "Rainy"])

        with right:
            st.markdown("### 🛵 Restaurant & delivery partner")
            order_time = st.selectbox(
                "Order time", ["Morning", "Afternoon", "Evening", "Night"]
            )
            prep_time = st.slider(
                "Restaurant preparation time (minutes)",
                min_value=10,
                max_value=40,
                value=20,
                step=1,
            )
            experience = st.slider(
                "Delivery partner experience (years)",
                min_value=1,
                max_value=10,
                value=5,
                step=1,
            )

        submitted = st.form_submit_button(
            "✨ Predict Delivery Time", use_container_width=True
        )

    if submitted:
        new_order = pd.DataFrame(
            [
                {
                    "Distance_km": distance,
                    "Items": items,
                    "Weather": weather,
                    "Traffic": traffic,
                    "Order_Time": order_time,
                    "Restaurant_Preparation_Time": prep_time,
                    "Delivery_Partner_Experience": experience,
                }
            ]
        )

        # Encode categorical inputs in the same style as the training code.
        new_order = pd.get_dummies(new_order, drop_first=True, dtype=int)
        # Ensure the input columns and their order match the saved model.
        new_order = new_order.reindex(columns=feature_columns, fill_value=0)

        try:
            prediction = float(model.predict(new_order)[0])
            st.divider()
            st.markdown("### 🎯 Your delivery estimate")

            result_col, distance_col, prep_col = st.columns(3)
            with result_col:
                st.metric("Estimated delivery time", f"{prediction:.0f} min")
            with distance_col:
                st.metric("Delivery distance", f"{distance:.1f} km")
            with prep_col:
                st.metric("Preparation time", f"{prep_time} min")

            st.markdown(
                f"""
                <div class="hero">
                    <div class="eyebrow">YOUR PREDICTION</div>
                    <h1>⏱️ {prediction:.0f} minutes</h1>
                    <p>Your model's estimated delivery duration for this order.</p>
                </div>
                """,
                unsafe_allow_html=True,
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
                "This is a machine learning estimate, not a live delivery-company ETA. "
                "Actual delivery time can vary."
            )
        except Exception as exc:
            st.error("Prediction failed. Please check the model inputs and feature names.")
            st.exception(exc)
    else:
        st.info(
            "👆 Set your order details above and click 'Predict Delivery Time' "
            "to see your estimate."
        )

# --------------------------------------------------
# MODEL INSIGHTS
# --------------------------------------------------
elif page == "📊 Model Insights":
    st.markdown("## 📊 Model Insights")
    st.write(
        "Explore which input features contribute most to the trained model's predictions."
    )

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_

        if len(importances) != len(feature_columns):
            st.warning(
                "The number of model feature importances does not match the saved feature list. "
                "Please check that `model_features.pkl` matches the trained model."
            )
        else:
            importance_df = pd.DataFrame(
                {"Feature": feature_columns, "Importance": importances}
            ).sort_values("Importance", ascending=False)
            top_features = importance_df.head(10).copy()

            st.markdown("### 🏆 Top 10 important features")
            st.bar_chart(
                top_features.set_index("Feature")["Importance"],
                horizontal=True,
            )

            st.markdown("### Detailed feature importance")
            display_df = importance_df.copy()
            display_df["Importance"] = display_df["Importance"].round(4)
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            st.caption(
                "Feature importance measures the model's relative reliance on features. "
                "It does not prove causation."
            )
    elif hasattr(model, "coef_"):
        st.info(
            "This model exposes coefficients rather than `feature_importances_`. "
            "Coefficient values are shown below; their interpretation depends on the model and feature scaling."
        )
        coefficients = model.coef_
        if getattr(coefficients, "ndim", 1) > 1:
            coefficients = coefficients[0]
        if len(coefficients) == len(feature_columns):
            coefficient_df = pd.DataFrame(
                {"Feature": feature_columns, "Coefficient": coefficients}
            )
            coefficient_df["Coefficient"] = coefficient_df["Coefficient"].round(4)
            st.dataframe(coefficient_df, use_container_width=True, hide_index=True)
        else:
            st.warning("The coefficient count does not match the saved feature list.")
    else:
        st.info(
            "This saved model does not expose feature importances or coefficients. "
            "The prediction app can still work normally."
        )

# --------------------------------------------------
# DATASET EXPLORER
# --------------------------------------------------
elif page == "🗂️ Dataset Explorer":
    st.markdown("## 🗂️ Dataset Explorer")
    st.write(
        "Explore the dataset used in this project, if the CSV file is available in the deployed repository."
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
            st.dataframe(df.head(100), use_container_width=True, hide_index=True)

            numeric_columns = df.select_dtypes(include="number").columns.tolist()
            if numeric_columns:
                st.markdown("### 📈 Explore a numeric feature")
                selected_column = st.selectbox(
                    "Choose a column", numeric_columns
                )
                st.bar_chart(df[selected_column].dropna())
                st.markdown("### Statistical summary")
                st.dataframe(
                    df[numeric_columns].describe().round(2),
                    use_container_width=True,
                )
            else:
                st.info("The dataset does not contain numeric columns to chart.")
        except Exception as exc:
            st.error("The dataset could not be read.")
            st.exception(exc)
    else:
        st.warning(
            "`food_delivery_dataset.csv` was not found. Add the dataset to the repository to use this page."
        )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()
st.markdown(
    """
    <div style="text-align:center; padding:12px; color:#FFFFFF;">
        <b style="color:#FF2E93;">Food Delivery AI</b><br>
        Minor Project · Data Analysis & Machine Learning<br>
        <small style="color:#FFFFFF;">Predictions are estimates generated by a trained model.</small>
    </div>
    """,
    unsafe_allow_html=True,
)
