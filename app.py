
# ============================================================
# SMART DELIVERY ANALYTICS
# Delivery Time Prediction Using Machine Learning
# ============================================================

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Smart Delivery Analytics",
    page_icon="🛵",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
BANNER_PATH = BASE_DIR / "IMG_2855.PNG"


# ============================================================
# 2. PROFESSIONAL DARK THEME
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

.stApp {
    background:
        radial-gradient(ellipse at top left, #35112f 0%, transparent 38%),
        radial-gradient(ellipse at bottom right, #21113d 0%, transparent 35%),
        #090714;
    color: #F8F7FF;
    font-family: 'Inter', sans-serif;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #191128, #0D0B19);
    border-right: 1px solid #442343;
}

[data-testid="stSidebar"] * {
    color: #F8F7FF;
}

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: #FFFFFF !important;
}

p, label, li {
    color: #D0C9E0;
}

.hero {
    padding: 30px;
    margin: 10px 0 25px 0;
    border: 1px solid #663052;
    border-radius: 22px;
    background: linear-gradient(120deg, #1B1028, #110D20);
    box-shadow: 0 10px 35px #00000040;
}

.hero-title {
    font-size: clamp(32px, 4vw, 48px);
    line-height: 1.12;
    font-weight: 800;
    color: white;
}

.hero-subtitle {
    font-size: 16px;
    line-height: 1.7;
    color: #D0C5E0;
}

.pink {
    color: #FF4FA5;
}

.metric-card {
    background: linear-gradient(145deg, #21172F, #141021);
    border: 1px solid #50304E;
    border-radius: 16px;
    padding: 20px;
    min-height: 125px;
}

.metric-label {
    color: #C7BBD8;
    font-size: 13px;
    font-weight: 600;
}

.metric-value {
    color: #FF70B7;
    font-size: 28px;
    font-weight: 800;
    margin-top: 10px;
}

.metric-note {
    color: #B5A9C9;
    font-size: 12px;
    margin-top: 7px;
}

.prediction-card {
    background: linear-gradient(135deg, #32142F, #21133E);
    border: 1px solid #FF4FA5;
    border-radius: 22px;
    padding: 30px;
    text-align: center;
    margin: 15px 0;
}

.prediction-value {
    color: #FF69B4;
    font-size: 52px;
    font-weight: 800;
}

.stButton > button,
.stDownloadButton > button,
.stFormSubmitButton > button {
    background: linear-gradient(90deg, #FF3F9B, #955BFF);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 700;
    min-height: 43px;
}

.stButton > button:hover,
.stDownloadButton > button:hover,
.stFormSubmitButton > button:hover {
    color: white;
    border: 1px solid #FF8BC6;
    filter: brightness(1.1);
}

div[data-testid="stDataFrame"] {
    border: 1px solid #49304E;
    border-radius: 12px;
}

hr {
    border-color: #38243F;
}

footer {
    visibility: hidden;
}

#MainMenu {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. HEADER AND BANNER
# ============================================================

if BANNER_PATH.exists():
    st.image(str(BANNER_PATH), use_container_width=True)

st.markdown("""
<div class="hero">
    <div style="color:#FF70B7;letter-spacing:4px;
                font-size:12px;font-weight:800;">
        SMART DELIVERY ANALYTICS
    </div>
    <div class="hero-title">
        Predict Smarter.<br>
        <span style="color:#FF4FA5;">Deliver Faster.</span>
    </div>
    <p class="hero-subtitle">
        An intelligent machine learning dashboard for food delivery
        time prediction, performance analysis, and data exploration.
    </p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

TARGET_NAMES = [
    "Time_taken(min)",
    "Time_taken (min)",
    "Delivery_Time",
    "Delivery_Time_Minutes",
    "Delivery_Time_Min",
    "delivery_time",
    "DeliveryTime",
    "time_taken"
]


def detect_target(columns):
    """Identify the delivery-time target column automatically."""

    normalized = {
        str(col).strip().lower(): col for col in columns
    }

    for name in TARGET_NAMES:
        if name.lower() in normalized:
            return normalized[name.lower()]

    for col in columns:
        name = str(col).lower().replace(" ", "_")

        if (
            "time" in name
            and ("taken" in name or "delivery" in name)
            and not any(word in name for word in [
                "order", "pickup", "pick_up"
            ])
        ):
            return col

    return None


def clean_data(data):
    """Clean column names, duplicate rows, and missing-value text."""

    data = data.copy()

    data.columns = [
        str(col).strip().replace("\ufeff", "")
        for col in data.columns
    ]

    data = data.loc[:, ~pd.Index(data.columns).duplicated()]
    data = data.drop_duplicates()

    unwanted = [
        col for col in data.columns
        if col.lower().startswith("unnamed:")
        or col.lower() in ["index", "row_id"]
    ]

    data = data.drop(columns=unwanted, errors="ignore")

    for col in data.select_dtypes(
        include=["object", "string"]
    ).columns:
        data[col] = data[col].astype("string").str.strip()

        data[col] = data[col].replace({
            "": pd.NA,
            "nan": pd.NA,
            "None": pd.NA,
            "NULL": pd.NA,
            "null": pd.NA
        })

    return data


def convert_numeric(data, target):
    """Convert numeric-looking strings into actual numbers."""

    data = data.copy()

    for col in data.columns:
        if col == target:
            data[col] = pd.to_numeric(
                data[col].astype(str).str.extract(
                    r"(-?\d+(?:\.\d+)?)",
                    expand=False
                ),
                errors="coerce"
            )

        elif (
            data[col].dtype == "object"
            or str(data[col].dtype) == "string"
        ):
            original = data[col]

            converted = pd.to_numeric(
                original.astype(str).str.replace(
                    ",", "", regex=False
                ),
                errors="coerce"
            )

            total = original.notna().sum()

            if (
                total > 0
                and converted.notna().sum() / total >= 0.90
            ):
                data[col] = converted

    return data


def metric_card(label, value, note):
    """Display a consistent dashboard metric card."""

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-note">{note}</div>
    </div>
    """, unsafe_allow_html=True)


def style_chart(fig):
    """Apply the dashboard's dark chart theme."""

    fig.patch.set_facecolor("#141021")

    for ax in fig.axes:
        ax.set_facecolor("#141021")
        ax.tick_params(colors="#E0D8EC")
        ax.xaxis.label.set_color("#E0D8EC")
        ax.yaxis.label.set_color("#E0D8EC")
        ax.title.set_color("#FFFFFF")

        for spine in ax.spines.values():
            spine.set_color("#49344F")

        ax.grid(
            True,
            linestyle="--",
            alpha=0.18
        )

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ============================================================
# 5. SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:
    st.markdown("## 🛵 Smart Delivery")
    st.caption("Delivery Time Prediction")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Predict Delivery",
            "Model Insights",
            "Dataset Explorer",
            "About Project"
        ]
    )

    st.divider()

    st.markdown("### Dataset")
    uploaded_file = st.file_uploader(
        "Upload your CSV dataset",
        type=["csv"],
        help="Upload the historical delivery dataset."
    )

    st.caption(
        "Target column examples: Time_taken(min), Delivery_Time"
    )


# ============================================================
# 6. LOAD DATASET
# ============================================================

data = None
source_name = None

try:
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        source_name = uploaded_file.name

    else:
        possible_files = [
            BASE_DIR / "delivery_time.csv",
            BASE_DIR / "Food_Delivery_Time_Prediction.csv",
            BASE_DIR / "food_delivery.csv",
            BASE_DIR / "dataset.csv"
        ]

        for file_path in possible_files:
            if file_path.exists():
                data = pd.read_csv(file_path)
                source_name = file_path.name
                break

except Exception as error:
    st.error(f"Unable to read the CSV file: {error}")
    st.stop()


if data is None:
    st.warning(
        "Please upload your delivery dataset using the sidebar, "
        "or place delivery_time.csv in the same folder as app.py."
    )

    st.markdown("### Expected dataset information")
    st.write(
        "Your dataset should contain historical delivery records, "
        "input features such as weather, traffic, distance or "
        "delivery-person information, and a numeric delivery-time "
        "column."
    )

    st.info(
        "Example target column: Time_taken(min). "
        "The application cannot train a meaningful model until "
        "a dataset is provided."
    )
    st.stop()


data = clean_data(data)
target = detect_target(data.columns)

if target is None:
    st.error(
        "Delivery-time target column not detected. Rename the "
        "target column to Time_taken(min) or Delivery_Time."
    )
    st.write("Detected columns:", list(data.columns))
    st.stop()

data = convert_numeric(data, target)
data[target] = pd.to_numeric(data[target], errors="coerce")

data = data.dropna(subset=[target])
data = data[data[target] > 0]

if len(data) < 10:
    st.error(
        "At least 10 usable records with positive delivery times "
        "are required. Please check your CSV dataset."
    )
    st.stop()


# ============================================================
# 7. MODEL SETTINGS
# ============================================================

with st.sidebar:
    st.markdown("### Model settings")

    model_name = st.selectbox(
        "Choose algorithm",
        [
            "Random Forest",
            "Extra Trees",
            "Linear Regression"
        ]
    )

    test_size = st.slider(
        "Test dataset percentage",
        min_value=10,
        max_value=35,
        value=20,
        step=5
    ) / 100

    train_button = st.button(
        "Train / Refresh Model",
        use_container_width=True
    )


# ============================================================
# 8. PREPARE FEATURES AND TRAIN MODEL
# ============================================================

features = data.drop(columns=[target]).copy()

# Remove identifier columns to avoid learning meaningless IDs.
id_columns = [
    col for col in features.columns
    if col.strip().lower() in [
        "id",
        "order_id",
        "orderid",
        "delivery_person_id",
        "serial_no",
        "serial number"
    ]
]

features = features.drop(columns=id_columns, errors="ignore")

# Remove entirely empty and constant columns.
empty_columns = [
    col for col in features.columns
    if features[col].isna().all()
]

constant_columns = [
    col for col in features.columns
    if features[col].nunique(dropna=True) <= 1
]

features = features.drop(
    columns=list(set(empty_columns + constant_columns)),
    errors="ignore"
)

if features.shape[1] == 0:
    st.error("No usable input features were found.")
    st.stop()

numeric_features = features.select_dtypes(
    include=[np.number]
).columns.tolist()

categorical_features = [
    col for col in features.columns
    if col not in numeric_features
]

if not numeric_features and not categorical_features:
    st.error("No usable numeric or categorical features were found.")
    st.stop()


# Save model results in session state.
signature = (
    source_name,
    len(data),
    tuple(data.columns),
    model_name,
    test_size,
    int(pd.util.hash_pandas_object(data, index=True).sum())
)

if (
    train_button
    or st.session_state.get("model_signature") != signature
    or "trained_model" not in st.session_state
):
    with st.spinner("Training the machine learning model..."):

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                features,
                data[target],
                test_size=test_size,
                random_state=42
            )

            transformers = []

            if numeric_features:
                numeric_pipeline = Pipeline([
                    (
                        "imputer",
                        SimpleImputer(strategy="median")
                    ),
                    (
                        "scaler",
                        StandardScaler()
                    )
                ])

                transformers.append(
                    ("numeric", numeric_pipeline, numeric_features)
                )

            if categorical_features:
                categorical_pipeline = Pipeline([
                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent")
                    ),
                    (
                        "encoder",
                        OneHotEncoder(handle_unknown="ignore")
                    )
                ])

                transformers.append(
                    (
                        "categorical",
                        categorical_pipeline,
                        categorical_features
                    )
                )

            preprocessing = ColumnTransformer(
                transformers=transformers,
                remainder="drop"
            )

            if model_name == "Extra Trees":
                estimator = ExtraTreesRegressor(
                    n_estimators=200,
                    random_state=42,
                    n_jobs=-1
                )

            elif model_name == "Linear Regression":
                estimator = LinearRegression()

            else:
                estimator = RandomForestRegressor(
                    n_estimators=200,
                    random_state=42,
                    n_jobs=-1
                )

            model = Pipeline([
                ("preprocessing", preprocessing),
                ("regressor", estimator)
            ])

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            mae = mean_absolute_error(y_test, predictions)

            rmse = np.sqrt(
                mean_squared_error(y_test, predictions)
            )

            r2 = (
                r2_score(y_test, predictions)
                if len(y_test) > 1
                else float("nan")
            )

            st.session_state["trained_model"] = model
            st.session_state["X_test"] = X_test
            st.session_state["y_test"] = y_test
            st.session_state["predictions"] = predictions
            st.session_state["mae"] = mae
            st.session_state["rmse"] = rmse
            st.session_state["r2"] = r2
            st.session_state["model_signature"] = signature
            st.session_state["training_error"] = None

        except Exception as error:
            st.session_state["training_error"] = str(error)


if st.session_state.get("training_error"):
    st.error(
        f"Model training failed: "
        f"{st.session_state['training_error']}"
    )
    st.stop()

if "trained_model" not in st.session_state:
    st.error("The model could not be trained.")
    st.stop()


model = st.session_state["trained_model"]
X_test = st.session_state["X_test"]
y_test = st.session_state["y_test"]
predictions = st.session_state["predictions"]
mae = st.session_state["mae"]
rmse = st.session_state["rmse"]
r2 = st.session_state["r2"]


# ============================================================
# 9. OVERVIEW DASHBOARD
# ============================================================

if page == "Overview":

    st.markdown("## 📊 Dashboard Overview")

    st.write(
        "Monitor delivery-time patterns and evaluate the "
        "performance of your machine learning model."
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(
            "TOTAL RECORDS",
            f"{len(data):,}",
            "Usable delivery records"
        )

    with c2:
        metric_card(
            "AVERAGE DELIVERY",
            f"{data[target].mean():.1f} min",
            "Mean delivery duration"
        )

    with c3:
        metric_card(
            "MEDIAN DELIVERY",
            f"{data[target].median():.1f} min",
            "Middle delivery duration"
        )

    with c4:
        metric_card(
            "INPUT FEATURES",
            str(features.shape[1]),
            "Features used by the model"
        )

    st.divider()

    left, right = st.columns(2)

    with left:
        st.markdown("### Delivery Time Distribution")

        fig, ax = plt.subplots(figsize=(7, 4))

        ax.hist(
            data[target],
            bins=25,
            edgecolor="#25172F"
        )

        ax.set_xlabel("Delivery Time (minutes)")
        ax.set_ylabel("Number of Deliveries")
        ax.set_title("Distribution of Delivery Duration")

        style_chart(fig)

    with right:
        st.markdown("### Statistical Summary")

        summary = data[target].describe().to_frame(
            "Delivery Time (minutes)"
        )

        summary.index = [
            "Count",
            "Mean",
            "Standard Deviation",
            "Minimum",
            "25th Percentile",
            "Median",
            "75th Percentile",
            "Maximum"
        ]

        st.dataframe(
            summary.round(2),
            use_container_width=True
        )

    st.divider()

    st.markdown("### Model Performance")

    a, b, c = st.columns(3)

    with a:
        metric_card(
            "MAE",
            f"{mae:.2f} min",
            "Lower is better"
        )

    with b:
        metric_card(
            "RMSE",
            f"{rmse:.2f} min",
            "Penalizes larger errors"
        )

    with c:
        metric_card(
            "R² SCORE",
            f"{r2:.3f}" if np.isfinite(r2) else "N/A",
            "Closer to 1 is generally better"
        )

    st.caption(
        f"Dataset: {source_name} | "
        f"Algorithm: {model_name} | "
        f"Training records: {len(X_train) if 'X_train' in globals() else len(data) - len(X_test)} | "
        f"Test records: {len(X_test)}"
    )


# ============================================================
# 10. DELIVERY TIME PREDICTION
# ============================================================

elif page == "Predict Delivery":

    st.markdown("## 🛵 Predict Delivery Time")

    st.write(
        "Enter the delivery conditions to estimate how long "
        "the food delivery may take."
    )

    with st.form("prediction_form"):

        st.markdown("### Delivery Details")

        inputs = {}

        left, right = st.columns(2)

        for index, col in enumerate(features.columns):

            container = left if index % 2 == 0 else right

            with container:

                label = col.replace("_", " ").title()

                if col in numeric_features:

                    values = pd.to_numeric(
                        data[col],
                        errors="coerce"
                    ).dropna()

                    if values.empty:
                        default = 0.0
                        minimum = -1_000_000.0
                        maximum = 1_000_000.0
                    else:
                        default = float(values.median())

                        minimum = float(values.min())
                        maximum = float(values.max())

                        if minimum == maximum:
                            minimum -= 1
                            maximum += 1

                    inputs[col] = st.number_input(
                        label,
                        min_value=minimum,
                        max_value=maximum,
                        value=float(np.clip(
                            default, minimum, maximum
                        )),
                        key=f"input_{col}"
                    )

                else:

                    options = sorted(
                        data[col]
                        .dropna()
                        .astype(str)
                        .unique()
                        .tolist()
                    )

                    if options:
                        inputs[col] = st.selectbox(
                            label,
                            options=options,
                            key=f"input_{col}"
                        )
                    else:
                        inputs[col] = st.text_input(
                            label,
                            key=f"input_{col}"
                        )

        submitted = st.form_submit_button(
            "✨ Predict Delivery Time",
            use_container_width=True
        )

    if submitted:

        try:
            input_df = pd.DataFrame(
                [inputs],
                columns=features.columns
            )

            for col in numeric_features:
                input_df[col] = pd.to_numeric(
                    input_df[col],
                    errors="coerce"
                )

            estimate = float(model.predict(input_df)[0])
            estimate = max(0.0, estimate)

            st.markdown(f"""
            <div class="prediction-card">
                <div style="font-size:14px;letter-spacing:2px;
                            color:#D8CBE8;">
                    ESTIMATED DELIVERY TIME
                </div>
                <div class="prediction-value">
                    {estimate:.1f} min
                </div>
                <div style="color:#E3D8F2;">
                    Prediction generated using {model_name}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.info(
                f"The model's test-set MAE is {mae:.2f} minutes. "
                f"As a rough error guide only, the estimate plus "
                f"or minus the MAE is "
                f"{max(0, estimate - mae):.1f} to "
                f"{estimate + mae:.1f} minutes. This is not a "
                f"guaranteed delivery window."
            )

            result = input_df.copy()

            result["Predicted_Delivery_Time_Minutes"] = round(
                estimate, 2
            )

            st.download_button(
                "⬇ Download Prediction",
                data=result.to_csv(index=False).encode("utf-8"),
                file_name="delivery_prediction.csv",
                mime="text/csv",
                use_container_width=True
            )

            with st.expander("View prediction input details"):
                st.dataframe(
                    result,
                    use_container_width=True
                )

        except Exception as error:
            st.error(f"Prediction failed: {error}")


# ============================================================
# 11. MODEL INSIGHTS
# ============================================================

elif page == "Model Insights":

    st.markdown("## 📈 Model Performance & Insights")

    st.write(
        "Evaluate the model using test data that was kept "
        "separate from the training data."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        metric_card("MAE", f"{mae:.2f} min", "Lower is better")

    with c2:
        metric_card("RMSE", f"{rmse:.2f} min", "Lower is better")

    with c3:
        metric_card(
            "R² SCORE",
            f"{r2:.3f}" if np.isfinite(r2) else "N/A",
            "Can be negative"
        )

    actual = np.asarray(y_test, dtype=float)
    predicted = np.asarray(predictions, dtype=float)

    residuals = actual - predicted

    left, right = st.columns(2)

    with left:

        st.markdown("### Actual vs Predicted")

        fig, ax = plt.subplots(figsize=(7, 5))

        ax.scatter(
            actual,
            predicted,
            alpha=0.65
        )

        lower = min(actual.min(), predicted.min())
        upper = max(actual.max(), predicted.max())

        ax.plot(
            [lower, upper],
            [lower, upper],
            linestyle="--",
            linewidth=2
        )

        ax.set_xlabel("Actual Time (minutes)")
        ax.set_ylabel("Predicted Time (minutes)")
        ax.set_title("Actual vs Predicted Delivery Time")

        style_chart(fig)

    with right:

        st.markdown("### Prediction Error Distribution")

        fig, ax = plt.subplots(figsize=(7, 5))

        ax.hist(
            residuals,
            bins=20,
            edgecolor="#25172F"
        )

        ax.axvline(
            0,
            linestyle="--",
            linewidth=2
        )

        ax.set_xlabel("Actual - Predicted (minutes)")
        ax.set_ylabel("Number of Deliveries")
        ax.set_title("Residual Distribution")

        style_chart(fig)

    # Feature importance
    st.markdown("### 🔍 Feature Importance")

    try:

        preprocessing = model.named_steps["preprocessing"]
        estimator = model.named_steps["regressor"]

        transformed_names = preprocessing.get_feature_names_out()

        if hasattr(estimator, "feature_importances_"):

            importance_values = estimator.feature_importances_

        elif hasattr(estimator, "coef_"):

            importance_values = np.abs(
                np.ravel(estimator.coef_)
            )

        else:

            importance_values = None

        if (
            importance_values is not None
            and len(transformed_names) == len(importance_values)
        ):

            importance_df = pd.DataFrame({
                "Feature": [
                    str(name)
                    .replace("numeric__", "")
                    .replace("categorical__", "")
                    for name in transformed_names
                ],
                "Importance": importance_values
            })

            importance_df = (
                importance_df
                .groupby("Feature", as_index=False)["Importance"]
                .sum()
                .sort_values("Importance", ascending=False)
            )

            top_features = importance_df.head(12).sort_values(
                "Importance", ascending=True
            )

            fig, ax = plt.subplots(
                figsize=(9, max(4, len(top_features) * 0.4))
            )

            ax.barh(
                top_features["Feature"],
                top_features["Importance"]
            )

            ax.set_xlabel("Importance / Absolute Coefficient")
            ax.set_title("Most Influential Features")

            style_chart(fig)

            st.dataframe(
                importance_df.head(20),
                use_container_width=True
            )

            st.caption(
                "Tree-based importance and absolute linear "
                "coefficients have different meanings. They do "
                "not establish causation."
            )

        else:

            st.info(
                "Feature importance is unavailable for the "
                "selected algorithm."
            )

    except Exception as error:

        st.info(f"Could not calculate feature importance: {error}")

    st.markdown("### Test Dataset Predictions")

    results = pd.DataFrame({
        "Actual_Time_Minutes": actual,
        "Predicted_Time_Minutes": predicted,
        "Error_Minutes": residuals,
        "Absolute_Error_Minutes": np.abs(residuals)
    })

    st.dataframe(
        results.round(2),
        use_container_width=True
    )

    st.download_button(
        "⬇ Download Evaluation Results",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="delivery_model_evaluation.csv",
        mime="text/csv"
    )


# ============================================================
# 12. DATASET EXPLORER
# ============================================================

elif page == "Dataset Explorer":

    st.markdown("## 🗂 Dataset Explorer")

    st.write(
        "Inspect records, identify missing values, and "
        "understand relationships between numeric variables."
    )

    search = st.text_input(
        "Search dataset",
        placeholder="Search city, weather, traffic, or other values..."
    )

    rows_to_show = st.slider(
        "Number of rows to display",
        min_value=10,
        max_value=200,
        value=50,
        step=10
    )

    view_data = data.copy()

    if search.strip():

        mask = view_data.astype(str).apply(
            lambda column: column.str.contains(
                search.strip(),
                case=False,
                regex=False,
                na=False
            )
        ).any(axis=1)

        view_data = view_data.loc[mask]

    st.caption(
        f"Displaying {min(rows_to_show, len(view_data))} "
        f"of {len(view_data)} matching records."
    )

    st.dataframe(
        view_data.head(rows_to_show),
        use_container_width=True,
        height=400
    )

    st.download_button(
        "⬇ Download Dataset",
        data=data.to_csv(index=False).encode("utf-8"),
        file_name="delivery_dataset.csv",
        mime="text/csv"
    )

    st.markdown("### Missing Values")

    missing = data.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if missing.empty:

        st.success("No missing values found in the cleaned dataset.")

    else:

        missing_df = missing.to_frame("Missing Values")
        missing_df["Missing Percentage"] = (
            missing_df["Missing Values"] / len(data) * 100
        ).round(2)

        st.dataframe(
            missing_df,
            use_container_width=True
        )

    st.markdown("### Correlation Heatmap")

    numeric_data = data.select_dtypes(include=[np.number])

    if numeric_data.shape[1] >= 2:

        correlation = numeric_data.corr()

        fig, ax = plt.subplots(
            figsize=(
                min(14, max(7, len(correlation.columns) * 0.9)),
                min(12, max(5, len(correlation.columns) * 0.7))
            )
        )

        image = ax.imshow(
            correlation,
            vmin=-1,
            vmax=1,
            aspect="auto"
        )

        ax.set_xticks(range(len(correlation.columns)))
        ax.set_xticklabels(
            correlation.columns,
            rotation=60,
            ha="right",
            fontsize=8
        )

        ax.set_yticks(range(len(correlation.columns)))
        ax.set_yticklabels(
            correlation.columns,
            fontsize=8
        )

        ax.set_title("Numeric Feature Correlation")

        fig.colorbar(
            image,
            ax=ax,
            fraction=0.046,
            pad=0.04
        )

        style_chart(fig)

    else:

        st.info(
            "At least two numeric columns are required "
            "to generate a correlation heatmap."
        )


# ============================================================
# 13. ABOUT THE PROJECT
# ============================================================

elif page == "About Project":

    st.markdown("## 💡 About Smart Delivery Analytics")

    st.markdown("""
    ### Project Title
    Delivery Time Prediction Using Machine Learning

    ### Objective
    To estimate the time required for food delivery by learning
    patterns from historical delivery data.

    ### Project Workflow

    1. Collect historical delivery data.
    2. Clean duplicate rows and normalize missing values.
    3. Identify the delivery-time target.
    4. Separate input features from the target.
    5. Split data into training and testing sets.
    6. Impute missing values and encode categorical features.
    7. Train a regression model.
    8. Evaluate predictions using MAE, RMSE, and R².
    9. Generate predictions for new delivery inputs.

    ### Machine Learning Algorithms

    - Random Forest Regressor
    - Extra Trees Regressor
    - Linear Regression

    ### Technology Stack

    - Python
    - Streamlit
    - Pandas and NumPy
    - Scikit-learn
    - Matplotlib

    ### Limitations

    Predictions depend on the quality and coverage of the
    training dataset. This application does not automatically
    access live GPS or traffic information. Estimated delivery
    times are not guaranteed arrival times.
    """)

    st.markdown("### Current Configuration")

    st.write({
        "Dataset": source_name,
        "Target Column": target,
        "Algorithm": model_name,
        "Total Records": len(data),
        "Input Features": len(features.columns),
        "Test Records": len(X_test)
    })


# ============================================================
# 14. FOOTER
# ============================================================

st.divider()

st.markdown("""
<div style="text-align:center;padding:12px;color:#A69AB9;
            font-size:12px;line-height:1.8;">
    <b style="color:#FF70B7;">SMART DELIVERY ANALYTICS</b><br>
    Delivery Time Prediction Using Machine Learning<br>
    Built with Python · Streamlit · Scikit-learn
</div>
""", unsafe_allow_html=True)
