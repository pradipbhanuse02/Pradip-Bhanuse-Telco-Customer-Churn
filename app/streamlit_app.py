"""Streamlit dashboard for exploring and predicting Telco customer churn."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix, roc_curve

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import evaluate_model, train_model
from src.preprocessing import load_dataset, prepare_features, split_dataset

DATA_PATH = PROJECT_ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_NAMES = ["Logistic Regression", "KNN", "SVM", "Decision Tree", "Random Forest"]
FEATURE_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]

st.set_page_config(
    page_title="Customer Risk Predictor",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="auto",
)

st.markdown(
    """
    <style>
    :root {
        --paper: #f3f7f4;
        --ink: #182722;
        --muted: #60716a;
        --forest: #176b53;
        --lime: #d6ee91;
        --coral: #d96f52;
        --line: #dce6df;
    }
    .stApp { background: var(--paper); color: var(--ink); }
    [data-testid="stSidebar"] { background: #e7efe9; border-right: 1px solid var(--line); }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 { color: var(--ink); }
    h1, h2, h3 { color: var(--ink); letter-spacing: 0; }
    [data-testid="stMetric"] {
        background: #ffffff; border: 1px solid var(--line); border-left: 4px solid var(--forest);
        border-radius: 6px; padding: 14px 16px;
    }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    div.stButton > button, div.stFormSubmitButton > button, div.stDownloadButton > button {
        border-radius: 6px; font-weight: 650; border-color: var(--forest);
    }
    div.stButton > button[kind="primary"], div.stFormSubmitButton > button[kind="primary"] {
        background: var(--forest); color: white;
    }
    [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] { gap: 1rem; }
    .eyebrow { color: var(--forest); font-size: 0.78rem; font-weight: 750; text-transform: uppercase; }
    .subtle { color: var(--muted); }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Loading customer data...")
def get_data() -> pd.DataFrame:
    """Load the project Telco dataset for dashboard summaries."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    return load_dataset(DATA_PATH)


@st.cache_resource(show_spinner="Training the churn model...")
def get_model_artifacts(model_name: str) -> tuple:
    """Train and cache one selected model with its held-out predictions and metrics."""
    data = get_data()
    x_train, x_test, y_train, y_test = split_dataset(data)
    model = train_model(x_train, y_train, model_name=model_name)
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    metrics = evaluate_model(model, x_test, y_test)
    return model, x_test, y_test, predictions, probabilities, metrics


@st.cache_data(show_spinner="Comparing all classifiers on the held-out test set...")
def get_model_comparison() -> pd.DataFrame:
    """Return held-out classification metrics for every supported model."""
    records = []
    for model_name in MODEL_NAMES:
        metrics = get_model_artifacts(model_name)[-1]
        records.append({"Model": model_name, **metrics})
    return pd.DataFrame(records).sort_values("f1", ascending=False)


def churn_rate_by(data: pd.DataFrame, column: str) -> pd.DataFrame:
    """Return customer counts and churn percentages grouped by one feature."""
    grouped = data.assign(ChurnFlag=data["Churn"].eq("Yes").astype(int))
    summary = (
        grouped.groupby(column, dropna=False, observed=True)
        .agg(Customers=("ChurnFlag", "size"), ChurnRate=("ChurnFlag", "mean"))
        .reset_index()
    )
    summary["ChurnRate"] *= 100
    return summary


def render_dashboard(data: pd.DataFrame, metrics: dict[str, float], model_name: str) -> None:
    """Render portfolio-level churn KPIs and customer-segment charts."""
    st.markdown('<div class="eyebrow">Customer retention / portfolio</div>', unsafe_allow_html=True)
    st.title("Churn at a glance")
    st.caption("A snapshot of customer risk, tenure patterns, and contract mix.")

    total_customers = len(data)
    churned_customers = int(data["Churn"].eq("Yes").sum())
    churn_rate = churned_customers / total_customers
    kpi_columns = st.columns(4)
    kpi_columns[0].metric("Customers", f"{total_customers:,}")
    kpi_columns[1].metric("Churn rate", f"{churn_rate:.1%}")
    kpi_columns[2].metric("Customers churned", f"{churned_customers:,}")
    kpi_columns[3].metric(f"{model_name} ROC-AUC", f"{metrics['roc_auc']:.3f}")

    left, right = st.columns([1, 1.35])
    with left:
        st.subheader("Customer status")
        status = data["Churn"].value_counts().rename_axis("Status").reset_index(name="Customers")
        fig = px.pie(
            status,
            names="Status",
            values="Customers",
            hole=0.66,
            color="Status",
            color_discrete_map={"No": "#176b53", "Yes": "#e17a5b"},
        )
        fig.update_traces(textinfo="percent", textfont_size=13)
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), legend_title_text="Churn")
        st.plotly_chart(fig, width="stretch")

    with right:
        st.subheader("Churn by tenure")
        tenure_data = data.copy()
        tenure_data["Tenure group"] = pd.cut(
            tenure_data["tenure"],
            bins=[-1, 6, 12, 24, 36, 48, 60, 72],
            labels=["0-6 mo", "7-12 mo", "1-2 yr", "2-3 yr", "3-4 yr", "4-5 yr", "5-6 yr"],
            include_lowest=True,
        )
        tenure_summary = churn_rate_by(tenure_data, "Tenure group")
        fig = px.bar(
            tenure_summary,
            x="Tenure group",
            y="ChurnRate",
            color="ChurnRate",
            color_continuous_scale=["#d6ee91", "#176b53"],
            labels={"ChurnRate": "Churn rate (%)"},
        )
        fig.update_layout(coloraxis_showscale=False, yaxis_ticksuffix="%", margin=dict(t=8, b=4))
        st.plotly_chart(fig, width="stretch")

    left, right = st.columns(2)
    with left:
        st.subheader("Contract risk")
        contract_summary = churn_rate_by(data, "Contract").sort_values("ChurnRate", ascending=True)
        fig = px.bar(
            contract_summary,
            x="ChurnRate",
            y="Contract",
            orientation="h",
            text=contract_summary["ChurnRate"].map(lambda value: f"{value:.1f}%"),
            color="ChurnRate",
            color_continuous_scale=["#d6ee91", "#d96f52"],
            labels={"ChurnRate": "Churn rate (%)"},
        )
        fig.update_layout(coloraxis_showscale=False, xaxis_ticksuffix="%", margin=dict(t=8, b=4))
        st.plotly_chart(fig, width="stretch")

    with right:
        st.subheader("Churn by internet service")
        internet_summary = churn_rate_by(data, "InternetService").sort_values("ChurnRate", ascending=False)
        fig = px.bar(
            internet_summary,
            x="InternetService",
            y="ChurnRate",
            text=internet_summary["ChurnRate"].map(lambda value: f"{value:.1f}%"),
            color="InternetService",
            color_discrete_sequence=["#d96f52", "#176b53", "#91b4a1"],
            labels={"ChurnRate": "Churn rate (%)", "InternetService": "Service"},
        )
        fig.update_layout(showlegend=False, yaxis_ticksuffix="%", margin=dict(t=8, b=4))
        st.plotly_chart(fig, width="stretch")


def render_prediction(model, model_name: str) -> None:
    """Render an individual-customer input form and churn-risk result."""
    st.markdown('<div class="eyebrow">Individual assessment</div>', unsafe_allow_html=True)
    st.title("Predict customer risk")
    st.caption(f"Enter customer account and service details to estimate churn risk with {model_name}.")

    with st.form("customer_prediction_form"):
        st.subheader("Customer and account")
        row_one = st.columns(4)
        customer_id = row_one[0].text_input("Customer ID", value="New customer")
        gender = row_one[1].selectbox("Gender", ["Female", "Male"])
        senior = row_one[2].selectbox("Senior citizen", ["No", "Yes"])
        tenure = row_one[3].slider("Tenure (months)", min_value=0, max_value=72, value=12)

        row_two = st.columns(4)
        partner = row_two[0].selectbox("Partner", ["No", "Yes"])
        dependents = row_two[1].selectbox("Dependents", ["No", "Yes"])
        contract = row_two[2].selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = row_two[3].selectbox("Paperless billing", ["Yes", "No"])

        st.subheader("Services and billing")
        row_three = st.columns(4)
        phone = row_three[0].selectbox("Phone service", ["Yes", "No"])
        multiple_lines = row_three[1].selectbox("Multiple lines", ["No", "Yes", "No phone service"])
        internet = row_three[2].selectbox("Internet service", ["Fiber optic", "DSL", "No"])
        payment = row_three[3].selectbox(
            "Payment method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )

        row_four = st.columns(4)
        online_security = row_four[0].selectbox("Online security", ["No", "Yes", "No internet service"])
        online_backup = row_four[1].selectbox("Online backup", ["No", "Yes", "No internet service"])
        device_protection = row_four[2].selectbox("Device protection", ["No", "Yes", "No internet service"])
        tech_support = row_four[3].selectbox("Tech support", ["No", "Yes", "No internet service"])

        row_five = st.columns(4)
        streaming_tv = row_five[0].selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = row_five[1].selectbox("Streaming movies", ["No", "Yes", "No internet service"])
        monthly_charges = row_five[2].number_input("Monthly charges ($)", min_value=0.0, max_value=200.0, value=70.0)
        total_charges = row_five[3].number_input(
            "Total charges ($)", min_value=0.0, max_value=20000.0, value=monthly_charges * tenure
        )

        submitted = st.form_submit_button("Assess churn risk", type="primary", width="stretch")

    if submitted:
        customer = pd.DataFrame(
            [
                {
                    "customerID": customer_id,
                    "gender": gender,
                    "SeniorCitizen": int(senior == "Yes"),
                    "Partner": partner,
                    "Dependents": dependents,
                    "tenure": tenure,
                    "PhoneService": phone,
                    "MultipleLines": multiple_lines,
                    "InternetService": internet,
                    "OnlineSecurity": online_security,
                    "OnlineBackup": online_backup,
                    "DeviceProtection": device_protection,
                    "TechSupport": tech_support,
                    "StreamingTV": streaming_tv,
                    "StreamingMovies": streaming_movies,
                    "Contract": contract,
                    "PaperlessBilling": paperless,
                    "PaymentMethod": payment,
                    "MonthlyCharges": monthly_charges,
                    "TotalCharges": total_charges,
                }
            ]
        )
        features = prepare_features(customer).reindex(columns=FEATURE_COLUMNS)
        probability = float(model.predict_proba(features)[0, 1])
        predicted_churn = probability >= 0.5
        result_left, result_right = st.columns([1, 2])
        with result_left:
            st.metric("Estimated churn probability", f"{probability:.1%}")
        with result_right:
            if probability >= 0.7:
                st.error("Higher risk: consider a timely retention follow-up.")
            elif probability >= 0.4:
                st.warning("Elevated risk: review service and contract needs.")
            else:
                st.success("Lower risk based on the details entered.")
            st.progress(probability)
        st.caption(f"Predicted class at a 50% threshold: {'Churn' if predicted_churn else 'No churn'}. This estimate is decision support, not a guarantee.")


def render_batch(model, model_name: str) -> None:
    """Render batch CSV validation, prediction, and download controls."""
    st.markdown('<div class="eyebrow">Batch processing</div>', unsafe_allow_html=True)
    st.title("Score a customer file")
    st.caption(f"Upload a CSV containing the same customer fields used during training. Selected model: {model_name}.")
    template = pd.DataFrame(columns=FEATURE_COLUMNS).to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV template", template, "customer_churn_template.csv", "text/csv")
    uploaded_file = st.file_uploader("Customer CSV", type=["csv"], accept_multiple_files=False)
    if uploaded_file is None:
        st.info("Upload a CSV to preview and score its customers.")
        return

    try:
        customers = pd.read_csv(uploaded_file)
    except Exception as error:
        st.error(f"Could not read this CSV: {error}")
        return

    missing_columns = sorted(set(FEATURE_COLUMNS) - set(customers.columns))
    if missing_columns:
        st.error("Required columns are missing: " + ", ".join(missing_columns))
        return
    if customers.empty:
        st.warning("The uploaded CSV contains no customer rows.")
        return

    features = prepare_features(customers).reindex(columns=FEATURE_COLUMNS)
    probabilities = model.predict_proba(features)[:, 1]
    predictions = model.predict(features).astype(int)
    scored = customers.copy()
    scored["Predicted_Churn"] = pd.Series(predictions, index=scored.index).map({0: "No", 1: "Yes"})
    scored["Churn_Probability"] = probabilities

    churned = int((predictions == 1).sum())
    metrics = st.columns(3)
    metrics[0].metric("Rows scored", f"{len(scored):,}")
    metrics[1].metric("Predicted churn", f"{churned:,}")
    metrics[2].metric("Average risk", f"{probabilities.mean():.1%}")
    st.dataframe(scored, width="stretch", hide_index=True)
    st.download_button(
        "Download scored CSV",
        scored.to_csv(index=False).encode("utf-8"),
        "churn_predictions.csv",
        "text/csv",
        type="primary",
    )


def render_analytics(
    model_name: str,
    comparison: pd.DataFrame,
    y_test: pd.Series,
    predictions,
    probabilities,
    metrics: dict[str, float],
) -> None:
    """Render held-out model metrics, ROC curve, and confusion matrix."""
    st.markdown('<div class="eyebrow">Model performance</div>', unsafe_allow_html=True)
    st.title("Model analytics")
    st.caption("All classifiers are compared on the same stratified held-out test set.")
    comparison_display = comparison.rename(
        columns={
            "accuracy": "Accuracy",
            "precision": "Precision",
            "recall": "Recall",
            "f1": "F1",
            "roc_auc": "ROC-AUC",
        }
    )
    st.dataframe(
        comparison_display.style.highlight_max(
            subset=["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
            color="#dcebdc",
        ).format({"Accuracy": "{:.3f}", "Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}", "ROC-AUC": "{:.3f}"}),
        width="stretch",
        hide_index=True,
    )
    st.subheader(f"Selected model: {model_name}")
    metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    metric_columns = st.columns(len(metric_names))
    for column, name in zip(metric_columns, metric_names):
        column.metric(name.replace("_", " ").title(), f"{metrics[name]:.3f}")

    left, right = st.columns(2)
    with left:
        st.subheader("Classification metrics")
        metric_frame = pd.DataFrame(
            {"Metric": [name.replace("_", " ").title() for name in metric_names],
             "Score": [metrics[name] for name in metric_names]}
        )
        fig = px.bar(
            metric_frame,
            x="Metric",
            y="Score",
            color="Metric",
            color_discrete_sequence=["#176b53", "#d96f52", "#99bd69", "#4f8d75", "#e1a25f"],
            text=metric_frame["Score"].map(lambda value: f"{value:.3f}"),
        )
        fig.update_layout(showlegend=False, yaxis_range=[0, 1], margin=dict(t=8, b=4))
        st.plotly_chart(fig, width="stretch")

    with right:
        st.subheader("ROC curve")
        false_positive_rate, true_positive_rate, _ = roc_curve(y_test, probabilities)
        roc_figure = go.Figure()
        roc_figure.add_trace(
            go.Scatter(x=false_positive_rate, y=true_positive_rate, mode="lines", name=f"Model ({metrics['roc_auc']:.3f})", line=dict(color="#176b53", width=3))
        )
        roc_figure.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(color="#d96f52", dash="dash"))
        )
        roc_figure.update_layout(
            xaxis_title="False positive rate",
            yaxis_title="True positive rate",
            xaxis_range=[0, 1],
            yaxis_range=[0, 1],
            margin=dict(t=8, b=4),
        )
        st.plotly_chart(roc_figure, width="stretch")

    st.subheader("Confusion matrix")
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    matrix_figure = px.imshow(
        matrix,
        text_auto=True,
        x=["Predicted no churn", "Predicted churn"],
        y=["Actual no churn", "Actual churn"],
        color_continuous_scale=[[0, "#edf3ee"], [1, "#176b53"]],
        aspect="auto",
    )
    matrix_figure.update_layout(coloraxis_showscale=False, margin=dict(t=8, b=4))
    st.plotly_chart(matrix_figure, width="stretch")


def main() -> None:
    """Load cached data/model artifacts and route to the selected app page."""
    try:
        data = get_data()
    except Exception as error:
        st.error(f"The app could not prepare its data/model: {error}")
        st.stop()

    with st.sidebar:
        st.markdown("### RETENTION / LAB")
        st.caption("Telco customer intelligence")
        model_name = st.selectbox("Prediction model", MODEL_NAMES, index=0)
        page = st.radio(
            "Workspace",
            ["Dashboard", "Predict", "Batch", "Analytics"],
            label_visibility="collapsed",
        )
        st.divider()
        st.caption(f"{len(data):,} customer records")
        st.caption("Select a model to use across prediction and batch scoring.")

    try:
        model, x_test, y_test, predictions, probabilities, metrics = get_model_artifacts(model_name)
    except Exception as error:
        st.error(f"The selected model could not be trained: {error}")
        st.stop()

    if page == "Dashboard":
        render_dashboard(data, metrics, model_name)
    elif page == "Predict":
        render_prediction(model, model_name)
    elif page == "Batch":
        render_batch(model, model_name)
    else:
        comparison = get_model_comparison()
        render_analytics(model_name, comparison, y_test, predictions, probabilities, metrics)


if __name__ == "__main__":
    main()