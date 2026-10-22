import json
import warnings

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Risk Alert Classifier", page_icon="🚨", layout="wide")

MODEL_PATH, META_PATH, DATA_PATH = "model/risk_model.joblib", "model/meta.json", "Risk_Alert_Classifier_Dataset_4600.csv"


def add_days_since(df, ref_date):
    df = df.copy()
    if "days_since_last_txn" not in df.columns and "last_transaction_date" in df.columns:
        df["days_since_last_txn"] = (pd.Timestamp(ref_date) - pd.to_datetime(df["last_transaction_date"])).dt.days
    return df


@st.cache_resource(show_spinner="Training compatible model...")
def load_model():
    """Train a fresh pipeline using the current environment."""

    with open(META_PATH, encoding="utf-8") as f:
        meta = json.load(f)

    df = pd.read_csv(DATA_PATH)
    df = add_days_since(df, meta["ref_date"])

    num_cols = meta["num_cols"]
    cat_cols = meta["cat_cols"]

    X = df[num_cols + cat_cols]
    y = df["risk_status"]

    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    preprocessor = ColumnTransformer([
        (
            "num",
            Pipeline([
                ("scaler", StandardScaler()),
                ("knn_imputer", KNNImputer(n_neighbors=5))
            ]),
            num_cols
        ),
        (
            "cat",
            Pipeline([
                (
                    "imputer",
                    SimpleImputer(
                        strategy="constant",
                        fill_value="Unknown"
                    )
                ),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False
                    )
                )
            ]),
            cat_cols
        )
    ])

    pipe = Pipeline([
        ("prep", preprocessor),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ])

    pipe.fit(X_train, y_train)

    return pipe, meta


pipe, meta = load_model()
thr_default = float(meta["threshold"])

st.title("🚨 Risk Alert Classifier")
st.caption("Early-warning system for a digital bank: predicts whether a customer is **High Risk** (likely to default / commit fraud). "
           f"Model: {meta['final_model']}")

tab1, tab2, tab3 = st.tabs(["Single customer", "Batch prediction (CSV)", "Model results"])

with tab1:
    threshold = st.slider("Decision threshold (lower = catch more risky customers, more false alarms)", 0.05, 0.95, thr_default, 0.01)
    st.write("Enter the customer details:")
    c1, c2, c3 = st.columns(3)
    values = {}
    for i, col in enumerate(meta["num_cols"]):
        lo, med, hi = meta["num_ranges"][col]
        target = [c1, c2, c3][i % 3]
        is_int = float(lo).is_integer() and float(hi).is_integer() and float(med).is_integer() or col in (
            "age", "missed_payments_12m", "monthly_transaction_count", "cash_advance_count_6m", "complaints_last_6m",
            "failed_login_attempts_3m", "account_tenure_months", "days_since_last_txn", "annual_income_inr",
            "monthly_spend_inr", "credit_score", "debt_balance_inr")
        label = col.replace("_", " ").title()
        if is_int:
            values[col] = target.number_input(label, min_value=int(lo), max_value=int(hi) * 2,
                                              value=int(med), step=1)
        else:
            values[col] = target.number_input(label, min_value=float(lo), max_value=float(hi) * 2, value=float(round(med, 3)), step=0.01)
    d1, d2, d3 = st.columns(3)
    for col, target in zip(meta["cat_cols"], [d1, d2, d3]):
        values[col] = target.selectbox(col.replace("_", " ").title(), meta["cat_options"][col])

    if st.button("Predict risk", type="primary"):
        row = pd.DataFrame([values])[meta["num_cols"] + meta["cat_cols"]]
        prob = float(pipe.predict_proba(row)[0, 1])
        if prob >= threshold:
            st.error(f"🚨 HIGH RISK – probability {prob:.1%} (threshold {threshold:.0%}). Recommend manual review.")
        else:
            st.success(f"✅ Low risk – probability of high risk {prob:.1%} (threshold {threshold:.0%}).")
        st.progress(min(max(prob, 0.0), 1.0))

with tab2:
    st.write("Upload a CSV with the same columns as the training data (`customer_id`, `risk_status` are optional). "
             "`last_transaction_date` or `days_since_last_txn` is needed.")
    up = st.file_uploader("CSV file", type=["csv"])
    thr_b = st.slider("Decision threshold ", 0.05, 0.95, thr_default, 0.01, key="thr_batch")
    if up is not None:
        data = add_days_since(pd.read_csv(up), meta["ref_date"])
        needed = meta["num_cols"] + meta["cat_cols"]
        missing = [c for c in needed if c not in data.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            prob = pipe.predict_proba(data[needed])[:, 1]
            out = data.copy()
            out["risk_probability"] = prob.round(4)
            out["predicted_risk"] = (prob >= thr_b).astype(int)
            st.metric("Customers flagged High Risk", f"{int(out['predicted_risk'].sum())} of {len(out)}")
            st.dataframe(out.sort_values("risk_probability", ascending=False), width="stretch")
            st.download_button("Download predictions", out.to_csv(index=False).encode(), "predictions.csv", "text/csv")

with tab3:
    st.subheader("All models – test-set results")
    try:
        res = pd.read_csv("model/all_model_results.csv")
        show = res[["Model", "Group", "Test Acc", "Precision", "Recall", "F1", "F2", "AUC", "FP (Type-I)", "FN (Type-II)"]].round(4)
        st.dataframe(show, width="stretch")
    except FileNotFoundError:
        st.info("Run the notebook first to create model/all_model_results.csv")
    for img, cap in [("images/11_roc_all_models.png", "ROC curves – all models"),
                     ("images/06_balancing_comparison.png", "Before vs after balancing"),
                     ("images/10_tuned_vs_untuned.png", "Untuned vs tuned models")]:
        try:
            st.image(img, caption=cap)
        except Exception:
            pass
