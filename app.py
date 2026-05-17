import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from models.registry import MODEL_ENTRIES, entry_path_exists
from utils.charts import (
    delta_bar_chart,
    fraud_probability_gauge,
    metrics_bar_chart,
    pr_curve_figure,
)
from utils.evaluation import (
    build_metrics_dataframe,
    evaluate_model_on_test,
    pr_curve_data,
    smote_delta_table,
)
from utils.model_loader import load_all_models, predict_all_models
from utils.preprocessing import (
    get_processed_row_by_index,
    get_train_test_split,
    load_and_preprocess,
    load_scaler,
    prepare_row_features,
)

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
)

DEFAULT_THRESHOLD = 0.5


def init_session_state():
    defaults = {
        "transaction_row": None,
        "transaction_raw": None,
        "transaction_index": None,
        "predictions": None,
        "analysis_run": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_predictions():
    st.session_state.predictions = None
    st.session_state.analysis_run = False


def load_transaction_random(processed: pd.DataFrame):
    idx = int(np.random.randint(0, len(processed)))
    load_transaction_by_index(idx, processed)


def load_transaction_by_index(index: int, processed: pd.DataFrame):
    row, raw = get_processed_row_by_index(index, processed)
    st.session_state.transaction_row = row
    st.session_state.transaction_raw = raw
    st.session_state.transaction_index = index
    clear_predictions()


@st.cache_data(show_spinner="Loading dataset and computing metrics...")
def get_cached_data():
    processed, y, scaler = load_and_preprocess()
    X_train, X_test, y_train, y_test = get_train_test_split(processed, y)
    loaded = load_all_models()
    metrics_df = build_metrics_dataframe(loaded, X_test, y_test)
    return processed, scaler, X_test, y_test, loaded, metrics_df


@st.cache_resource(show_spinner="Loading models...")
def get_cached_models():
    return load_all_models()


def render_sidebar(processed, metrics_df, loaded):
    st.sidebar.header("Dataset")
    fraud_rate = processed["Class"].mean() * 100
    st.sidebar.metric("Fraud rate", f"{fraud_rate:.3f}%")
    st.sidebar.metric("Total transactions", f"{len(processed):,}")

    st.sidebar.header("Models")
    available = [e for e in MODEL_ENTRIES if entry_path_exists(e)]
    missing = [e for e in MODEL_ENTRIES if not entry_path_exists(e)]
    st.sidebar.success(f"{len(loaded)} model(s) loaded")
    if missing:
        with st.sidebar.expander(f"{len(missing)} missing file(s)"):
            for e in missing:
                st.caption(e["path"])

    threshold = st.sidebar.slider(
        "Fraud threshold",
        min_value=0.0,
        max_value=1.0,
        value=DEFAULT_THRESHOLD,
        step=0.05,
        help="Used for verdict labels and gauge zones in Fraud Inspector.",
    )
    return threshold


def render_comparison_tab(metrics_df, X_test, y_test, loaded, threshold):
    st.header("Model Comparison")
    st.caption("Metrics computed on the held-out test split (cached).")

    if metrics_df.empty:
        st.warning("No models loaded. Add `.pkl` files under `models/` and refresh.")
        return

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        models_filter = st.multiselect(
            "Model family",
            options=sorted(metrics_df["model"].unique()),
            default=sorted(metrics_df["model"].unique()),
        )
    with col_f2:
        sampling_filter = st.multiselect(
            "Sampling strategy",
            options=sorted(metrics_df["sampling"].unique()),
            default=sorted(metrics_df["sampling"].unique()),
        )

    filtered = metrics_df[
        metrics_df["model"].isin(models_filter) & metrics_df["sampling"].isin(sampling_filter)
    ]

    display_cols = [
        "model",
        "sampling",
        "precision",
        "recall",
        "f1",
        "auprc",
        "latency_ms",
    ]
    st.dataframe(
        filtered[display_cols].style.format(
            {
                "precision": "{:.3f}",
                "recall": "{:.3f}",
                "f1": "{:.3f}",
                "auprc": "{:.3f}",
                "latency_ms": "{:.2f}",
            },
            subset=["precision", "recall", "f1", "auprc", "latency_ms"],
        ),
        use_container_width=True,
        hide_index=True,
    )

    metric_choice = st.selectbox(
        "Chart metric",
        ["precision", "recall", "f1", "auprc"],
        index=3,
    )
    st.plotly_chart(
        metrics_bar_chart(filtered, metric_choice, f"{metric_choice.upper()} by model"),
        use_container_width=True,
    )

    st.subheader("Precision–Recall curves")
    pr_models = st.multiselect(
        "Models for PR overlay",
        options=list(loaded.keys()),
        default=list(loaded.keys())[: min(3, len(loaded))],
        format_func=lambda k: f"{loaded[k]['display']} ({loaded[k]['sampling']})",
    )
    if pr_models:
        fig = go.Figure()
        for key in pr_models:
            info = loaded[key]
            result = evaluate_model_on_test(info["model"], X_test, y_test)
            prec, rec = pr_curve_data(y_test, result["y_proba"])
            fig.add_trace(
                pr_curve_figure(prec, rec, f"{info['display']} ({info['sampling']})")
            )
        fig.update_layout(
            title="Precision–Recall (test set)",
            xaxis_title="Recall",
            yaxis_title="Precision",
            height=450,
        )
        st.plotly_chart(fig, use_container_width=True)


def render_sampling_tab(metrics_df):
    st.header("Sampling Strategy Analysis")
    st.caption("SMOTE vs No SMOTE impact per algorithm.")

    if metrics_df.empty:
        st.warning("No models loaded.")
        return

    delta_df = smote_delta_table(metrics_df)
    if delta_df.empty:
        st.info(
            "Need both 'No SMOTE' and 'SMOTE' models per family to show deltas. "
            "SMOTE-ENN entries appear in comparison when those files exist."
        )
        enn_rows = metrics_df[metrics_df["sampling"] == "SMOTE-ENN"]
        if not enn_rows.empty:
            st.subheader("SMOTE-ENN models")
            st.dataframe(enn_rows, use_container_width=True, hide_index=True)
        return

    best = metrics_df.loc[metrics_df["auprc"].idxmax()]
    st.success(
        f"Best AUPRC: **{best['model']}** ({best['sampling']}) — AUPRC = {best['auprc']:.3f}"
    )

    for metric_col, title in [
        ("delta_recall", "Δ Recall (SMOTE − No SMOTE)"),
        ("delta_precision", "Δ Precision (SMOTE − No SMOTE)"),
        ("delta_f1", "Δ F1 (SMOTE − No SMOTE)"),
        ("delta_auprc", "Δ AUPRC (SMOTE − No SMOTE)"),
    ]:
        st.plotly_chart(delta_bar_chart(delta_df, metric_col, title), use_container_width=True)

    enn_rows = metrics_df[metrics_df["sampling"] == "SMOTE-ENN"]
    if enn_rows.empty:
        st.info(
            "SMOTE-ENN models are optional. Add `*_smote_enn.pkl` files to the registry to compare."
        )
    else:
        st.subheader("SMOTE-ENN results")
        st.dataframe(enn_rows, use_container_width=True, hide_index=True)


def verdict_label(prob: float, threshold: float) -> str:
    return "FRAUD" if prob >= threshold else "LEGITIMATE"


def render_fraud_inspector(processed, scaler, loaded, threshold):
    st.header("Transaction Fraud Inspector")
    st.caption(
        "Load a transaction as input, then run detection on demand. "
        "Predictions are not executed until you click **Run fraud detection**."
    )

    st.subheader("1. Transaction input")
    col_a, col_b, col_c = st.columns([1, 1, 1])
    with col_a:
        if st.button("Load random transaction", type="primary", use_container_width=True):
            load_transaction_random(processed)
            st.rerun()
    with col_b:
        max_idx = len(processed) - 1
        row_idx = st.number_input("Row index", min_value=0, max_value=max_idx, value=0, step=1)
    with col_c:
        if st.button("Load this row", use_container_width=True):
            load_transaction_by_index(int(row_idx), processed)
            st.rerun()

    if st.session_state.transaction_row is None:
        st.info("Load a transaction to begin. No models will run until you analyze.")
        return

    row = st.session_state.transaction_row
    raw = st.session_state.transaction_raw
    idx = st.session_state.transaction_index
    actual_class = int(row["Class"])
    actual_label = "Fraud" if actual_class == 1 else "Legitimate"
    badge_color = "red" if actual_class == 1 else "green"

    st.markdown(
        f"**Row {idx}** — Ground truth: "
        f":{badge_color}[{actual_label}] "
        f"(Class = {actual_class})"
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Amount", f"${raw['Amount']:.2f}")
    c2.metric("Time", f"{raw['Time']:.0f}")
    c3.metric("Class", actual_label)

    with st.expander("Transaction features"):
        feature_cols = [c for c in row.index if c != "Class"]
        st.dataframe(row[feature_cols].to_frame().T, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("2. Run analysis")

    if not loaded:
        st.error("No models available. Add model files under `models/`.")
        return

    run_clicked = st.button(
        "Run fraud detection",
        type="primary",
        disabled=False,
        use_container_width=False,
    )

    if run_clicked:
        features = prepare_row_features(raw, scaler)
        st.session_state.predictions = predict_all_models(loaded, features)
        st.session_state.analysis_run = True
        st.rerun()

    if not st.session_state.analysis_run or not st.session_state.predictions:
        st.warning("Click **Run fraud detection** to score this transaction with all models.")
        return

    predictions = st.session_state.predictions
    st.divider()
    st.subheader("3. Results")

    fraud_votes = sum(1 for p in predictions if p["fraud_probability"] >= threshold)
    total = len(predictions)
    majority_fraud = fraud_votes > total / 2
    consensus = "Fraud" if majority_fraud else "Legitimate"
    consensus_color = "red" if majority_fraud else "green"
    st.markdown(
        f"### Consensus: :{consensus_color}[{consensus}] "
        f"({fraud_votes}/{total} models above threshold)"
    )

    n_cols = min(3, len(predictions))
    for i in range(0, len(predictions), n_cols):
        cols = st.columns(n_cols)
        for j, col in enumerate(cols):
            if i + j >= len(predictions):
                break
            pred = predictions[i + j]
            prob = pred["fraud_probability"]
            verdict = verdict_label(prob, threshold)
            v_color = "red" if verdict == "FRAUD" else "green"
            title = pred["display"]
            subtitle = pred["sampling"]
            with col:
                st.plotly_chart(
                    fraud_probability_gauge(prob, threshold, title, subtitle),
                    use_container_width=True,
                    key=f"gauge_{pred['id']}",
                )
                st.markdown(f"**Verdict:** :{v_color}[{verdict}]")
                st.caption(f"Latency: {pred['latency_ms']:.2f} ms")

    result_df = pd.DataFrame(predictions)
    result_df["verdict"] = result_df["fraud_probability"].apply(
        lambda p: verdict_label(p, threshold)
    )
    result_df["fraud_probability"] = result_df["fraud_probability"].round(4)
    result_df["latency_ms"] = result_df["latency_ms"].round(2)

    st.dataframe(
        result_df[
            ["display", "sampling", "fraud_probability", "verdict", "latency_ms"]
        ].rename(
            columns={
                "display": "Model",
                "sampling": "Sampling",
                "fraud_probability": "P(fraud)",
                "latency_ms": "Latency (ms)",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    verdicts = result_df["verdict"].unique()
    if len(verdicts) > 1:
        st.warning("Models disagree on this transaction. Review gauges and threshold.")


def main():
    init_session_state()
    st.title("Credit Card Fraud Detection Dashboard")
    st.markdown(
        "Compare **Random Forest**, **XGBoost**, and **MLP** with and without **SMOTE** "
        "on the credit card fraud dataset."
    )

    processed, scaler, X_test, y_test, loaded_cached, metrics_df = get_cached_data()
    loaded = get_cached_models()
    if not loaded:
        loaded = loaded_cached

    threshold = render_sidebar(processed, metrics_df, loaded)

    tab_compare, tab_sampling, tab_inspector = st.tabs(
        [
            "Model Comparison",
            "Sampling Analysis",
            "Fraud Inspector",
        ]
    )

    with tab_compare:
        render_comparison_tab(metrics_df, X_test, y_test, loaded, threshold)

    with tab_sampling:
        render_sampling_tab(metrics_df)

    with tab_inspector:
        render_fraud_inspector(processed, scaler, loaded, threshold)


if __name__ == "__main__":
    main()
