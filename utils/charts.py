import plotly.graph_objects as go


def fraud_probability_gauge(
    prob: float,
    threshold: float,
    title: str,
    subtitle: str = "",
) -> go.Figure:
    pct = prob * 100
    thresh_pct = threshold * 100
    is_fraud = prob >= threshold

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pct,
            number={"suffix": "%", "font": {"size": 28}},
            title={"text": f"{title}<br><span style='font-size:12px'>{subtitle}</span>"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#e74c3c" if is_fraud else "#2ecc71"},
                "steps": [
                    {"range": [0, thresh_pct], "color": "#d5f5e3"},
                    {"range": [thresh_pct, 100], "color": "#fadbd8"},
                ],
                "threshold": {
                    "line": {"color": "#2c3e50", "width": 3},
                    "thickness": 0.8,
                    "value": thresh_pct,
                },
            },
        )
    )
    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def metrics_bar_chart(metrics_df, metric: str, title: str) -> go.Figure:
    plot_df = metrics_df.copy()
    plot_df["label"] = plot_df["model"] + " (" + plot_df["sampling"] + ")"

    colors = plot_df["sampling"].map(
        {
            "No SMOTE": "#3498db",
            "SMOTE": "#9b59b6",
            "SMOTE-ENN": "#e67e22",
        }
    ).fillna("#95a5a6")

    fig = go.Figure(
        go.Bar(
            x=plot_df["label"],
            y=plot_df[metric],
            marker_color=colors,
            text=plot_df[metric].round(3),
            textposition="outside",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Model",
        yaxis_title=metric.upper(),
        yaxis_range=[0, 1.05],
        height=420,
        xaxis_tickangle=-35,
    )
    return fig


def delta_bar_chart(delta_df, metric_col: str, title: str) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=delta_df["model"],
            y=delta_df[metric_col],
            marker_color=delta_df[metric_col].apply(
                lambda v: "#2ecc71" if v >= 0 else "#e74c3c"
            ),
            text=delta_df[metric_col].round(3),
            textposition="outside",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Model",
        yaxis_title="Change (SMOTE − No SMOTE)",
        height=380,
    )
    return fig


def pr_curve_figure(precision, recall, label: str) -> go.Scatter:
    return go.Scatter(
        x=recall,
        y=precision,
        mode="lines",
        name=label,
    )
