from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


st.set_page_config(page_title="Churn & Revenue Insights", layout="wide")

st.write("Test edit")

NUMERIC_USAGE_FEATURES = [
    "weekly_active_days_avg",
    "queries_run",
    "seats_used",
    "support_tickets",
    "nps",
    "discount_pct",
]


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator is None or pd.isna(denominator) or denominator == 0:
        return np.nan
    return float(numerator) / float(denominator)


def fmt_currency(value: float) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"${value:,.2f}"


def fmt_pct(value: float) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value * 100:.2f}%"


def parse_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    mapped = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False})
    )
    return mapped.fillna(False)


@st.cache_data(show_spinner=False)
def load_data(data_dir: str = ".") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = Path(data_dir)
    customers = pd.read_csv(base / "customers.csv")
    subscriptions = pd.read_csv(base / "subscriptions.csv")
    usage = pd.read_csv(base / "usage_metrics.csv")
    invoices = pd.read_csv(base / "invoices.csv")

    customers["signup_date"] = pd.to_datetime(customers["signup_date"], errors="coerce")
    subscriptions["start_date"] = pd.to_datetime(subscriptions["start_date"], errors="coerce")
    subscriptions["end_date"] = pd.to_datetime(subscriptions["end_date"], errors="coerce")
    subscriptions["is_churned"] = parse_bool(subscriptions["is_churned"])

    usage["month"] = pd.to_datetime(usage["month"].astype(str) + "-01", errors="coerce").dt.to_period("M").dt.to_timestamp()
    invoices["month"] = (
        pd.to_datetime(invoices["month"].astype(str) + "-01", errors="coerce").dt.to_period("M").dt.to_timestamp()
    )

    for col in ["price_usd_month"]:
        customers[col] = pd.to_numeric(customers[col], errors="coerce")
    for col in ["price_usd_month"]:
        subscriptions[col] = pd.to_numeric(subscriptions[col], errors="coerce")
    for col in NUMERIC_USAGE_FEATURES:
        usage[col] = pd.to_numeric(usage[col], errors="coerce")
    for col in ["mrr_usd", "add_on_usd"]:
        invoices[col] = pd.to_numeric(invoices[col], errors="coerce")

    return customers, subscriptions, usage, invoices


@st.cache_data(show_spinner=False)
def prepare_model_table(
    customers: pd.DataFrame,
    subscriptions: pd.DataFrame,
    usage: pd.DataFrame,
    invoices: pd.DataFrame,
) -> pd.DataFrame:
    subs = subscriptions.sort_values(["customer_id", "start_date", "end_date"]).drop_duplicates("customer_id", keep="last")

    customer_month = usage.merge(invoices, on=["customer_id", "month"], how="outer")
    customer_month = customer_month.merge(customers, on="customer_id", how="left")

    sub_cols = [
        "customer_id",
        "plan",
        "price_usd_month",
        "start_date",
        "end_date",
        "is_churned",
        "churn_reason",
    ]
    sub_join = subs[sub_cols].rename(
        columns={
            "plan": "subscription_plan",
            "price_usd_month": "subscription_price_usd_month",
        }
    )
    df = customer_month.merge(sub_join, on="customer_id", how="left")

    df["month"] = pd.to_datetime(df["month"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    df = df.dropna(subset=["customer_id", "month"]).copy()
    df = df.sort_values(["customer_id", "month"]).drop_duplicates(["customer_id", "month"], keep="last")

    df["month_start"] = df["month"]
    df["month_end"] = df["month"] + pd.offsets.MonthEnd(0)

    df["active"] = (df["start_date"].isna() | (df["start_date"] <= df["month_end"])) & (
        df["end_date"].isna() | (df["end_date"] >= df["month_end"])
    )
    df["active_start_month"] = (df["start_date"].isna() | (df["start_date"] <= df["month_start"])) & (
        df["end_date"].isna() | (df["end_date"] >= df["month_start"])
    )
    df["churned_in_month"] = df["end_date"].between(df["month_start"], df["month_end"], inclusive="both")

    signup_period = df["signup_date"].dt.to_period("M")
    month_period = df["month"].dt.to_period("M")
    tenure = month_period - signup_period
    df["tenure_months"] = tenure.apply(lambda x: x.n if pd.notna(x) else np.nan)
    df["tenure_months"] = pd.to_numeric(df["tenure_months"], errors="coerce").clip(lower=0)

    df["price_usd_month"] = pd.to_numeric(df["price_usd_month"], errors="coerce")
    df["subscription_price_usd_month"] = pd.to_numeric(df["subscription_price_usd_month"], errors="coerce")
    df["effective_price_usd_month"] = df["price_usd_month"].fillna(df["subscription_price_usd_month"])
    df["mrr_usd"] = pd.to_numeric(df["mrr_usd"], errors="coerce").fillna(0.0)
    df["add_on_usd"] = pd.to_numeric(df["add_on_usd"], errors="coerce").fillna(0.0)
    df["total_revenue_usd"] = df["mrr_usd"] + df["add_on_usd"]

    for col in NUMERIC_USAGE_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["region", "industry", "company_size", "segment", "acquisition_channel", "plan"]:
        df[col] = df[col].fillna("Unknown").astype(str)
    df["churn_reason"] = df["churn_reason"].fillna("Unknown").astype(str)

    return df


def render_sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")
    months = sorted(df["month"].dropna().unique())
    if not months:
        return df.iloc[0:0].copy()

    month_options = [pd.Timestamp(m).date() for m in months]
    start_date, end_date = st.sidebar.select_slider(
        "Date range (month)",
        options=month_options,
        value=(month_options[0], month_options[-1]),
        format_func=lambda d: d.strftime("%Y-%m"),
    )
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    filtered = df[(df["month"] >= start_ts) & (df["month"] <= end_ts)].copy()

    dimensions = {
        "region": "Region",
        "segment": "Segment",
        "plan": "Plan",
        "industry": "Industry",
        "company_size": "Company size",
        "acquisition_channel": "Acquisition channel",
    }

    for col, label in dimensions.items():
        options = sorted(filtered[col].dropna().astype(str).unique().tolist())
        selected = st.sidebar.multiselect(label, options, default=options)
        if len(selected) == 0:
            return filtered.iloc[0:0].copy()
        filtered = filtered[filtered[col].astype(str).isin(selected)]

    drill_options = ["All"] + sorted(filtered["segment"].dropna().astype(str).unique().tolist())
    drill_segment = st.sidebar.selectbox("Drill-down segment", drill_options, index=0)
    if drill_segment != "All":
        filtered = filtered[filtered["segment"] == drill_segment]

    return filtered


def compute_monthly_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["month", "mrr_usd", "active_customers", "active_start", "churned", "churn_rate", "arpu"])

    monthly = (
        df.groupby("month", as_index=False)
        .agg(
            mrr_usd=("mrr_usd", "sum"),
            active_customers=("active", "sum"),
            active_start=("active_start_month", "sum"),
            churned=("churned_in_month", "sum"),
        )
        .sort_values("month")
    )
    monthly["churn_rate"] = np.where(monthly["active_start"] > 0, monthly["churned"] / monthly["active_start"], np.nan)
    monthly["arpu"] = np.where(monthly["active_customers"] > 0, monthly["mrr_usd"] / monthly["active_customers"], np.nan)
    return monthly


def compute_kpis(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {
            "last_month": None,
            "current_mrr": np.nan,
            "churn_rate": np.nan,
            "net_mrr_change": np.nan,
            "arpu": np.nan,
            "active_customers": 0,
        }

    last_month = df["month"].max()
    prev_month = last_month - pd.DateOffset(months=1)
    last_slice = df[df["month"] == last_month]
    prev_slice = df[df["month"] == prev_month]

    current_mrr = float(last_slice["mrr_usd"].sum())
    prev_mrr = float(prev_slice["mrr_usd"].sum()) if not prev_slice.empty else np.nan
    net_mrr_change = current_mrr - prev_mrr if pd.notna(prev_mrr) else np.nan

    churned = float(last_slice["churned_in_month"].sum())
    active_start = float(last_slice["active_start_month"].sum())
    churn_rate = safe_divide(churned, active_start)

    active_customers = int(last_slice["active"].sum())
    arpu = safe_divide(current_mrr, active_customers)

    return {
        "last_month": last_month,
        "current_mrr": current_mrr,
        "churn_rate": churn_rate,
        "net_mrr_change": net_mrr_change,
        "arpu": arpu,
        "active_customers": active_customers,
    }


def churn_by_dimension(df: pd.DataFrame, dimension: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[dimension, "churned", "active_start", "churn_rate"])
    out = (
        df.groupby(dimension, as_index=False)
        .agg(churned=("churned_in_month", "sum"), active_start=("active_start_month", "sum"))
        .sort_values("churned", ascending=False)
    )
    out["churn_rate"] = np.where(out["active_start"] > 0, out["churned"] / out["active_start"], np.nan)
    return out.sort_values("churn_rate", ascending=False)


def build_retention_matrix(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    cohort = df.dropna(subset=["signup_date", "month"]).copy()
    cohort["cohort_month"] = cohort["signup_date"].dt.to_period("M").dt.to_timestamp()
    cohort["age_month"] = (
        (cohort["month"].dt.year - cohort["cohort_month"].dt.year) * 12
        + (cohort["month"].dt.month - cohort["cohort_month"].dt.month)
    )
    cohort = cohort[cohort["age_month"] >= 0]
    if cohort.empty:
        return pd.DataFrame()

    cohort_size = cohort.groupby("cohort_month")["customer_id"].nunique().rename("cohort_size")
    active_counts = (
        cohort[cohort["active"]]
        .groupby(["cohort_month", "age_month"])["customer_id"]
        .nunique()
        .rename("active_customers")
        .reset_index()
    )
    active_counts = active_counts.merge(cohort_size, on="cohort_month", how="left")
    active_counts["retention_pct"] = np.where(
        active_counts["cohort_size"] > 0,
        (active_counts["active_customers"] / active_counts["cohort_size"]) * 100,
        np.nan,
    )
    matrix = active_counts.pivot(index="cohort_month", columns="age_month", values="retention_pct").sort_index()
    return matrix


def segment_summary(df: pd.DataFrame, group_col: str, last_month: pd.Timestamp) -> pd.DataFrame:
    view = df[df["month"] == last_month].copy()
    if view.empty:
        return pd.DataFrame(columns=[group_col, "mrr_usd", "active_customers", "arpu", "churn_rate"])
    summary = (
        view.groupby(group_col, as_index=False)
        .agg(
            mrr_usd=("mrr_usd", "sum"),
            active_customers=("active", "sum"),
            churned=("churned_in_month", "sum"),
            active_start=("active_start_month", "sum"),
        )
        .sort_values("mrr_usd", ascending=False)
    )
    summary["arpu"] = np.where(summary["active_customers"] > 0, summary["mrr_usd"] / summary["active_customers"], np.nan)
    summary["churn_rate"] = np.where(summary["active_start"] > 0, summary["churned"] / summary["active_start"], np.nan)
    return summary


def pareto_customers(df: pd.DataFrame, last_month: pd.Timestamp) -> tuple[pd.DataFrame, int, float]:
    view = df[df["month"] == last_month].copy()
    if view.empty:
        return pd.DataFrame(columns=["customer_id", "mrr_usd", "cum_mrr_share"]), 0, np.nan

    pareto = view.groupby("customer_id", as_index=False)["mrr_usd"].sum().sort_values("mrr_usd", ascending=False)
    total = pareto["mrr_usd"].sum()
    pareto["cum_mrr_share"] = np.where(total > 0, pareto["mrr_usd"].cumsum() / total, np.nan)

    n_customers = len(pareto)
    top_n = int(np.ceil(0.2 * n_customers)) if n_customers > 0 else 0
    top_n = max(top_n, 1) if n_customers > 0 else 0
    top_share = float(pareto.iloc[top_n - 1]["cum_mrr_share"]) if top_n > 0 else np.nan
    return pareto, top_n, top_share


def train_churn_model(df: pd.DataFrame) -> dict[str, Any] | None:
    feature_cols_num = [
        "weekly_active_days_avg",
        "queries_run",
        "seats_used",
        "support_tickets",
        "nps",
        "discount_pct",
        "effective_price_usd_month",
        "tenure_months",
    ]
    feature_cols_cat = ["segment", "plan", "region"]

    model_df = df[df["active_start_month"]].copy()
    model_df = model_df.dropna(subset=["churned_in_month"])
    if model_df.empty:
        return None

    for col in feature_cols_num:
        model_df[col] = pd.to_numeric(model_df[col], errors="coerce")
    for col in feature_cols_cat:
        model_df[col] = model_df[col].astype(str).fillna("Unknown")

    y = model_df["churned_in_month"].astype(int)
    if len(model_df) < 80 or y.nunique() < 2:
        return None

    X = model_df[feature_cols_num + feature_cols_cat].copy()

    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                feature_cols_num,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                feature_cols_cat,
            ),
        ]
    )

    model = LogisticRegression(max_iter=1200, class_weight="balanced", solver="liblinear")
    pipe = Pipeline(steps=[("preprocess", preprocess), ("model", model)])
    pipe.fit(X, y)

    prob = pipe.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, prob) if y.nunique() == 2 else np.nan

    feature_names = pipe.named_steps["preprocess"].get_feature_names_out()
    coefficients = pipe.named_steps["model"].coef_[0]
    feature_importance = pd.DataFrame({"feature": feature_names, "coefficient": coefficients})
    feature_importance["abs_coefficient"] = feature_importance["coefficient"].abs()
    feature_importance["direction"] = np.where(
        feature_importance["coefficient"] >= 0, "Higher churn propensity", "Lower churn propensity"
    )
    feature_importance = feature_importance.sort_values("abs_coefficient", ascending=False)

    baseline = {}
    for col in feature_cols_num:
        baseline[col] = float(model_df[col].median()) if model_df[col].notna().any() else 0.0
    for col in feature_cols_cat:
        baseline[col] = str(model_df[col].mode(dropna=True).iloc[0]) if model_df[col].notna().any() else "Unknown"

    return {
        "pipeline": pipe,
        "features_num": feature_cols_num,
        "features_cat": feature_cols_cat,
        "feature_importance": feature_importance,
        "auc": auc,
        "sample_size": len(model_df),
        "churn_prevalence": float(y.mean()),
        "baseline": baseline,
    }


def compute_mrr_correlations(df: pd.DataFrame) -> pd.DataFrame:
    candidates = [
        "weekly_active_days_avg",
        "queries_run",
        "seats_used",
        "support_tickets",
        "nps",
        "discount_pct",
        "effective_price_usd_month",
        "tenure_months",
        "add_on_usd",
    ]
    rows = []
    for col in candidates:
        tmp = df[[col, "mrr_usd"]].dropna()
        if len(tmp) < 3 or tmp[col].nunique() < 2:
            continue
        pearson = tmp[col].corr(tmp["mrr_usd"], method="pearson")
        spearman = tmp[col].corr(tmp["mrr_usd"], method="spearman")
        rows.append({"feature": col, "pearson_corr": pearson, "spearman_corr": spearman, "n": len(tmp)})
    if not rows:
        return pd.DataFrame(columns=["feature", "pearson_corr", "spearman_corr", "n"])
    corr_df = pd.DataFrame(rows)
    corr_df["abs_spearman"] = corr_df["spearman_corr"].abs()
    return corr_df.sort_values("abs_spearman", ascending=False).drop(columns=["abs_spearman"])


def build_download_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    monthly = compute_monthly_metrics(df).copy()
    monthly["aggregate_type"] = "monthly"
    monthly["group"] = monthly["month"].dt.strftime("%Y-%m")

    pieces = [monthly[["aggregate_type", "group", "mrr_usd", "active_customers", "churned", "churn_rate", "arpu"]]]
    for dim in ["segment", "plan", "region"]:
        summary = (
            df.groupby(dim, as_index=False)
            .agg(
                mrr_usd=("mrr_usd", "sum"),
                active_customers=("active", "sum"),
                churned=("churned_in_month", "sum"),
                active_start=("active_start_month", "sum"),
            )
            .rename(columns={dim: "group"})
        )
        summary["churn_rate"] = np.where(summary["active_start"] > 0, summary["churned"] / summary["active_start"], np.nan)
        summary["arpu"] = np.where(summary["active_customers"] > 0, summary["mrr_usd"] / summary["active_customers"], np.nan)
        summary["aggregate_type"] = f"by_{dim}"
        pieces.append(summary[["aggregate_type", "group", "mrr_usd", "active_customers", "churned", "churn_rate", "arpu"]])
    return pd.concat(pieces, ignore_index=True)


def render_kpi_row(kpis: dict[str, Any]) -> None:
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric(
            "Current MRR",
            fmt_currency(kpis["current_mrr"]),
            help="Sum of mrr_usd for the latest selected month.",
        )
    with c2:
        st.metric(
            "Churn rate (last month)",
            fmt_pct(kpis["churn_rate"]),
            help="churned_in_month / active at start of month, in latest selected month.",
        )
    with c3:
        st.metric(
            "Net MRR change",
            fmt_currency(kpis["net_mrr_change"]),
            help="Current MRR minus previous month MRR.",
        )
    with c4:
        st.metric(
            "ARPU",
            fmt_currency(kpis["arpu"]),
            help="Current month MRR / active customers in latest selected month.",
        )
    with c5:
        st.metric(
            "Active customers",
            f"{kpis['active_customers']:,}",
            help="Count of active customers in the latest selected month.",
        )


def render_trends_and_cohort(df: pd.DataFrame) -> None:
    st.subheader("Churn Trends & Cohorts")
    monthly = compute_monthly_metrics(df)

    left, right = st.columns([1.1, 1.0])
    with left:
        if monthly.empty:
            st.info("No data available for churn trend.")
        else:
            fig = px.line(
                monthly,
                x="month",
                y="churn_rate",
                markers=True,
                title="Monthly Churn Rate",
                labels={"month": "Month", "churn_rate": "Churn rate"},
            )
            fig.update_yaxes(tickformat=".1%")
            fig.update_layout(legend_title_text="")
            st.plotly_chart(fig, use_container_width=True)

    with right:
        matrix = build_retention_matrix(df)
        if matrix.empty:
            st.info("No data available for cohort retention heatmap.")
        else:
            fig = px.imshow(
                matrix,
                aspect="auto",
                color_continuous_scale="Blues",
                labels={"x": "Months since signup", "y": "Signup cohort", "color": "Retention %"},
            )
            fig.update_layout(title="Cohort Retention Heatmap")
            st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    for col, dim, title in [
        (c1, "plan", "Churn Rate by Plan"),
        (c2, "segment", "Churn Rate by Segment"),
        (c3, "region", "Churn Rate by Region"),
    ]:
        with col:
            by_dim = churn_by_dimension(df, dim)
            if by_dim.empty:
                st.info(f"No data for {dim}.")
            else:
                fig = px.bar(
                    by_dim.sort_values("churn_rate", ascending=False),
                    x=dim,
                    y="churn_rate",
                    title=title,
                    labels={dim: dim.capitalize(), "churn_rate": "Churn rate"},
                )
                fig.update_yaxes(tickformat=".1%")
                st.plotly_chart(fig, use_container_width=True)

    churn_reasons = df[df["churned_in_month"]].copy()
    if not churn_reasons.empty:
        reason_counts = churn_reasons["churn_reason"].value_counts().rename_axis("churn_reason").reset_index(name="count")
        fig = px.bar(reason_counts, x="churn_reason", y="count", title="Churn Reasons (selected range)")
        st.plotly_chart(fig, use_container_width=True)


def render_segment_section(df: pd.DataFrame, last_month: pd.Timestamp) -> None:
    st.subheader("Customer Segments")
    group_col = st.selectbox("Compare by", ["segment", "plan", "region"], index=0)
    summary = segment_summary(df, group_col, last_month)
    if summary.empty:
        st.info("No segment summary for selected filters.")
        return

    display = summary[[group_col, "mrr_usd", "active_customers", "arpu", "churn_rate"]].copy()
    display = display.rename(columns={group_col: group_col.capitalize(), "mrr_usd": "MRR (USD)", "arpu": "ARPU"})
    st.dataframe(display, use_container_width=True)

    plot_df = summary[[group_col, "mrr_usd", "arpu", "churn_rate"]].melt(
        id_vars=[group_col], value_vars=["mrr_usd", "arpu", "churn_rate"], var_name="metric", value_name="value"
    )
    fig = px.bar(
        plot_df,
        x=group_col,
        y="value",
        color="metric",
        barmode="group",
        title=f"{group_col.capitalize()} comparison in {last_month:%Y-%m}",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_revenue_drivers(df: pd.DataFrame, last_month: pd.Timestamp) -> None:
    st.subheader("Revenue Drivers")
    dim = st.selectbox("MRR breakdown dimension", ["plan", "region", "segment"], index=0)

    trend = df.groupby(["month", dim], as_index=False)["mrr_usd"].sum().sort_values("month")
    if trend.empty:
        st.info("No revenue trend data available.")
    else:
        fig = px.area(
            trend,
            x="month",
            y="mrr_usd",
            color=dim,
            title=f"MRR over time by {dim}",
            labels={"month": "Month", "mrr_usd": "MRR (USD)"},
        )
        st.plotly_chart(fig, use_container_width=True)

    pareto, top_n, top_share = pareto_customers(df, last_month)
    if pareto.empty:
        st.info("No Pareto data for selected range.")
    else:
        pareto_plot = pareto.head(50).copy()
        pareto_plot["rank"] = np.arange(1, len(pareto_plot) + 1)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=pareto_plot["rank"], y=pareto_plot["mrr_usd"], name="MRR"))
        fig.add_trace(
            go.Scatter(
                x=pareto_plot["rank"],
                y=pareto_plot["cum_mrr_share"] * 100,
                mode="lines+markers",
                name="Cumulative MRR %",
                yaxis="y2",
            )
        )
        fig.update_layout(
            title=f"Pareto Contribution ({last_month:%Y-%m}, top 50 customers)",
            xaxis_title="Customer rank",
            yaxis_title="MRR (USD)",
            yaxis2=dict(title="Cumulative %", overlaying="y", side="right", range=[0, 100]),
            legend=dict(orientation="h"),
        )
        st.plotly_chart(fig, use_container_width=True)
        if pd.notna(top_share):
            st.caption(f"Top 20% customers ({top_n} customers) contribute {top_share * 100:.2f}% of selected-month MRR.")


def render_model_and_correlations(df: pd.DataFrame) -> None:
    st.subheader("Churn & Revenue Driver Modeling")

    model_info = train_churn_model(df)
    if model_info is None:
        st.warning("Model not trained: need at least 80 active-start records with both churn classes in selected filters.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Training sample size", f"{model_info['sample_size']:,}")
        c2.metric("Observed churn prevalence", fmt_pct(model_info["churn_prevalence"]))
        c3.metric("Train ROC AUC", f"{model_info['auc']:.3f}" if pd.notna(model_info["auc"]) else "N/A")

        top_features = model_info["feature_importance"].head(20).copy()
        fig = px.bar(
            top_features.sort_values("abs_coefficient"),
            x="abs_coefficient",
            y="feature",
            color="direction",
            orientation="h",
            title="Top Churn Propensity Drivers (Logistic Regression)",
            labels={"abs_coefficient": "Absolute coefficient", "feature": "Feature"},
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("What-if churn propensity simulation", expanded=False):
            baseline = model_info["baseline"].copy()
            numeric_cols = model_info["features_num"]
            categorical_cols = model_info["features_cat"]

            def slider_bounds(col: str) -> tuple[float, float]:
                col_data = pd.to_numeric(df[col], errors="coerce").dropna()
                if col_data.empty:
                    return 0.0, 1.0
                low = float(col_data.quantile(0.05))
                high = float(col_data.quantile(0.95))
                if low == high:
                    high = low + 1.0
                return low, high

            w_low, w_high = slider_bounds("weekly_active_days_avg")
            nps_low, nps_high = slider_bounds("nps")
            d_low, d_high = slider_bounds("discount_pct")

            scenario = baseline.copy()
            scenario["weekly_active_days_avg"] = st.slider(
                "weekly_active_days_avg",
                min_value=float(w_low),
                max_value=float(w_high),
                value=float(np.clip(baseline["weekly_active_days_avg"], w_low, w_high)),
            )
            scenario["nps"] = st.slider(
                "nps",
                min_value=float(nps_low),
                max_value=float(nps_high),
                value=float(np.clip(baseline["nps"], nps_low, nps_high)),
            )
            scenario["discount_pct"] = st.slider(
                "discount_pct",
                min_value=float(d_low),
                max_value=float(d_high),
                value=float(np.clip(baseline["discount_pct"], d_low, d_high)),
                step=0.01,
            )

            for col in numeric_cols:
                scenario.setdefault(col, baseline[col])
            for col in categorical_cols:
                scenario.setdefault(col, baseline[col])

            base_input = pd.DataFrame([baseline])[numeric_cols + categorical_cols]
            scen_input = pd.DataFrame([scenario])[numeric_cols + categorical_cols]
            pipe = model_info["pipeline"]
            base_score = float(pipe.predict_proba(base_input)[0, 1])
            scen_score = float(pipe.predict_proba(scen_input)[0, 1])

            wc1, wc2 = st.columns(2)
            wc1.metric("Baseline churn propensity", fmt_pct(base_score))
            wc2.metric("Scenario churn propensity", fmt_pct(scen_score), delta=fmt_pct(scen_score - base_score))

    corr = compute_mrr_correlations(df)
    if corr.empty:
        st.info("Insufficient data for MRR correlation analysis.")
    else:
        st.markdown("**MRR correlation with numeric drivers**")
        st.dataframe(corr, use_container_width=True)
        fig = px.bar(
            corr.sort_values("spearman_corr"),
            x="spearman_corr",
            y="feature",
            orientation="h",
            title="Spearman Correlation vs MRR",
            labels={"spearman_corr": "Spearman correlation", "feature": "Feature"},
        )
        st.plotly_chart(fig, use_container_width=True)


def render_ltv(df: pd.DataFrame, kpis: dict[str, Any], last_month: pd.Timestamp) -> None:
    st.subheader("LTV Estimate")
    month_view = df[df["month"] == last_month]
    active_view = month_view[month_view["active"]]
    avg_tenure = float(active_view["tenure_months"].mean()) if not active_view.empty else np.nan
    gross_margin = st.slider("Gross margin assumption", min_value=0.30, max_value=0.95, value=0.80, step=0.01)
    ltv = kpis["arpu"] * avg_tenure * gross_margin if pd.notna(kpis["arpu"]) and pd.notna(avg_tenure) else np.nan
    st.metric("Estimated LTV", fmt_currency(ltv))
    st.caption("Formula: ARPU x average active tenure (months) x gross margin.")


def main() -> None:
    st.title("SaaS Churn & Revenue Insights")
    st.caption("Customer-month analytics from customers, subscriptions, usage_metrics, and invoices CSV files.")

    try:
        customers, subscriptions, usage, invoices = load_data(".")
        df = prepare_model_table(customers, subscriptions, usage, invoices)
    except FileNotFoundError as err:
        st.error(f"Missing input file: {err}")
        return
    except Exception as err:  # noqa: BLE001
        st.error(f"Failed to load/prepare data: {err}")
        return

    if df.empty:
        st.warning("No records available after data preparation.")
        return

    filtered = render_sidebar_filters(df)
    if filtered.empty:
        st.warning("No rows match current filters.")
        return

    kpis = compute_kpis(filtered)
    last_month = kpis["last_month"]
    st.markdown("### KPI Snapshot")
    render_kpi_row(kpis)

    with st.expander("Metric Definitions", expanded=False):
        st.markdown(
            "- `Current MRR`: Sum of `mrr_usd` for the latest selected month.\n"
            "- `Churn rate`: `churned_in_month / active_start_month` in latest selected month.\n"
            "- `Net MRR change`: Current month MRR minus previous month MRR.\n"
            "- `ARPU`: Current month MRR divided by active customers in that month.\n"
            "- `Active customers`: Count of rows marked active in latest selected month."
        )

    render_trends_and_cohort(filtered)
    render_segment_section(filtered, last_month)
    render_revenue_drivers(filtered, last_month)
    render_model_and_correlations(filtered)
    render_ltv(filtered, kpis, last_month)

    st.subheader("Download Aggregates")
    export_df = build_download_aggregates(filtered)
    st.download_button(
        "Download filtered aggregates (CSV)",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name="filtered_aggregates.csv",
        mime="text/csv",
    )

    st.subheader("Filtered Customer-Month Table")
    data_cols = [
        "customer_id",
        "month",
        "region",
        "segment",
        "plan",
        "industry",
        "company_size",
        "acquisition_channel",
        "mrr_usd",
        "add_on_usd",
        "weekly_active_days_avg",
        "queries_run",
        "seats_used",
        "support_tickets",
        "nps",
        "discount_pct",
        "effective_price_usd_month",
        "tenure_months",
        "active",
        "churned_in_month",
        "churn_reason",
    ]
    available_cols = [c for c in data_cols if c in filtered.columns]
    st.dataframe(filtered[available_cols].sort_values(["month", "customer_id"]), use_container_width=True, height=380)


if __name__ == "__main__":
    main()
