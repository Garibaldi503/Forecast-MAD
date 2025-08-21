# 03_Forecast_Deviation — Streamlit App (Patched WAPE)
# -----------------------------------------------------------
# Fix: WAPE computation now merges group sums to avoid broadcasting issues
# -----------------------------------------------------------

import os
import platform
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

st.set_page_config(page_title="03 — Forecast Deviation Analyzer", layout="wide")
st.title("03 — Forecast Deviation Analyzer")
st.caption("Upload a CSV or use embedded demo. Patched WAPE calculation to avoid broadcasting errors.")

REQUIRED = ["date", "product_id", "forecast", "actual"]
SYNONYMS = {
    "date": {"date", "day", "period", "week", "month", "txn_date"},
    "product_id": {"product_id", "sku", "item", "product", "material"},
    "forecast": {"forecast", "forecast_qty", "fcst", "f_qty", "prediction", "pred"},
    "actual": {"actual", "actual_qty", "demand", "sales", "qty", "a_qty", "sold"},
}

def _normalize_columns(cols):
    return [c.strip().lower().replace(" ", "_") for c in cols]

def _auto_map_columns(df):
    mapped = {}
    cols = df.columns.tolist()
    for need in REQUIRED:
        found = None
        for c in cols:
            if c in SYNONYMS[need] or c == need:
                found = c
                break
        mapped[need] = found
    return mapped

@st.cache_data
def load_demo(n_products=6, periods=90, seed=42):
    rng = np.random.default_rng(seed)
    start = pd.Timestamp.today().normalize() - pd.Timedelta(days=periods-1)
    dates = pd.date_range(start, periods=periods, freq="D")
    rows = []
    for p in range(1, n_products + 1):
        base = rng.uniform(15, 80)
        season = 10 * np.sin(np.linspace(0, 3*np.pi, periods)) + rng.normal(0, 3, periods)
        demand = np.maximum(0, base + season + rng.normal(0, 4, periods)).round().astype(int)
        bias = rng.normal(0, 0.08)
        forecast = np.clip(demand * (1 + bias) + rng.normal(0, 3, periods), 0, None).round(0)
        for d, a, f in zip(dates, demand, forecast):
            rows.append({
                "date": d.date().isoformat(),
                "product_id": f"P{p:03d}",
                "forecast": float(f),
                "actual": float(a),
                "category": f"Cat-{(p % 3)+1}"
            })
    return pd.DataFrame(rows)

def _compute_metrics(df):
    work = df.copy()
    work["error"] = work["forecast"] - work["actual"]
    work["abs_error"] = (work["error"]).abs()
    non_zero = work["actual"] != 0
    work["ape"] = np.where(non_zero, (work["abs_error"] / work["actual"]).astype(float), np.nan)

    # Base aggregation
    g = work.groupby("product_id", as_index=False)
    agg = g.agg(
        periods=("date", "count"),
        actual_sum=("actual", "sum"),
        forecast_sum=("forecast", "sum"),
        mae=("abs_error", "mean"),
        rmse=("error", lambda x: float(np.sqrt(np.mean(np.square(x))))),
        mape=("ape", "mean"),
        mad=("abs_error", "mean"),
        rsfe=("error", "sum"),
    )

    # Robust WAPE: merge sums to avoid shape/broadcast problems
    sum_abs = work.groupby("product_id", as_index=False)["abs_error"].sum().rename(columns={"abs_error":"abs_error_sum"})
    agg = agg.merge(sum_abs, on="product_id", how="left")

    eps = 1e-9
    agg["wape"] = agg["abs_error_sum"] / (agg["actual_sum"] + eps)
    agg["bias_pct"] = (agg["forecast_sum"] - agg["actual_sum"]) / (agg["actual_sum"] + eps)
    agg["tracking_signal"] = agg.apply(
        lambda r: (r["rsfe"] / r["mad"]) if pd.notna(r["mad"]) and r["mad"] != 0 else np.nan,
        axis=1
    )

    # Overall row
    overall = {
        "product_id": "ALL",
        "periods": int(work.shape[0]),
        "actual_sum": float(work["actual"].sum()),
        "forecast_sum": float(work["forecast"].sum()),
        "mae": float(work["abs_error"].mean()),
        "rmse": float(np.sqrt(np.mean(np.square(work["error"])))),
        "mape": float(np.nanmean(work.loc[non_zero, "ape"])),
        "mad": float(work["abs_error"].mean()),
        "rsfe": float(work["error"].sum()),
        "abs_error_sum": float(work["abs_error"].sum()),
        "wape": float(work["abs_error"].sum() / (work["actual"].sum() + eps)),
        "bias_pct": float((work["forecast"].sum() - work["actual"].sum()) / (work["actual"].sum() + eps)),
        "tracking_signal": float((work["error"].sum() / (work["abs_error"].mean() if work["abs_error"].mean() not in [0, np.nan, 0.0] else np.nan))) if work["abs_error"].mean() not in [0, np.nan, 0.0] else np.nan,
    }
    summary = pd.concat([agg, pd.DataFrame([overall])], ignore_index=True)
    return work, summary

def _status_from_bias_ts(bias_pct, ts, bias_thr=0.10, ts_thr=4):
    if pd.isna(bias_pct) or pd.isna(ts):
        return "Check"
    if bias_pct <= -abs(bias_thr) or ts <= -abs(ts_thr):
        return "Under-forecast (stockout risk)"
    if bias_pct >= abs(bias_thr) or ts >= abs(ts_thr):
        return "Over-forecast (overstock risk)"
    return "On track"

def _fmt_pct(x):
    return f"{x:.1%}" if pd.notna(x) else ""

# Sidebar
with st.sidebar:
    st.header("Data")
    use_demo = st.toggle("Use embedded demo", value=True)
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    st.header("Filters")
    bias_thr = st.slider("Bias threshold (±%)", 0, 50, 10, format="%d") / 100.0
    ts_thr = st.slider("Tracking Signal threshold", 1, 8, 4)
    show_all = st.checkbox("Show ALL row in table", value=True)

# Load
if (not use_demo) and (uploaded is None):
    st.info("Upload a CSV or switch on the embedded demo in the sidebar.")
    st.stop()
df_raw = load_demo() if use_demo else pd.read_csv(uploaded)
df_raw.columns = _normalize_columns(df_raw.columns)
mapping = _auto_map_columns(df_raw)

missing = [k for k, v in mapping.items() if v is None]
if missing:
    with st.expander("Column mapping required (auto-detect failed). Click to map."):
        for need in missing:
            mapping[need] = st.selectbox(
                f"Select column for: {need}",
                options=[None] + df_raw.columns.tolist(),
                index=0,
            )
if any(mapping[k] is None for k in REQUIRED):
    st.error("Missing required columns after mapping. Please map date, product_id, forecast, actual.")
    st.write("Detected mapping:", mapping)
    st.stop()

work = df_raw.rename(columns={
    mapping["date"]: "date",
    mapping["product_id"]: "product_id",
    mapping["forecast"]: "forecast",
    mapping["actual"]: "actual",
}).copy()

work["date"] = pd.to_datetime(work["date"], errors="coerce")
work = work.dropna(subset=["date"])
for c in ["forecast", "actual"]:
    work[c] = pd.to_numeric(work[c], errors="coerce")
work = work.dropna(subset=["forecast", "actual"])

if work.empty:
    st.warning("No usable rows after cleaning.")
    st.stop()

# Date filter
min_d, max_d = work["date"].min(), work["date"].max()
c1, c2 = st.columns([1,3], gap="large")
with c1:
    st.subheader("Date range")
    date_range = st.date_input("Filter by date", value=(min_d.date(), max_d.date()), min_value=min_d.date(), max_value=max_d.date())
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_d, end_d = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    else:
        start_d, end_d = min_d, max_d

work = work[(work["date"] >= start_d) & (work["date"] <= end_d)]
if work.empty:
    st.warning("No data after applying filters.")
    st.stop()

# Compute & classify
row_level, summary = _compute_metrics(work)
summary["status"] = summary.apply(lambda r: _status_from_bias_ts(r["bias_pct"], r["tracking_signal"], bias_thr, ts_thr), axis=1)

# KPIs
overall = summary.loc[summary["product_id"] == "ALL"].iloc[0]
k1,k2,k3,k4,k5 = st.columns(5)
k1.metric("Overall WAPE", _fmt_pct(overall["wape"]))
k2.metric("MAPE (non-zero actuals)", _fmt_pct(overall["mape"]))
k3.metric("Bias", _fmt_pct(overall["bias_pct"]))
k4.metric("Tracking Signal", f'{overall["tracking_signal"]:.2f}' if pd.notna(overall["tracking_signal"]) else "")
k5.metric("RMSE", f'{overall["rmse"]:.2f}')

# Table
st.subheader("Product accuracy and bias")
tbl = summary.copy()
if not show_all:
    tbl = tbl[tbl["product_id"] != "ALL"]
display_cols = ["product_id","periods","actual_sum","forecast_sum","wape","mape","bias_pct","tracking_signal","mae","rmse","status"]
fmt_tbl = tbl.copy()
fmt_tbl["wape"] = fmt_tbl["wape"].apply(_fmt_pct)
fmt_tbl["mape"] = fmt_tbl["mape"].apply(_fmt_pct)
fmt_tbl["bias_pct"] = fmt_tbl["bias_pct"].apply(_fmt_pct)
st.dataframe(fmt_tbl[display_cols], use_container_width=True, hide_index=True)

# Focus lists
cA, cB = st.columns(2)
with cA:
    st.markdown("**Top under-forecast (stockout risk)**")
    under = tbl[(tbl["bias_pct"] < -abs(bias_thr)) | (tbl["tracking_signal"] < -abs(ts_thr))]
    st.dataframe(
        under.sort_values(["bias_pct","tracking_signal"]).head(10)[["product_id","bias_pct","tracking_signal","wape","mape"]].assign(
            bias_pct=lambda d: d["bias_pct"].apply(_fmt_pct),
            wape=lambda d: d["wape"].apply(_fmt_pct),
            mape=lambda d: d["mape"].apply(_fmt_pct),
        ),
        use_container_width=True, hide_index=True
    )
with cB:
    st.markdown("**Top over-forecast (overstock risk)**")
    over = tbl[(tbl["bias_pct"] > abs(bias_thr)) | (tbl["tracking_signal"] > abs(ts_thr))]
    st.dataframe(
        over.sort_values(["bias_pct","tracking_signal"], ascending=False).head(10)[["product_id","bias_pct","tracking_signal","wape","mape"]].assign(
            bias_pct=lambda d: d["bias_pct"].apply(_fmt_pct),
            wape=lambda d: d["wape"].apply(_fmt_pct),
            mape=lambda d: d["mape"].apply(_fmt_pct),
        ),
        use_container_width=True, hide_index=True
    )

# Product drill-down
st.subheader("Product drill-down")
pids = row_level["product_id"].unique().tolist()
pids = [p for p in pids if p != "ALL"]
if pids:
    picked = st.selectbox("Choose a product", options=pids, index=0)
    prod = row_level[row_level["product_id"] == picked].sort_values("date")
    c3, c4 = st.columns(2, gap="large")

    with c3:
        st.markdown("**Actual vs Forecast**")
        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(prod["date"], prod["actual"], label="Actual")
        ax.plot(prod["date"], prod["forecast"], label="Forecast")
        ax.set_xlabel("Date")
        ax.set_ylabel("Units")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig, clear_figure=True)

    with c4:
        st.markdown("**Cumulative RSFE (Forecast - Actual)**")
        fig2, ax2 = plt.subplots(figsize=(8,4))
        rsfe = (prod["forecast"] - prod["actual"]).cumsum()
        ax2.plot(prod["date"], rsfe, label="Cumulative RSFE")
        ax2.axhline(0, linestyle="--")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Units")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2, clear_figure=True)
else:
    st.info("No products to show after filtering.")

# Downloads
st.subheader("Downloads")
st.download_button(
    "Download summary CSV",
    data=summary.to_csv(index=False).encode("utf-8"),
    file_name="forecast_deviation_summary.csv",
    mime="text/csv",
)
st.download_button(
    "Download row-level CSV (with errors)",
    data=row_level.to_csv(index=False).encode("utf-8"),
    file_name="forecast_deviation_row_errors.csv",
    mime="text/csv",
)
