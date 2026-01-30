#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
flexi_csv_streamlit_plotly.py — Streamlit CSV parser + summarizer + Plotly charts

- Handles: messy headers (auto/none/manual), delimiter sniffing, numeric/datetime coercion,
  low-cardinality label-encoding for plotting, and quick summaries.
- Charts powered by Plotly (line/scatter/bar with optional time resample).
"""

import io
import csv
import re
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# -------------------- Power/Energy helpers --------------------

def add_power_energy(
        df: pd.DataFrame,
        v_col: str,
        i_col: str,
        current_scale: float = 0.001,
        voltage_scale: float = 0.001,
        time_col: str = "Time(s)",      # The column that has time in seconds
        energy_from_abs_power: bool = False,
        ) -> pd.DataFrame:
    
    # Check if the time column exists
    if time_col not in df.columns:
        st.error(f"The required time column '{time_col}' is missing in the CSV data.")

    out = df.copy()

    # 1. Power (W) = V * I
    v = pd.to_numeric(out[v_col], errors = "coerce") * float(voltage_scale) # Convert to V
    i = pd.to_numeric(out[i_col], errors = "coerce") * float(current_scale) # Convert to A
    out["Power_W"] = v * i

    # 2. Build dt in hours
    t = pd.to_numeric(out[time_col], errors = "coerce")
    dt_s = t.diff().fillna(0)   # time difference (delta) in seconds
    dt_h = dt_s / 3600.0    # convert seconds to hours
    out["Time(h)"] = out["Time(s)"] / 3600.0

    # 3. Energy integration (Wh) via trapezoidal rule
    power = out["Power_W"].astype(float)
    power_prev = power.shift(1).fillna(power)   # previous power value, handle first row
    dE = 0.5 * (power + power_prev) * dt_h  # trapezoidal area for each interval is (P1 + P2)/2 * dt
    out["Energy_Wh"] = dE.cumsum().fillna(0)

    return out

# -------------------- Coercion helpers --------------------
_BOOL_MAP = {
    "true": 1, "false": 0,
    "yes": 1, "no": 0,
    "y": 1, "n": 0,
    "on": 1, "off": 0
}

def coerce_scalar(val: str, percent_as_fraction: bool = False):
    if val is None:
        return np.nan
    s = str(val).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return np.nan

    lo = s.lower()
    if lo in _BOOL_MAP:
        return _BOOL_MAP[lo]

    pct = s.endswith("%")
    core = s[:-1].strip() if pct else s

    core = re.sub(r"^[\$€£₹]\s*", "", core)              # currency
    core = re.sub(r"(?<=\d),(?=\d{3}\b)", "", core)      # 1,234 -> 1234
    if re.match(r"^[+-]?[\d\.]+[a-zA-Z]+$", core):       # 12V -> 12
        core = re.sub(r"[a-zA-Z]+$", "", core)

    try:
        num = float(core)
        if pct:
            return num / 100.0 if percent_as_fraction else num
        return num
    except Exception:
        pass

    try:
        return pd.to_datetime(s, errors="raise", format="mixed", cache=True)
    except Exception:
        return np.nan

def coerce_series(sr: pd.Series, percent_as_fraction: bool = False) -> Tuple[pd.Series, Optional[str]]:
    dt = pd.to_datetime(sr, errors="coerce", format="mixed", cache=True)
    if dt.notna().mean() > 0.8 and dt.notna().mean() >= sr.notna().mean() * 0.8:
        return dt, "datetime"

    coerced = sr.map(lambda x: coerce_scalar(x, percent_as_fraction))
    num = pd.to_numeric(coerced, errors="coerce")
    if num.notna().mean() > 0.6:
        return num, "numeric"

    unique_vals = sr.dropna().unique()
    if 0 < len(unique_vals) <= 20:
        mapping = {k: i for i, k in enumerate(sorted(map(str, unique_vals)))}
        enc = sr.map(lambda x: mapping.get(str(x), np.nan))
        enc.attrs["label_mapping"] = mapping
        return enc, "numeric"

    return num, "numeric"

# -------------------- CSV helpers --------------------
def sniff_delimiter(sample_bytes: bytes) -> str:
    try:
        dialect = csv.Sniffer().sniff(
            sample_bytes.decode("utf-8", errors="ignore"),
            delimiters=",;\t|"
        )
        return dialect.delimiter
    except Exception:
        return ","

def infer_header_row(df_raw: pd.DataFrame) -> Optional[int]:
    rows = min(len(df_raw), 50)
    patt = re.compile(r"^[\$€£₹]?\s*[+-]?\d[\d,\.]*\s*%?$")
    for r in range(rows):
        row = df_raw.iloc[r].astype(str).str.strip()
        nonempty = row.replace("", np.nan).dropna()
        if nonempty.empty:
            continue
        numeric_like = nonempty.map(lambda s: bool(patt.match(s)))
        non_numeric_count = (~numeric_like).sum()
        numeric_count = numeric_like.sum()
        unique_ratio = nonempty.nunique() / len(nonempty)
        if non_numeric_count > numeric_count and unique_ratio >= 0.7:
            return r
    return None

@st.cache_data(show_spinner=False)
def read_flexi_csv_from_bytes(
    data: bytes,
    header_option: str = "auto",
    force_delim: Optional[str] = None,
    percent_as_fraction: bool = False
    ) -> pd.DataFrame:

    delim = force_delim if force_delim else sniff_delimiter(data)

    # Read raw as strings. Prefer fast C engine; fall back to Python engine.
    # IMPORTANT: Don't pass low_memory when engine="python".
    read_kwargs: dict = {
        "header": None,
        "sep": delim,
        "dtype": str,
        "engine": "c",              # try C engine first
        "on_bad_lines": "skip"      # works on modern pandas with C engine; else we'll retry
    }

    try:
        df_raw = pd.read_csv(io.BytesIO(data), encoding="utf-8", **read_kwargs)
    except Exception:
        # Fallback: Python engine (handles weird CSVs better)
        read_kwargs_fallback: dict = {
            "header": None,
            "sep": delim,
            "dtype": object,
            "engine": "python",
            "on_bad_lines": "skip"
            # NO low_memory here
        }
        df_raw = pd.read_csv(io.BytesIO(data), encoding="utf-8", **read_kwargs_fallback)

    # drop fully empty rows
    mask_empty = df_raw.apply(lambda r: r.isna().all() or (r.astype(str).str.strip() == "").all(), axis=1)
    df_raw = df_raw.loc[~mask_empty].reset_index(drop=True)

    # header handling
    if header_option == "auto":
        header_row = infer_header_row(df_raw)
    elif header_option == "none":
        header_row = None
    else:
        try:
            header_row = int(header_option)
        except Exception:
            header_row = None

    if header_row is not None and 0 <= header_row < len(df_raw):
        columns = df_raw.iloc[header_row].astype(str).str.strip()
        data_df = df_raw.iloc[header_row + 1:].reset_index(drop=True)
        data_df.columns = columns
    else:
        data_df = df_raw.copy()
        data_df.columns = [f"col_{i}" for i in range(data_df.shape[1])]

    # coerce columns
    coerced_cols, col_kinds = {}, {}
    for c in data_df.columns:
        coerced, kind = coerce_series(data_df[c], percent_as_fraction=percent_as_fraction)
        coerced_cols[c] = coerced
        col_kinds[c] = kind
    df = pd.DataFrame(coerced_cols)

    # prefer datetime column as index
    dt_candidates = [c for c, k in col_kinds.items() if k == "datetime"]
    if dt_candidates:
        best = max(dt_candidates, key=lambda c: df[c].notna().sum())
        df = df.set_index(best, drop=False)

    df = df.dropna(axis=1, how="all")
    return df

def summarize_df(df: pd.DataFrame) -> dict:
    summary = {}
    num = df.select_dtypes(include=[np.number])
    if not num.empty:
        summary["numeric_overview"] = num.describe().T
    miss = df.isna().sum().sort_values(ascending=False)
    summary["missing_counts"] = miss[miss > 0]
    cat = df.select_dtypes(exclude=[np.number, "datetime64"])
    cat_top = {}
    for c in cat.columns:
        vc = cat[c].astype(str).value_counts(dropna=True).head(10)
        if not vc.empty:
            cat_top[c] = vc
    summary["categorical_tops"] = cat_top
    return summary

def process_csv(data: bytes, num: int) -> pd.DataFrame:
    """This function takes a single CSV file, does parsing, coercion, and derived metrics."""
    try:
        df = read_flexi_csv_from_bytes(data=data, header_option=header_opt, force_delim=delim_val, percent_as_fraction=percent_as_fraction)
        st.success(f"Successfully parsed Hydrophone {num} data: {df.shape[0]} rows × {df.shape[1]} columns")
    except Exception as e:
        st.error(f"Error parsing Hydrophone {num} data: {e}")
        st.stop()
    if show_raw:
        with st.expander(f"Hydrophone {num} Raw (first 4 KB)", expanded=False):
            st.code(data_1[:4096].decode("utf-8", errors="ignore"))
    st.success(f"Parsed: {df.shape[0]} rows × {df.shape[1]} columns")
    # Compute derived energy and power columns
    try:
        df = add_power_energy(
            df,
            v_col="battVoltage",
            i_col="battCurrent",
            voltage_scale=0.001,   # mV→V default 0.001
            current_scale=0.001,   # mA→A default 0.001
            time_col="Time(s)",
        )
        st.success("Added: **Power_W** and **Energy_Wh**")
    except Exception as e:
        st.warning(f"Could not add derived metrics: {e}")
    df = df.rename(columns={col: f"{num}_{col}" for col in df.columns})
    st.success(f"Hydrophone {num} columns renamed with '{num}_' prefix.")

    # Reset index to default integer index
    df.reset_index(drop=True, inplace=True)
    return df

# -------------------- Plotly charting --------------------
def build_plotly_dual_axis(
    df: pd.DataFrame,
    x: Optional[str],
    y_left: list[str],
    y_right: list[str],
    kind: str,
    title: str
    ) -> go.Figure:
    """
    Build a Plotly chart with dual y-axes (left and right).
    - `y_left` and `y_right` are lists of column names to plot on the left and right Y axes, respectively.
    """
    # Resolve X
    if x is None and isinstance(df.index, pd.DatetimeIndex):
        x_vals = df.index
        x_label = df.index.name or "index"
    elif x is None:
        x_vals = np.arange(len(df))
        x_label = "index"
    else:
        x_vals = df[x]
        x_label = x

    fig = go.Figure()

    # LEFT AXIS (e.g., Power)
    for c in y_left:
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=df[c],
                name=c,
                mode="lines" if kind == "line" else "markers",
                yaxis="y1"
            )
        )

    # RIGHT AXIS (e.g., Energy)
    for c in y_right:
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=df[c],
                name=c,
                mode="lines" if kind == "line" else "markers",
                yaxis="y2"
            )
        )

    # Update layout with dual axes
    fig.update_layout(
        title=title,
        xaxis=dict(title=x_label),
        yaxis=dict(
            title=" • ".join(y_left) if y_left else "Value",
            side="left",
            showgrid=True
        ),
        yaxis2=dict(
            title=" • ".join(y_right),
            side="right",
            overlaying="y",
            showgrid=False
        ),
        hovermode="x unified",
        margin=dict(l=60, r=60, t=60, b=40),
        legend=dict(
            orientation="h",  # horizontal legend
            yanchor="bottom",
            y=-0.4,
            xanchor="center",
            x=0.5,
            title="Click to toggle • Double-click to isolate"
        )
    )

    return fig


# -------------------- UI --------------------
st.set_page_config(page_title="CSV Plotter", layout="wide")
st.title("CSV Plotter")

with st.sidebar:
    st.header("Parse Options")
    header_opt = st.selectbox("Header row", ["auto", "none", "0", "1", "2", "3", "4", "5"], index=0)
    force_delim = st.selectbox("Delimiter", ["(auto)", ",", ";", "\\t", "|"], index=0)
    delim_val = None if force_delim == "(auto)" else ("\t" if force_delim == "\\t" else force_delim)
    percent_as_fraction = st.checkbox("Interpret % as fractions (45% → 0.45)", value=False)
    show_raw = st.checkbox("Show raw file head (debug)", value=False)

# Upload two hydrophone CSV files
uploaded_file_1 = st.file_uploader("Upload Hydrophone 1 CSV", type=["csv"])
uploaded_file_2 = st.file_uploader("Upload Hydrophone 2 CSV", type=["csv"])

if not uploaded_file_1 and not uploaded_file_2:
    st.info("Upload a file to begin.")
    st.stop()

if uploaded_file_1:
    data_1 = uploaded_file_1.read()
    df_1 = process_csv(data=data_1, num=1)

if uploaded_file_2:
    data_2 = uploaded_file_2.read()
    df_2 = process_csv(data=data_2, num=2)
    
# If both files are uploaded, MERGE THEM
if uploaded_file_1 and uploaded_file_2:
    df = pd.concat([df_1, df_2], axis=1)
    st.success(f"Merged DataFrame shape: {df.shape[0]} rows × {df.shape[1]} columns")
else:
    # Use whichever dataframe is uploaded
    df = df_1 if uploaded_file_1 else (df_2 if uploaded_file_2 else None)

if df is not None:
    with st.expander("Data preview", expanded=True):
        st.dataframe(df.head(200), width = "stretch", hide_index=True)

    summary = summarize_df(df)
else:
    st.error("No data available to summarize.")
    st.stop()
col1, col2 = st.columns([4,1])
with col1:
    if summary.get("numeric_overview") is not None:
        st.subheader("Numeric Overview")
        st.dataframe(summary["numeric_overview"], width = "stretch")
with col2:
    miss = summary.get("missing_counts", pd.Series(dtype=int))
    if not miss.empty:
        st.subheader("Missing Values")
        st.dataframe(miss.to_frame("missing"), width = "stretch")

if summary.get("categorical_tops"):
    with st.expander("Categorical: Top values", expanded=False):
        for c, vc in summary["categorical_tops"].items():
            st.markdown(f"**{c}**")
            st.dataframe(vc.to_frame("count"))

# ---- Plotly controls
st.subheader("Chart")

# Let user select columns
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
all_cols = list(df.columns)

c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.2, 1])
with c1:
    x_choice = st.selectbox(
        "X axis",
        ["(index)"] + all_cols,
        index=("Time(h)" in all_cols and all_cols.index("Time(h)") + 1) or 0
    )

with c2:
    y_left = st.multiselect(
        "Left Y axis",
        num_cols,
        default=[c for c in ["Power_W"] if c in num_cols]  # Pre-select Power if available
    )

with c3:
    y_right = st.multiselect(
        "Right Y axis",
        num_cols,
        default=[c for c in ["Energy_Wh"] if c in num_cols]  # Pre-select Energy if available
    )

with c4:
    kind = st.selectbox("Chart type", ["line", "scatter"], index=0)

# Dynamically build the figure with dual axes
fig = build_plotly_dual_axis(
    df=df,
    x=x_choice,
    y_left=y_left,
    y_right=y_right,
    kind=kind,
    title="Time Series Chart"
)

# Display the plot
st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

# ---- Downloads
st.subheader("Downloads")
st.download_button(
    "Download cleaned CSV",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="cleaned.csv",
    mime="text/csv"
)

# Simple profile CSVs
profile_files = []
if summary.get("numeric_overview") is not None:
    num_desc = summary["numeric_overview"].copy()
    num_desc.insert(0, "metric", num_desc.index)
    profile_files.append(("numeric_overview.csv", num_desc.to_csv(index=False)))
miss = summary.get("missing_counts", pd.Series(dtype=int))
if not miss.empty:
    ms = miss.reset_index()
    ms.columns = ["column", "missing"]
    profile_files.append(("missing_counts.csv", ms.to_csv(index=False)))

if profile_files:
    for fname, csv_text in profile_files:
        st.download_button(f"Download {fname}", data=csv_text.encode("utf-8"),
                           file_name=fname, mime="text/csv")
else:
    st.caption("No numeric summary or missing values to export.")
