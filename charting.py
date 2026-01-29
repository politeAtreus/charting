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
def _guess_col(df, candidates):
    cols = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in cols:
            return cols[name.lower()]
    # fuzzy match
    for k, orig in cols.items():
        if any(name.lower() in k for name in candidates):
            return orig
    return None

def add_power_energy(
        df: pd.DataFrame,
        v_col: str,
        i_col: str,
        current_scale: float = 0.001,
        voltage_scale: float = 0.001,
        time_source: str = "auto",
        time_col: str | None = None,
        constant_dt_s: float | None = None,
        ) -> pd.DataFrame:
    out = df.copy()

    # 1. Power (W) = V * I
    v = pd.to_numeric(out[v_col], errors = "coerce") * float(voltage_scale)
    i = pd.to_numeric(out[i_col], errors = "coerce") * float(current_scale)
    out["Power_W"] = v * i

    # 2. Build dt in hours
    if time_source == "auto":
        if isinstance(out.index, pd.DatetimeIndex):
            t = out.index.to_series()
        else:
            # pick the first date-time like col
            dt_cols = [c for c in out.columns if pd.api.types.is_datetime64_any_dtype(out[c])]
            t = out[dt_cols[0]] if dt_cols else None
    
    elif time_source == "column" and time_col:
        t = out[time_col]
    
    else:
        t = None

    if t is not None and pd.api.types.is_datetime64_any_dtype(t):
        dt_h = t.diff().dt.total_seconds().fillna(0) / 3600.0
    else:
        # constant sample period to fallback on (sec)
        if not constant_dt_s:
            constant_dt_s = 1.0 # default 1 second
        dt_h = pd.Series(constant_dt_s / 3600.0, index=out.index)

    # 3. Energy integration (Wh) via trapezoidal rule
    power = out["Power_W"].astype(float)
    power_prev = power.shift(1).fillna(power)
    dE = 0.5 * (power + power_prev) * dt_h
    out["Energy_Wh"] = dE.cumsum().ffill().fillna(0)

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
            delimiters=[",", ";", "\t", "|"]
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
    read_kwargs = dict(
        header=None,
        sep=delim,
        dtype=str,
        engine="c",              # try C engine first
        on_bad_lines="skip"      # works on modern pandas with C engine; else we'll retry
    )

    try:
        df_raw = pd.read_csv(io.BytesIO(data), encoding="utf-8", errors="ignore", **read_kwargs)
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
        df_raw = pd.read_csv(io.BytesIO(data), encoding="utf-8", errors="ignore", **read_kwargs_fallback)

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

# -------------------- Plotly charting --------------------
def build_plotly_figure(df: pd.DataFrame, kind: str, x: Optional[str], y_cols: list,
                        downsample: Optional[str], agg: str, title: str) -> go.Figure:
    if x is None and isinstance(df.index, pd.DatetimeIndex):
        x_vals = df.index
    elif x is None:
        x_vals = np.arange(len(df))
    else:
        x_vals = df[x]

    num = df.select_dtypes(include=[np.number])
    y_cols = [c for c in y_cols if c in num.columns]
    if not y_cols:
        fig = go.Figure()
        fig.add_annotation(x=0.5, y=0.5, text="No numeric columns selected.", showarrow=False)
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        fig.update_layout(title=title or "Quick Plot")
        return fig

    plot_df = pd.DataFrame({"__x__": x_vals})
    for c in y_cols:
        plot_df[c] = df[c]

    # optional time resample (if x is datetime-like index/column)
    if downsample and (isinstance(plot_df["__x__"], pd.Series) and
                       pd.api.types.is_datetime64_any_dtype(plot_df["__x__"])):
        g = plot_df.set_index("__x__").resample(downsample)
        if agg == "mean":
            plot_df = g.mean().reset_index()
        elif agg == "median":
            plot_df = g.median().reset_index()
        elif agg == "min":
            plot_df = g.min().reset_index()
        elif agg == "max":
            plot_df = g.max().reset_index()
        elif agg == "sum":
            plot_df = g.sum().reset_index()
        plot_df.rename(columns={"__x__": "x"}, inplace=True)
        x_field = "x"
    else:
        plot_df.rename(columns={"__x__": "x"}, inplace=True)
        x_field = "x"

    if kind == "line":
        fig = go.Figure()
        for c in y_cols:
            fig.add_trace(go.Scatter(x=plot_df[x_field], y=plot_df[c], mode="lines", name=c))
    elif kind == "scatter":
        fig = go.Figure()
        for c in y_cols:
            fig.add_trace(go.Scatter(x=plot_df[x_field], y=plot_df[c], mode="markers", name=c))
    else:  # bar
        fig = go.Figure()
        for c in y_cols:
            fig.add_trace(go.Bar(x=plot_df[x_field], y=plot_df[c], name=c))

    fig.update_layout(
        title=title or "Quick Plot",
        xaxis_title=(x if x is not None else (df.index.name or "index")),
        yaxis_title="value",
        legend_title="Series",
        margin=dict(l=40, r=20, t=60, b=40),
        hovermode="x unified",
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

uploaded = st.file_uploader("Upload a CSV / TXT / LOG", type=["csv", "txt", "log"])
if not uploaded:
    st.info("Upload a file to begin.")
    st.stop()

data = uploaded.read()
if show_raw:
    with st.expander("Raw (first 4 KB)", expanded=False):
        st.code(data[:4096].decode("utf-8", errors="ignore"))

try:
    df = read_flexi_csv_from_bytes(
        data=data,
        header_option=header_opt,
        force_delim=delim_val,
        percent_as_fraction=percent_as_fraction
    )
except Exception as e:
    st.error(f"Failed to parse CSV: {e}")
    st.stop()

st.success(f"Parsed: {df.shape[0]} rows × {df.shape[1]} columns")
st.caption(f"Numeric: {len(df.select_dtypes(include=[np.number]).columns)} | "
           f"Datetime-like: {sum(pd.api.types.is_datetime64_any_dtype(df[c]) for c in df.columns)}")

st.markdown("### Derived metrics (Power & Energy)")

# Guess defaults
default_v = _guess_col(df, ["battVolts", "volts", "voltage", "V", "mV"])
default_i = _guess_col(df, ["battCurrent", "current", "I", "amps", "mA"])

colA, colB, colC, colD, colE = st.columns([1.5, 1.5, 1.1, 1.1, 1.4])

with colA:
    v_col = st.selectbox("Voltage column", options=list(df.columns), index=(list(df.columns).index(default_v) if default_v in df.columns else 0))

with colB:
    i_col = st.selectbox("Current column", options=list(df.columns), index=(list(df.columns).index(default_i) if default_i in df.columns else 0))
    
with colC:
    voltage_scale = st.number_input("Voltage scale (→ V)", value=0.001, step=0.001, format="%.6f", help="0.001 if column is in mV")

with colD:
    current_scale = st.number_input("Current scale (→ A)", value=0.001, step=0.001, format="%.6f", help="0.001 if column is in mA")

with colE:
    t_mode = st.selectbox("Time source", ["auto", "column", "constant"], index=0)

time_col = None
const_dt = None
if t_mode == "column":
    dt_candidates = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    opts = dt_candidates if dt_candidates else list(df.columns)
    time_col = st.selectbox("Datetime column", options=opts, index=0)
elif t_mode == "constant":
    const_dt = st.number_input("Constant sample period (seconds)", value=1.0, min_value=1e-6, step=0.1)

# Compute columns
try:
    df = add_power_energy(
        df,
        v_col=v_col,
        i_col=i_col,
        voltage_scale=voltage_scale,   # mV→V default 0.001
        current_scale=current_scale,   # mA→A default 0.001
        time_source=t_mode,
        time_col=time_col,
        constant_dt_s=const_dt
    )
    st.success("Added: **Power_W** and **Energy_Wh**")
except Exception as e:
    st.warning(f"Could not add derived metrics: {e}")

with st.expander("Data preview", expanded=True):
    st.dataframe(df.head(200), width = "stretch")

summary = summarize_df(df)
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
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
all_cols = list(df.columns)

c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.2, 1])
with c1:
    x_choice = st.selectbox("X axis", ["(index)"] + all_cols)
with c2:
    y_choices = st.multiselect("Y axis (numeric)", num_cols)
with c3:
    kind = st.selectbox("Chart type", ["line", "scatter", "bar"], index=0)
with c4:
    downsample = st.selectbox("Resample", ["(none)", "s", "5s", "1min", "5min", "15min", "1h", "1d"], index=0)
    downsample = None if downsample == "(none)" else downsample
    agg = st.selectbox("Agg", ["mean", "median", "min", "max", "sum"], index=0)

x_col = None if x_choice == "(index)" else x_choice
fig = build_plotly_figure(df, kind, x_col, y_choices, downsample, agg, title=uploaded.name)
st.plotly_chart(fig, width = "stretch", config={"displaylogo": False})

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
