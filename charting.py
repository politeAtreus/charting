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
import time
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

def extract_hydrophone_id(filename: str) -> str:
    """Extract serial number from 'Hydrophone_6956_20260415_132938.csv' → '6956'."""
    match = re.search(r"Hydrophone_(\d+)_", filename)
    return match.group(1) if match else filename.split(".")[0]

# -------------------- Power/Energy helpers --------------------

def add_power_energy(
        df: pd.DataFrame,
        v_col: str,
        i_col: str,
        p_out_col: str = "Power_W",
        e_out_col: str = "Energy_Wh",
        current_scale: float = 0.001,
        voltage_scale: float = 0.001,
        time_col: str = "Time(s)",      # The column that has time in seconds
        energy_from_abs_power: bool = False,
        ) -> pd.DataFrame:
    
    # Check if the time column exists
    if time_col not in df.columns:
        st.error(f"The required time column '{time_col}' is missing in the CSV data.")
        return df

    out = df.copy()

    # 1. Power (W) = V * I
    v = pd.to_numeric(out[v_col], errors = "coerce") * float(voltage_scale) # Convert to V
    i = pd.to_numeric(out[i_col], errors = "coerce") * float(current_scale) # Convert to A
    out[p_out_col] = v * i

    # 2. Build dt in hours
    t = pd.to_numeric(out[time_col], errors = "coerce")
    dt_s = t.diff().clip(lower=0).fillna(0)
    # Dropped invalid CSV rows leave index gaps. Do not integrate across outages.
    dt_s = dt_s.where(out.index.to_series().diff().eq(1), 0)
    dt_h = dt_s / 3600.0    # convert seconds to hours
    out["Time(h)"] = t / 3600.0

    # 3. Energy integration (Wh) via trapezoidal rule
    power = out[p_out_col].astype(float)
    if energy_from_abs_power:
        power = power.abs()
    power_prev = power.shift(1)   # previous power value; a missing sample stays missing
    dE = 0.5 * (power + power_prev) * dt_h  # trapezoidal area for each interval is (P1 + P2)/2 * dt
    out[e_out_col] = dE.fillna(0).cumsum()

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

    # Handle boolean-like values
    lo = s.lower()
    if lo in _BOOL_MAP:
        return _BOOL_MAP[lo]

    # Handle percentage values (e.g., 45%)
    pct = s.endswith("%")
    core = s[:-1].strip() if pct else s

    # Remove currency symbols
    core = re.sub(r"^[\$€£₹]\s*", "", core)
    core = re.sub(r"(?<=\d),(?=\d{3}\b)", "", core)  # 1,234 -> 1234
    if re.match(r"^[+-]?[\d\.]+[a-zA-Z]+$", core):   # 12V -> 12
        core = re.sub(r"[a-zA-Z]+$", "", core)

    try:
        num = float(core)
        if pct:
            return num / 100.0 if percent_as_fraction else num
        return num
    except Exception:
        pass

    # If it's not a number, try converting to datetime
    try:
        return pd.to_datetime(s, errors="raise", format="mixed", cache=True)
    except Exception:
        return np.nan


def coerce_series(sr: pd.Series, percent_as_fraction: bool = False) -> Tuple[pd.Series, Optional[str]]:
    # 1. Fast datetime check
    dt = pd.to_datetime(sr, errors="coerce", format="mixed", cache=True)
    if dt.notna().mean() > 0.8 and dt.notna().mean() >= sr.notna().mean() * 0.8:
        return dt, "datetime"

    # 2. Fast path: try raw to_numeric with no string cleaning.
    #    For columns that are already clean numbers this is ~50x faster than .map(coerce_scalar).
    #    Use sr.notna() only — no str ops needed since we read with dtype=str,
    #    so missing cells are already float NaN, not the string "nan".
    num = pd.to_numeric(sr, errors="coerce")
    populated = sr.notna()
    populated_count = populated.sum()

    if populated_count == 0:
        return num, "numeric"

    if num[populated].notna().mean() > 0.9:
        # Column is clean numeric — skip all regex
        return num, "numeric"

    # 3. Slow path: only reached for messy columns (currency, units, booleans, etc.)
    lo = sr.str.strip().str.lower()
    bool_mask = lo.isin(_BOOL_MAP.keys())

    cleaned = sr.str.strip()
    cleaned = cleaned.str.replace(r"^[\$€£₹]\s*", "", regex=True)
    cleaned = cleaned.str.replace(r"(?<=\d),(?=\d{3}\b)", "", regex=True)
    cleaned = cleaned.str.replace(r"^([+-]?[\d\.]+)[a-zA-Z]+$", r"\1", regex=True)

    if percent_as_fraction:
        pct_mask = cleaned.str.endswith("%")
        cleaned = cleaned.str.rstrip("%")

    num2 = pd.to_numeric(cleaned, errors="coerce")
    num2 = num2.where(~bool_mask, lo.map(_BOOL_MAP))

    if percent_as_fraction:
        num2 = num2.where(~pct_mask, num2 / 100.0)

    if num2[populated].notna().mean() > 0.6:
        return num2, "numeric"

    # 4. Categorical
    unique_vals = sr.dropna().unique()
    if len(unique_vals) <= 20:
        mapping = {k: i for i, k in enumerate(sorted(map(str, unique_vals)))}
        enc = sr.map(lambda x: mapping.get(str(x), np.nan))
        enc.attrs["label_mapping"] = mapping
        return enc, "categorical"

    return sr, "string"

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
    percent_as_fraction: bool = False,
    ) -> tuple[pd.DataFrame, list[str]]:

    STRING_COLUMNS = {"status", "battstatus"}

    delim = force_delim if force_delim else sniff_delimiter(data)

    df_raw = pd.read_csv(io.BytesIO(data), delimiter=delim, header=None, dtype=str, engine="python")

    # Drop fully empty rows
    mask_empty = df_raw.apply(lambda r: r.isna().all() or (r.astype(str).str.strip() == "").all(), axis=1)
    df_raw = df_raw.loc[~mask_empty].reset_index(drop=True)

    # Header handling
    if header_option == "auto":
        header_row = infer_header_row(df_raw)
        if header_row is None:
            header_row = 1  # Default to row 1 if no header found
    elif header_option == "none":
        header_row = None
    else:
        try:
            header_row = int(header_option) - 1  # Convert to 0-based index
        except Exception:
            header_row = None

    # ---------- metadata extraction ----------
    metadata_lines: list[str] = []
    if header_row is not None and header_row > 0:
        meta_df = df_raw.iloc[:header_row].copy()

        # join non-empty cells per row into one readable line
        for _, row in meta_df.iterrows():
            parts = [str(v).strip() for v in row.tolist() if v is not None and str(v).strip() != "" and str(v).strip().lower() != "nan"]
            if parts:
                metadata_lines.append(" | ".join(parts))

    if header_row is not None and 0 <= header_row < len(df_raw):
        columns = df_raw.iloc[header_row].astype(str).str.strip()
        data_df = df_raw.iloc[header_row + 1:].reset_index(drop=True)
        data_df.columns = columns
    else:
        data_df = df_raw.copy()
        data_df.columns = [f"col_{i}" for i in range(data_df.shape[1])]

    # Reject failed battery reads before coercion can encode battStatus as a number.
    # Keep the original index here so energy integration can recognize outages.
    invalid_rows = pd.Series(False, index=data_df.index)
    for c in data_df.columns:
        if str(c).lstrip("\ufeff").strip().casefold() == "battstatus":
            status = data_df[c].astype("string").str.strip().str.casefold()
            invalid_rows |= status.eq("no data available").fillna(False)
    skipped_rows = int(invalid_rows.sum())
    data_df = data_df.loc[~invalid_rows]

    # Coerce columns (numeric, string, or categorical)
    coerced_cols, col_kinds = {}, {}
    for c in data_df.columns:
        if str(c).casefold() in STRING_COLUMNS:
            coerced_cols[c] = data_df[c].astype("string").str.strip()
            col_kinds[c] = "string"
        else:
            coerced, kind = coerce_series(data_df[c], percent_as_fraction=percent_as_fraction)
            coerced_cols[c] = coerced
            col_kinds[c] = kind
    df = pd.DataFrame(coerced_cols)

    # Drop any columns that are completely empty
    if not df.empty:
        df = df.dropna(axis=1, how="all")
    df.attrs["skipped_no_data_rows"] = skipped_rows
    return df, metadata_lines

# -------------------- Data summarization --------------------

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

def process_csv(data: bytes, num: str, header_opt: str, delim_val, percent_as_fraction: bool, 
                show_raw: bool, enable_power_energy: bool) -> pd.DataFrame:
    """This function takes a single CSV file, does parsing, coercion, and derived metrics."""

    try:
        df, metadata_lines = read_flexi_csv_from_bytes(data=data, header_option=header_opt, force_delim=delim_val, 
                                                       percent_as_fraction=percent_as_fraction)
        st.caption(f"CSV {num}: {df.shape[0]} rows × {df.shape[1]} columns | Delimiter: '{delim_val or 'auto'}' | Header: '{header_opt}'")
    except Exception as e:
        st.error(f"Error parsing csv {num} data: {e}")
        st.stop()
    
    # --- display metadata ---
    if metadata_lines:
        with st.expander(f"Metadata (csv {num})", expanded=False):
            st.text("\n".join(metadata_lines))
    
    if show_raw:
        with st.expander(f"csv {num} Raw (first 4 KB)", expanded=False):
            st.code(data[:4096].decode("utf-8", errors="ignore"))
    st.toast(f"Parsed: {df.shape[0]} rows × {df.shape[1]} columns")
    skipped_rows = df.attrs.get("skipped_no_data_rows", 0)
    if skipped_rows:
        st.caption(f"Hydrophone {num}: skipped {skipped_rows} rows with battStatus = 'No Data Available'.")

    # Offer both elapsed-time units even when Power/Energy is disabled.
    if "Time(s)" in df.columns:
        df["Time(s)"] = pd.to_numeric(df["Time(s)"], errors="coerce")
        df["Time(h)"] = df["Time(s)"] / 3600.0
    elif "Time(h)" in df.columns:
        df["Time(h)"] = pd.to_numeric(df["Time(h)"], errors="coerce")
        df["Time(s)"] = df["Time(h)"] * 3600.0

    # Compute derived energy and power columns
    if enable_power_energy:
        try:
            df = add_power_energy(
                df,
                v_col="battVoltage",
                i_col="battCurrent",
                voltage_scale=0.001,   # mV→V default 0.001
                current_scale=0.001,   # mA→A default 0.001
                time_col="Time(s)",
                p_out_col="Power_W",
                e_out_col="Energy_Wh"
            )
            st.toast("Added: **Power_W** and **Energy_Wh**")
        except Exception as e:
            st.warning(f"Could not add derived metrics: {e}")
        
        # Try to add powe rand energy from dmm measurements
        try:
            df = add_power_energy(
                df=df,
                v_col="dmm_V",
                i_col="dmm_A",
                voltage_scale=1.0,   # DMM already in V
                current_scale=1.0,   # DMM already in A
                time_col="Time(s)",
                p_out_col="DMM_Power_W",
                e_out_col="DMM_Energy_Wh"
            )
            st.toast("Added: **DMM_Power_W** and **DMM_Energy_Wh**")
        except Exception as e:
            st.info(f"No DMM columns found or error computing DMM power/energy: {e}")
    
    # Prefix columns with hydrophone number
    df = df.rename(columns={col: f"{num}_{col}" for col in df.columns})
    st.toast(f"Hydrophone {num} columns renamed with '{num}_' prefix.")

    # Reset index to default integer index
    df.reset_index(drop=True, inplace=True)
    return df


def prepare_comparison_frame(
    df: pd.DataFrame,
    hydro_id: str,
    selected_stages: Optional[list[str]] = None,
    alignment: str = "Recorded time",
    anchor_stage: Optional[str] = None,
    offset_hours: float = 0.0,
) -> tuple[pd.DataFrame, Optional[float]]:
    """Filter this hydrophone's stages, then shift its own time columns.

    Return the shifted frame and the time (hours) used as zero, or None if
    there is no numeric time. A stage anchor comes from the original valid
    data, even if the stage itself is hidden by the filter.
    """
    status_col = f"{hydro_id}_Status"
    out = df.copy()
    if selected_stages is not None and status_col in out.columns:
        out = out.loc[out[status_col].isin(selected_stages)].copy()

    seconds_col, hours_col = f"{hydro_id}_Time(s)", f"{hydro_id}_Time(h)"
    time_col = seconds_col if seconds_col in df.columns else hours_col
    if time_col not in df.columns or out.empty:
        return out.reset_index(drop=True), None
    units_per_hour = 3600.0 if time_col == seconds_col else 1.0
    origin_hours = 0.0
    if alignment == "Start of selected data":
        times = pd.to_numeric(out[time_col], errors="coerce").dropna()
        if times.empty:
            return out.reset_index(drop=True), None
        origin_hours = float(times.iloc[0]) / units_per_hour
    elif alignment == "Start of a soak stage":
        if status_col not in df.columns:
            raise ValueError("no Status column for the selected stage anchor")
        times = pd.to_numeric(
            df.loc[df[status_col].eq(anchor_stage), time_col], errors="coerce"
        ).dropna()
        if times.empty:
            raise ValueError(f"no valid time for stage '{anchor_stage}'")
        origin_hours = float(times.iloc[0]) / units_per_hour

    for col, scale in [(seconds_col, 3600.0), (hours_col, 1.0)]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce") + (offset_hours - origin_hours) * scale
    return out.reset_index(drop=True), origin_hours


def time_axes_for_traces(
    frames: dict[str, pd.DataFrame], x: Optional[str]
) -> dict[str, Optional[str]]:
    """Map each trace to its own clock when any hydrophone time X is selected."""
    for hydro_id in frames:
        for unit in ("s", "h"):
            if x == f"{hydro_id}_Time({unit})":
                return {
                    col: f"{source}_Time({unit})" if f"{source}_Time({unit})" in frame.columns else None
                    for source, frame in frames.items()
                    for col in frame.columns
                }
    return {}


def reset_time_offsets(hydro_ids):
    for hydro_id in hydro_ids:
        st.session_state[f"time_offset_{hydro_id}"] = 0.0


def _clean_for_log(x_series, y_series, need_log_x: bool, need_log_y: bool):
        """Mask invalid log values without closing gaps between valid samples."""
        x_s = pd.Series(x_series).reset_index(drop=True)
        if need_log_x:
            x_s = pd.to_numeric(x_s, errors="coerce")
        y_s = pd.to_numeric(pd.Series(y_series).reset_index(drop=True), errors="coerce")
        mask = x_s.notna() & y_s.notna()
        if need_log_x:
            mask &= x_s > 0
        if need_log_y:
            mask &= y_s > 0
        return x_s, y_s.where(mask)

# -------------------- Plotly charting --------------------
def build_plotly_dual_axis(
    df: pd.DataFrame,
    x: Optional[str],
    y_left: list[str],
    y_right: list[str],
    kind: str,
    title: str,
    log_x: bool = False,
    log_y_left: bool = False,
    log_y_right: bool = False,
    trace_x_columns: Optional[dict[str, Optional[str]]] = None,
    ) -> go.Figure:
    """
    Build a Plotly chart with dual y-axes (left and right).
    - `y_left` and `y_right` are lists of column names to plot on the left and right Y axes, respectively.
    """
    # Resolve X
    if x in (None, "(index)"):
        x_vals = df.index if df.index.name is not None else np.arange(len(df))
        x_label = "index"
    elif x in df.columns:
        x_vals = df[x]
        x_label = x
    else:
        x_vals = np.arange(len(df))
        x_label = "index"
    
    # Coerce X to numeric if we're doing log-x
    if log_x:
        x_num = pd.to_numeric(pd.Series(x_vals), errors="coerce")
        x_vals = x_num

    fig = go.Figure()
    trace_x_columns = trace_x_columns or {}

    def trace_values(column, log_y):
        trace_x = trace_x_columns.get(column, x)
        if column in trace_x_columns and trace_x is None:
            return None
        own_x = df[trace_x] if trace_x in df.columns else x_vals
        if log_x or (log_y and pd.api.types.is_numeric_dtype(df[column])):
            return _clean_for_log(own_x, df[column], log_x, log_y)
        return own_x, df[column]

    # LEFT AXIS traces
    for c in y_left:
        if c not in df.columns:
            continue

        values = trace_values(c, log_y_left)
        if values is None:
            continue
        x_plot, y_plot = values

        fig.add_trace(
            go.Scatter(
                x=x_plot,
                y=y_plot,
                name=c,
                mode="lines" if kind == "line" else "markers",
                yaxis="y1",
                connectgaps=False,
            )
        )

    # RIGHT AXIS traces
    for c in y_right:
        if c not in df.columns:
            continue

        values = trace_values(c, log_y_right)
        if values is None:
            continue
        x_plot, y_plot = values

        fig.add_trace(
            go.Scatter(
                x=x_plot,
                y=y_plot,
                name=c,
                mode="lines" if kind == "line" else "markers",
                yaxis="y2",
                connectgaps=False,
            )
        )

    # Update layout with dual axes
    fig.update_layout(
        title=title,
        xaxis=dict(
            title="Time (hours)" if trace_x_columns and x_label.endswith("_Time(h)") else (
                "Time (seconds)" if trace_x_columns else x_label
            ),
            type="log" if log_x else (
                "date" if pd.api.types.is_datetime64_any_dtype(pd.Series(x_vals)) else "linear"
            ),
            nticks=60,
        ),
        yaxis=dict(
            title=" • ".join(y_left) if y_left else "Value",
            side="left",
            showgrid=True,
            type="log" if log_y_left else "linear",
            nticks=20,
        ),
        yaxis2=dict(
            title=" • ".join(y_right),
            side="right",
            overlaying="y",
            showgrid=False,
            type="log" if log_y_right else "linear",
            nticks=20,
        ),
        hovermode="x unified",
        margin=dict(l=60, r=60, t=60, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.4,
            xanchor="center",
            x=0.5,
            title="Click to toggle • Double-click to isolate",
        ),
    )

    return fig


# -------------------- UI --------------------
st.set_page_config(page_title="CSV Plotter", layout="wide", initial_sidebar_state="expanded")
st.title("CSV Plotter")

with st.sidebar:
    st.header("Parse Options")

    # Header and delimiter options
    header_opt = st.selectbox("Header row", ["auto", "none"] + [str(i) for i in range(1,20)], index=0)
    force_delim = st.selectbox("Delimiter", ["(auto)", ",", ";", "\\t", "|"], index=0)
    delim_val = None if force_delim == "(auto)" else ("\t" if force_delim == "\\t" else force_delim)
    percent_as_fraction = st.checkbox("Interpret % as fractions (45% → 0.45)", value=False)

    # Debug options
    show_raw = st.checkbox("Show raw file head (debug)", value=False)

    # Power/Energy option
    enable_power_energy = st.checkbox("Compute Power/Energy, requires V, I and T.", value=False)

    # Log axis options
    log_x = st.checkbox("X-axis Log Mode", value=False)
    log_y_left = st.checkbox("Left Y-axis Log Mode", value=False)
    log_y_right = st.checkbox("Right Y-axis Log Mode", value=False)

# Upload up to 8 CSV files
st.markdown("### Upload CSV Files")
NUM_FILES = 8
uploaded_files = []
cols = st.columns(4)
for i in range(NUM_FILES):
    with cols[i % 4]:
        f = st.file_uploader(f"Hydrophone {i+1}", type=["csv"], key=f"file_{i}")
        uploaded_files.append(f)

hydrophone_frames = {}
for i, uploaded_file in enumerate(uploaded_files):
    if uploaded_file is not None:
        t0 = time.perf_counter()
        data = uploaded_file.getvalue()
        base_id = extract_hydrophone_id(uploaded_file.name)
        hydro_id = base_id
        suffix = 2
        while hydro_id in hydrophone_frames:
            hydro_id = f"{base_id}_{suffix}"
            suffix += 1
        df_n = process_csv(
            data=data, num=hydro_id,   # pass string ID instead of integer index
            header_opt=header_opt,
            delim_val=delim_val,
            percent_as_fraction=percent_as_fraction,
            show_raw=show_raw,
            enable_power_energy=enable_power_energy,
        )
        st.caption(f"⏱ Hydrophone {hydro_id} parsed in {time.perf_counter() - t0:.2f}s")
        hydrophone_frames[hydro_id] = df_n

if not hydrophone_frames:
    st.info("Upload a file to begin.")
    st.stop()

all_stages = list(dict.fromkeys(
    stage
    for hydro_id, frame in hydrophone_frames.items()
    if f"{hydro_id}_Status" in frame.columns
    for stage in frame[f"{hydro_id}_Status"].dropna().unique()
))

with st.sidebar:
    st.divider()
    st.subheader("Hydrophone comparison")
    selected_stages = None
    if all_stages:
        selected_stages = st.multiselect(
            "Filter by Status", options=all_stages, default=all_stages,
            help="Applied separately to each hydrophone using its own Status column. Clear all to show no stages.",
        )
    alignment = st.selectbox(
        "Align time to",
        ["Start of selected data", "Recorded time", "Start of a soak stage"],
        help="Each hydrophone uses its own time. Start of selected data makes its first visible time zero.",
    )
    anchor_stage = None
    if alignment == "Start of a soak stage":
        if all_stages:
            anchor_stage = st.selectbox("Stage at time zero", all_stages)
            st.caption("Uses the first valid occurrence in each file, even if that stage is hidden.")
        else:
            st.warning("No Status stages found. Using recorded time.")
            alignment = "Recorded time"

    st.caption("Choose a Time(h) or Time(s) X axis to compare each hydrophone on its own clock.")
    offsets = {}
    with st.expander("Manual time offsets (hours)"):
        st.caption("Negative moves a curve earlier (left); positive moves it later (right). Added after alignment.")
        for hydro_id in hydrophone_frames:
            offsets[hydro_id] = st.number_input(
                f"Hydrophone {hydro_id} offset (h)", value=0.0, step=0.1, format="%.6f",
                key=f"time_offset_{hydro_id}",
            )
    st.button("Reset time offsets", on_click=reset_time_offsets, args=(list(hydrophone_frames),))

for hydro_id, frame in hydrophone_frames.items():
    try:
        filtered, origin_hours = prepare_comparison_frame(
            frame, hydro_id, selected_stages, alignment, anchor_stage, offsets[hydro_id]
        )
    except ValueError as e:
        st.warning(f"Hydrophone {hydro_id}: {e}. Omitted from this comparison; choose another alignment.")
        filtered, origin_hours = frame.iloc[:0].copy(), None
    hydrophone_frames[hydro_id] = filtered
    with st.sidebar:
        if origin_hours is not None:
            shift_hours = offsets[hydro_id] - origin_hours
            st.caption(f"{hydro_id}: {len(filtered)} rows · total time shift {shift_hours:+.6f} h")
        elif f"{hydro_id}_Time(s)" not in frame.columns:
            st.caption(f"{hydro_id}: {len(filtered)} rows · no elapsed-time column; time alignment unavailable.")
        else:
            st.caption(f"{hydro_id}: {len(filtered)} rows · no valid time to align.")
        if selected_stages is not None and f"{hydro_id}_Status" not in frame.columns:
            st.caption(f"{hydro_id}: no Status column; stage filter unavailable for this file.")

df = pd.concat(list(hydrophone_frames.values()), axis=1)
st.caption(
    f"Comparing {len(hydrophone_frames)} files — {df.shape[0]} rows × {df.shape[1]} columns. "
    "Time axes use each hydrophone's own timestamps; other X axes compare rows by position."
)

with st.sidebar:
    st.markdown("--------")
    st.subheader("Column Scaling")
    st.caption("Multiply columns by some factor to view mismatched units easily.")

    num_scalers = st.number_input("Number of scalers", min_value=0, max_value=20, value=0, step=1)
    scaling_rules = []
    if num_scalers > 0:
        all_cols_for_scaling = list(df.columns) if df is not None else []
        for i in range(num_scalers):
            sc1, sc2 = st.columns([2, 1])
            with sc1:
                col_name = st.selectbox(f"Column {i+1}", options=[""] + all_cols_for_scaling, key=f"scale_col_{i}")
            with sc2:
                factor = st.number_input(f"x factor", value=1.0, format="%.6g", key=f"scale_factor_{i}")
            if col_name:
                scaling_rules.append((col_name, factor))

if df is not None:
    # Apply scaling rules
    for col_name, factor in scaling_rules:
        if col_name in df.columns and pd.api.types.is_numeric_dtype(df[col_name]):
            df[col_name] = df[col_name] * factor
            st.toast(f"Scaled column '{col_name}' by factor {factor}.")

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
    x_options = ["(index)"] + all_cols
    default_time = next((c for c in all_cols if c.endswith("_Time(h)")), "(index)")
    x_choice = st.selectbox("X axis", x_options, index=x_options.index(default_time))

with c2:
    y_left = st.multiselect("Left Y axis", all_cols)  # No pre-selection

with c3:
    y_right = st.multiselect("Right Y axis", all_cols)  # No pre-selection

with c4:
    kind = st.selectbox("Chart type", ["line", "scatter"], index=0)

trace_x_columns = time_axes_for_traces(hydrophone_frames, x_choice)
missing_time_traces = [c for c in y_left + y_right if c in trace_x_columns and trace_x_columns[c] is None]
if missing_time_traces:
    st.warning("No matching time column; these traces are omitted: " + ", ".join(missing_time_traces))

# Dynamically build the figure with dual axes
fig = build_plotly_dual_axis(
    df=df,
    x=x_choice,
    y_left=y_left,
    y_right=y_right,
    kind=kind,
    title="Plotly Chart",
    log_x=log_x,
    log_y_left=log_y_left,
    log_y_right=log_y_right,
    trace_x_columns=trace_x_columns,
)

# Display the plot
st.plotly_chart(fig, width='stretch', config={"displaylogo": False})

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
