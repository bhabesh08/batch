# app.py
import math
from datetime import datetime, timedelta
from typing import List, Tuple

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Pipeline Batch Tracker", layout="wide")

st.title("Pipeline Batch Tracker")
st.caption("Compute batch locations along a multi-section pipeline given a pumping plan, pipeline geometry, "
           "flow rate, and start/view times. Assumes plug flow, incompressible fluids, and inner diameter = "
           "diameter − 2×wall thickness (inches).")

# -----------------------------
# Helpers
# -----------------------------
INCH_TO_M = 0.0254
KM_TO_M = 1000.0

BATCH_TYPES = ["LS", "HS", "MS", "HSD", "ATF"]

def sanitize_positive(x, name):
    try:
        v = float(x)
        return max(v, 0.0)
    except Exception:
        st.warning(f"Non-numeric value in {name}; treating as 0.")
        return 0.0

def sections_from_df(df_sec: pd.DataFrame):
    """Compute per-section inner diameter, area, length, volume, and cumulative arrays."""
    # Expect columns: diameter_inch, wall_inch, length_km
    di_in = df_sec["diameter (inch)"].astype(float).to_numpy()
    wt_in = df_sec["wall thickness (inch)"].astype(float).to_numpy()
    L_km  = df_sec["length (km)"].astype(float).to_numpy()

    # Inner diameter in meters (ensure non-negative)
    ID_m = (di_in - 2.0 * wt_in) * INCH_TO_M
    ID_m = np.where(ID_m > 0, ID_m, 0.0)

    A_m2 = (np.pi / 4.0) * ID_m**2
    L_m  = L_km * KM_TO_M
    V_m3 = A_m2 * L_m

    cumV = np.concatenate(([0.0], np.cumsum(V_m3)))
    cumL = np.concatenate(([0.0], np.cumsum(L_m)))

    return {
        "ID_m": ID_m,
        "A_m2": A_m2,
        "L_m": L_m,
        "V_m3": V_m3,
        "cumV": cumV,
        "cumL": cumL,
        "V_total": V_m3.sum(),
        "L_total": L_m.sum(),
    }

def dist_from_local_volume(local_v: np.ndarray, A_m2: np.ndarray, cumV: np.ndarray, cumL: np.ndarray) -> np.ndarray:
    """
    Map local volume measured from inlet (0 .. V_total) to distance along pipeline (m).
    Piecewise constant area per section.
    """
    # Clip within [0, V_total]
    V_total = cumV[-1]
    v = np.clip(local_v, 0.0, V_total)

    # Find section index for each v (cumV[k] <= v < cumV[k+1])
    idx = np.searchsorted(cumV[1:], v, side="right")
    idx = np.clip(idx, 0, len(A_m2) - 1)

    # Volume remaining inside section
    v_in_sec = v - cumV[idx]
    # Distance offset inside section: delta_x = v_in_sec / A
    with np.errstate(divide="ignore", invalid="ignore"):
        dx = np.where(A_m2[idx] > 0, v_in_sec / A_m2[idx], 0.0)

    return cumL[idx] + dx

def compute_batch_segments_in_pipe(
    batches_df: pd.DataFrame,
    V_pipe: float,
    V_displaced: float,
    A_m2: np.ndarray,
    cumV_sec: np.ndarray,
    cumL_sec: np.ndarray
):
    """
    For each batch (with start/end in pumped-volume axis), compute the overlap with the in-pipe window
    [Vd - Vpipe, Vd], map to distances, and return per-batch segment info.
    """
    # Pumped-volume intervals for each batch
    vols = batches_df["volume (m3)"].astype(float).clip(lower=0.0).to_numpy()
    types = batches_df["type"].astype(str).to_numpy()
    dens  = batches_df["density (kg/m3)"].astype(float).to_numpy()
    visc  = batches_df["viscosity (cSt)"].astype(float).to_numpy()

    v_starts = np.concatenate(([0.0], np.cumsum(vols)[:-1]))
    v_ends   = np.cumsum(vols)

    # In-pipe window
    win_start = max(V_displaced - V_pipe, 0.0)
    win_end   = max(min(V_displaced, v_ends[-1] if len(v_ends) else 0.0), 0.0)

    segments = []
    rows = []

    for i, (v0, v1, t, d, mu) in enumerate(zip(v_starts, v_ends, types, dens, visc)):
        # Compute overlap with in-pipe window
        ov_start = max(v0, win_start)
        ov_end   = min(v1, win_end)
        inside_vol = max(ov_end - ov_start, 0.0)

        if inside_vol > 0:
            # Map overlap to distances
            local_start = ov_start - win_start  # 0 at inlet
            local_end   = ov_end - win_start
            x0 = float(dist_from_local_volume(np.array([local_start]), A_m2, cumV_sec, cumL_sec)[0])
            x1 = float(dist_from_local_volume(np.array([local_end]),   A_m2, cumV_sec, cumL_sec)[0])
            x_start, x_end = (min(x0, x1), max(x0, x1))
            L_seg = x_end - x_start

            segments.append({
                "batch_index": i + 1,
                "type": t,
                "start_m": x_start,
                "end_m": x_end,
                "length_m": L_seg,
                "density_kgm3": d,
                "visc_cSt": mu,
                "inside_vol_m3": inside_vol,
            })

            rows.append({
                "Batch": i + 1,
                "Type": t,
                "Total volume (m3)": vols[i],
                "In-pipe volume (m3)": inside_vol,
                "Start (km)": x_start / KM_TO_M,
                "End (km)": x_end / KM_TO_M,
                "Center (km)": (x_start + x_end) / (2.0 * KM_TO_M),
                "Percent inside (%)": 100.0 * inside_vol / vols[i] if vols[i] > 0 else 0.0,
                "Status": "Inside",
            })
        else:
            # Status outside the pipe
            status = "Not yet entered" if v1 <= win_start else "Already passed"
            rows.append({
                "Batch": i + 1,
                "Type": t,
                "Total volume (m3)": vols[i],
                "In-pipe volume (m3)": 0.0,
                "Start (km)": None,
                "End (km)": None,
                "Center (km)": None,
                "Percent inside (%)": 0.0,
                "Status": status,
            })

    seg_df = pd.DataFrame(segments)
    loc_df = pd.DataFrame(rows)
    return seg_df, loc_df

def default_batches():
    return pd.DataFrame([
        {"volume (m3)": 1200.0, "type": "HSD", "density (kg/m3)": 830.0, "viscosity (cSt)": 3.0},
        {"volume (m3)": 900.0,  "type": "MS",  "density (kg/m3)": 740.0, "viscosity (cSt)": 1.0},
        {"volume (m3)": 1500.0, "type": "ATF", "density (kg/m3)": 800.0, "viscosity (cSt)": 1.5},
    ])

def default_sections():
    return pd.DataFrame([
        {"pipeline section": "Section 1", "diameter (inch)": 18.0, "wall thickness (inch)": 0.375, "length (km)": 40.0},
        {"pipeline section": "Section 2", "diameter (inch)": 16.0, "wall thickness (inch)": 0.344, "length (km)": 60.0},
        {"pipeline section": "Section 3", "diameter (inch)": 14.0, "wall thickness (inch)": 0.312, "length (km)": 50.0},
    ])

# -----------------------------
# Sidebar inputs
# -----------------------------
with st.sidebar:
    st.header("Operating inputs")

    flow_rate = st.number_input("Flow rate (m3/h)", min_value=0.0, value=800.0, step=50.0, format="%.2f")

    # Datetime inputs
    start_dt = st.datetime_input("Start date & time", value=datetime.now() - timedelta(hours=2))
    view_dt  = st.datetime_input("View date & time",  value=datetime.now())

    if view_dt < start_dt:
        st.info("View time is before start time. No batches in pipe yet.")
    elapsed_h = max((view_dt - start_dt).total_seconds() / 3600.0, 0.0)

# -----------------------------
# Main inputs: tables
# -----------------------------
st.subheader("Pumping plan (batches)")
batch_df = st.data_editor(
    default_batches(),
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "volume (m3)": st.column_config.NumberColumn("volume (m3)", min_value=0.0, step=10.0, format="%.2f"),
        "type": st.column_config.SelectboxColumn("type", options=BATCH_TYPES),
        "density (kg/m3)": st.column_config.NumberColumn("density (kg/m3)", min_value=0.0, step=1.0, format="%.1f"),
        "viscosity (cSt)": st.column_config.NumberColumn("viscosity (cSt)", min_value=0.0, step=0.1, format="%.2f"),
    },
    key="batches",
)

st.subheader("Pipeline details (sections)")
sec_df = st.data_editor(
    default_sections(),
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "pipeline section": st.column_config.TextColumn("pipeline section"),
        "diameter (inch)": st.column_config.NumberColumn("diameter (inch)", min_value=0.0, step=0.125, format="%.3f"),
        "wall thickness (inch)": st.column_config.NumberColumn("wall thickness (inch)", min_value=0.0, step=0.01, format="%.3f"),
        "length (km)": st.column_config.NumberColumn("length (km)", min_value=0.0, step=1.0, format="%.2f"),
    },
    key="sections",
)

# Validate basic inputs
if batch_df.empty or sec_df.empty:
    st.warning("Please provide both a pumping plan and pipeline sections.")
    st.stop()

# -----------------------------
# Calculations
# -----------------------------
# Sections geometry
ge = sections_from_df(sec_df)
V_pipe = ge["V_total"]
L_pipe = ge["L_total"]

# Displaced volume at view time
V_displaced = flow_rate * elapsed_h

# Batch segments currently inside the pipe
seg_df, loc_df = compute_batch_segments_in_pipe(
    batch_df, V_pipe, V_displaced, ge["A_m2"], ge["cumV"], ge["cumL"]
)

# -----------------------------
# KPIs
# -----------------------------
kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("Pipeline length", f"{L_pipe / KM_TO_M:.2f} km")
kpi2.metric("Pipeline capacity", f"{V_pipe:,.0f} m³")
kpi3.metric("Displaced volume", f"{V_displaced:,.0f} m³")

# -----------------------------
# Locations table
# -----------------------------
st.subheader("Batch locations at selected time")
st.dataframe(loc_df, use_container_width=True)

# -----------------------------
# Visualization
# -----------------------------
st.subheader("Pipeline view")

# Base dataframe for plotting pipeline extent
pipeline_base = pd.DataFrame({
    "start_km": [0.0],
    "end_km": [L_pipe / KM_TO_M],
    "Type": ["Pipeline"],
})

# Convert segments to km for plotting
if not seg_df.empty:
    plot_df = seg_df.copy()
    plot_df["start_km"] = plot_df["start_m"] / KM_TO_M
    plot_df["end_km"]   = plot_df["end_m"]   / KM_TO_M
    plot_df["Batch"]    = plot_df["batch_index"]
    plot_df["Type"]     = plot_df["type"]
else:
    plot_df = pd.DataFrame(columns=["start_km", "end_km", "Batch", "Type"])

# Define color scheme for types
color_scale = alt.Scale(
    domain=["HSD", "MS", "ATF", "HS", "LS"],
    range=["#3b82f6", "#ef4444", "#22c55e", "#a855f7", "#f59e0b"],
)

# Pipeline base bar
pipeline_chart = alt.Chart(pipeline_base).mark_bar(color="#e5e7eb").encode(
    x="start_km:Q",
    x2="end_km:Q",
    y=alt.value(20),
    tooltip=[alt.Tooltip("end_km:Q", title="Length (km)", format=".2f")]
)

# Batch segments bars
if not plot_df.empty:
    seg_chart = alt.Chart(plot_df).mark_bar().encode(
        x=alt.X("start_km:Q", title="Distance along pipeline (km)", scale=alt.Scale(domain=[0, L_pipe / KM_TO_M])),
        x2="end_km:Q",
        y=alt.value(20),
        color=alt.Color("Type:N", scale=color_scale),
        tooltip=[
            alt.Tooltip("Batch:Q"),
            alt.Tooltip("Type:N"),
            alt.Tooltip("start_km:Q", title="Start (km)", format=".2f"),
            alt.Tooltip("end_km:Q", title="End (km)", format=".2f"),
            alt.Tooltip("length_m:Q", title="Length (m)", format=".0f"),
        ],
    )
    chart = (pipeline_chart + seg_chart).properties(height=80, width=900)
else:
    chart = pipeline_chart.properties(height=80, width=900)

st.altair_chart(chart, use_container_width=True)

# -----------------------------
# Section markers (optional)
# -----------------------------
with st.expander("Show pipeline section markers"):
    # Create markers at section boundaries
    boundaries_km = ge["cumL"] / KM_TO_M
    marks = pd.DataFrame({"km": boundaries_km})
    rule = alt.Chart(marks).mark_rule(color="#9ca3af", strokeDash=[4,4]).encode(x="km:Q")
    st.altair_chart((chart + rule).properties(height=100, width=900), use_container_width=True)

# -----------------------------
# Notes
# -----------------------------
st.markdown("""
- Assumes plug flow with sharp interfaces, no mixing or slippage.
- Inner diameter is computed as `ID = diameter − 2 × wall thickness` (inches). Ensure inputs reflect OD and wall thickness consistently.
- If view time is early, part of the line may be unfilled; if late, some batches may have already passed.
""")
