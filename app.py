# streamlit_app.py
import math
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import altair as alt
import streamlit as st

st.set_page_config(page_title="Pipeline Batch Tracker", layout="wide")

# -----------------------------
# Helpers
# -----------------------------
INCH_TO_M = 0.0254
KM_TO_M = 1_000.0

BATCH_TYPES = ["LS", "HS", "MS", "HSD", "ATF"]
TYPE_COLORS = {
    "LS": "#1f77b4",
    "HS": "#ff7f0e",
    "MS": "#2ca02c",
    "HSD": "#d62728",
    "ATF": "#9467bd",
}

def sanitize_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def build_pipeline_profile(sections_df: pd.DataFrame):
    """
    From sections table:
      - Compute internal diameter (m) = (diameter - 2*wall) * 0.0254
      - Area (m^2) = pi * (ID/2)^2
      - Length (m) = length_km * 1000
      - Capacity per section (m^3) = Area * Length
    Returns dict with per-section arrays and cumulative profiles.
    """
    df = sections_df.copy()

    # Ensure numeric
    df = sanitize_numeric(df, ["diameter_in", "wall_in", "length_km"]).fillna(0.0)

    df["id_m"] = (df["diameter_in"] - 2.0 * df["wall_in"]) * INCH_TO_M
    df["id_m"] = df["id_m"].clip(lower=0.0)  # Avoid negative/NaN
    df["area_m2"] = math.pi * (df["id_m"] / 2.0) ** 2
    df["length_m"] = df["length_km"] * KM_TO_M
    df["capacity_m3"] = df["area_m2"] * df["length_m"]

    # Guard against zero-area sections
    df["capacity_m3"] = df["capacity_m3"].fillna(0.0).clip(lower=0.0)

    capacities = df["capacity_m3"].to_numpy()
    areas = df["area_m2"].to_numpy()
    lengths_m = df["length_m"].to_numpy()

    cum_cap = np.concatenate([[0.0], np.cumsum(capacities)])
    cum_len_m = np.concatenate([[0.0], np.cumsum(lengths_m)])

    profile = {
        "df": df,
        "areas": areas,
        "capacities": capacities,
        "lengths_m": lengths_m,
        "cum_cap": cum_cap,      # m^3, length n_sections+1
        "cum_len_m": cum_len_m,  # m, length n_sections+1
        "total_cap": float(cum_cap[-1]),
        "total_len_km": float(cum_len_m[-1] / KM_TO_M),
    }
    return profile

def volume_to_distance_km(V: float, profile: dict) -> float:
    """
    Convert a cumulative displaced volume V (m^3, measured from inlet) to distance (km) from inlet.
    Piecewise inversion of the cumulative capacity vs. distance.
    """
    Vcap = profile["total_cap"]
    if V <= 0.0:
        return 0.0
    if V >= Vcap and Vcap > 0:
        return profile["total_len_km"]
    if Vcap == 0:
        return 0.0

    cum_cap = profile["cum_cap"]
    cum_len_m = profile["cum_len_m"]
    areas = profile["areas"]

    # Find section k such that cum_cap[k] <= V < cum_cap[k+1]
    k = int(np.searchsorted(cum_cap, V, side="right") - 1)
    k = max(0, min(k, len(areas) - 1))
    V_prev = cum_cap[k]
    A_k = areas[k] if areas[k] > 0 else 1e-12  # avoid zero division
    dx_m = (V - V_prev) / A_k
    dist_m = cum_len_m[k] + dx_m
    return max(0.0, min(dist_m / KM_TO_M, profile["total_len_km"]))

def batch_intervals_by_time(batches_df: pd.DataFrame, profile: dict, flow_m3h: float,
                            start_dt: datetime, selected_dt: datetime):
    """
    Compute, for each batch, its in-pipeline interval [start_km, end_km] at selected time.
    """
    # Ensure time difference in hours (non-negative)
    dt_hours = max(0.0, (selected_dt - start_dt).total_seconds() / 3600.0)
    V_disp = max(0.0, flow_m3h) * dt_hours  # m^3 displaced since start

    # Prepare batches
    df = batches_df.copy()
    df = sanitize_numeric(df, ["volume_m3", "density_kgm3", "visc_cSt"]).fillna(0.0)

    # Keep only positive-volume batches in stated order
    df = df[df["volume_m3"] > 0].reset_index(drop=True)

    volumes = df["volume_m3"].to_numpy()
    types = df["type"].astype(str).replace({np.nan: ""}).to_numpy()

    # Precompute cumulative volumes at t=0 (injection order)
    cum_at_0 = np.concatenate([[0.0], np.cumsum(volumes)])

    Vcap = profile["total_cap"]

    rows = []
    for i in range(len(df)):
        V_i = volumes[i]
        typ = types[i]

        a_i = cum_at_0[i] + V_disp         # left boundary (m^3 from inlet)
        b_i = a_i + V_i                    # right boundary

        # Overlap with pipeline window [0, Vcap]
        L = max(0.0, a_i)
        U = min(Vcap, b_i)

        in_pipeline = max(0.0, U - L)
        pct_in = 100.0 * (in_pipeline / V_i) if V_i > 0 else 0.0

        if in_pipeline > 0.0 and Vcap > 0.0:
            start_km = volume_to_distance_km(L, profile)
            end_km = volume_to_distance_km(U, profile)
        else:
            start_km, end_km = None, None

        if a_i >= Vcap and Vcap > 0:
            status = "Delivered"
        elif b_i <= 0:
            status = "Not entered"
        elif in_pipeline > 0.0:
            status = "In pipeline"
        else:
            # This case can occur if Vcap==0
            status = "N/A"

        rows.append({
            "batch_no": i + 1,
            "type": typ,
            "volume_m3": V_i,
            "start_km": start_km,
            "end_km": end_km,
            "center_km": None if (start_km is None or end_km is None) else 0.5 * (start_km + end_km),
            "in_pipeline_%": round(pct_in, 2),
            "status": status,
        })

    out = pd.DataFrame(rows)
    # Ensure numerical columns well-formatted
    for c in ["start_km", "end_km", "center_km"]:
        if c in out:
            out[c] = out[c].astype(float).round(3)
    return out

def section_boundaries_df(profile: dict) -> pd.DataFrame:
    """Create a dataframe of section boundary distances (km) for plotting."""
    km = profile["cum_len_m"] / KM_TO_M
    return pd.DataFrame({"boundary_km": km[1:-1]})  # exclude 0 and total

# -----------------------------
# Defaults
# -----------------------------
default_batches = pd.DataFrame({
    "volume_m3": [5000, 6000, 4000],
    "type": ["HSD", "MS", "ATF"],
    "density_kgm3": [830, 750, 780],
    "visc_cSt": [3.0, 1.2, 1.5],
})

default_sections = pd.DataFrame({
    "section": [1, 2, 3],
    "diameter_in": [24.0, 24.0, 18.0],
    "wall_in": [0.5, 0.5, 0.375],
    "length_km": [120.0, 80.0, 60.0],
})

# -----------------------------
# UI
# -----------------------------
st.title("Pipeline Batch Tracker")

with st.expander("Instructions", expanded=False):
    st.markdown(
        "- **Batches table:** Add rows in the order of injection. Provide volume (m3), type, density, and viscosity.\n"
        "- **Pipeline sections:** Provide diameter and wall thickness (inch) and section length (km). Internal diameter is computed.\n"
        "- **Timing:** Set pumping start and a target time. Flow is assumed constant.\n"
        "- The visualization shows each batch's span along the pipeline in km."
    )

colA, colB = st.columns([1.2, 1.0], gap="large")

with colA:
    st.subheader("Pumping plan in batches")
    batches_df = st.data_editor(
        default_batches,
        key="batches",
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "volume_m3": st.column_config.NumberColumn("volume (m3)", min_value=0.0, step=100.0, format="%.0f"),
            "type": st.column_config.SelectboxColumn("type", options=BATCH_TYPES),
            "density_kgm3": st.column_config.NumberColumn("density (kg/m3)", min_value=0.0, step=1.0, format="%.0f"),
            "visc_cSt": st.column_config.NumberColumn("viscosity (cSt)", min_value=0.0, step=0.1, format="%.1f"),
        }
    )

with colB:
    st.subheader("Pipeline sections")
    sections_df = st.data_editor(
        default_sections,
        key="sections",
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "section": st.column_config.NumberColumn("section", disabled=True),
            "diameter_in": st.column_config.NumberColumn("diameter (inch)", min_value=0.0, step=0.5, format="%.3f"),
            "wall_in": st.column_config.NumberColumn("wall thickness (inch)", min_value=0.0, step=0.125, format="%.3f"),
            "length_km": st.column_config.NumberColumn("length (km)", min_value=0.0, step=1.0, format="%.1f"),
        }
    )

st.subheader("Flow and time")
col1, col2, col3 = st.columns([1, 1, 1.2], gap="large")
with col1:
    flow_m3h = st.number_input("Flow rate (m3/h)", min_value=0.0, value=1500.0, step=50.0, format="%.1f")
with col2:
    start_dt = st.datetime_input("Pumping start datetime", value=datetime.now(timezone.utc).astimezone().replace(second=0, microsecond=0))
with col3:
    selected_dt = st.datetime_input("Show positions at datetime", value=datetime.now(timezone.utc).astimezone().replace(second=0, microsecond=0))

# -----------------------------
# Calculations
# -----------------------------
profile = build_pipeline_profile(sections_df)

# Guardrail: If pipeline has zero capacity, warn
if profile["total_cap"] <= 0.0:
    st.warning("Pipeline capacity is zero. Check diameters, wall thicknesses, and lengths.")
else:
    positions_df = batch_intervals_by_time(batches_df, profile, flow_m3h, start_dt, selected_dt)

    # Summary KPIs
    k1, k2, k3 = st.columns(3)
    with k1:
        st.metric("Total pipeline length", f"{profile['total_len_km']:.2f} km")
    with k2:
        st.metric("Pipeline capacity", f"{profile['total_cap']:.0f} m³")
    with k3:
        dt_hours = max(0.0, (selected_dt - start_dt).total_seconds() / 3600.0)
        st.metric("Displaced volume since start", f"{flow_m3h * dt_hours:.0f} m³")

    st.divider()

    st.subheader("Batch locations at selected time")
    st.dataframe(
        positions_df[["batch_no", "type", "volume_m3", "start_km", "end_km", "center_km", "in_pipeline_%", "status"]],
        use_container_width=True
    )

    # -----------------------------
    # Visualization
    # -----------------------------
    st.subheader("Visual representation along pipeline")

    # Build segments for batches currently in pipeline
    seg_rows = []
    for _, r in positions_df.iterrows():
        if pd.notna(r["start_km"]) and pd.notna(r["end_km"]) and (r["end_km"] > r["start_km"]):
            seg_rows.append({
                "Batch": f"Batch {int(r['batch_no'])} ({r['type']})",
                "type": r["type"],
                "x_start": float(r["start_km"]),
                "x_end": float(r["end_km"]),
                "length_km": float(r["end_km"] - r["start_km"]),
            })

    seg_df = pd.DataFrame(seg_rows)

    # Base axis scale
    x_scale = alt.Scale(domain=[0, max(0.001, profile["total_len_km"])])

    # If there are segments, plot them; else show empty axis
    if not seg_df.empty:
        # Color scheme by type
        seg_df["color"] = seg_df["type"].map(TYPE_COLORS).fillna("#555")

        bars = alt.Chart(seg_df).mark_bar(height=18).encode(
            x=alt.X("x_start:Q", title="Distance from inlet (km)", scale=x_scale),
            x2="x_end:Q",
            y=alt.Y("Batch:N", sort=seg_df["Batch"].tolist(), title=None),
            color=alt.Color("type:N", scale=alt.Scale(domain=list(TYPE_COLORS.keys()),
                                                      range=list(TYPE_COLORS.values())),
                            legend=alt.Legend(title="Type")),
            tooltip=[
                alt.Tooltip("Batch:N"),
                alt.Tooltip("type:N", title="Type"),
                alt.Tooltip("x_start:Q", title="Start (km)", format=".2f"),
                alt.Tooltip("x_end:Q", title="End (km)", format=".2f"),
                alt.Tooltip("length_km:Q", title="Span (km)", format=".2f"),
            ],
        )

        # Section boundaries as vertical rules
        boundaries = section_boundaries_df(profile)
        rules = alt.Chart(boundaries).mark_rule(color="#999", strokeDash=[4, 4]).encode(
            x=alt.X("boundary_km:Q", scale=x_scale),
            tooltip=[alt.Tooltip("boundary_km:Q", title="Section boundary (km)", format=".2f")]
        )

        chart = (bars + rules).properties(height=max(80, 26 * len(seg_df) + 20), width=900)
        st.altair_chart(chart, use_container_width=True)
    else:
        # Empty baseline chart
        baseline = pd.DataFrame({"x": [0, profile["total_len_km"]], "y": ["Pipeline", "Pipeline"]})
        chart = alt.Chart(baseline).mark_line().encode(
            x=alt.X("x:Q", title="Distance from inlet (km)", scale=x_scale),
            y=alt.Y("y:N", title=None)
        ).properties(height=80, width=900)
        st.altair_chart(chart, use_container_width=True)

    # Notes
    with st.expander("Assumptions and notes", expanded=False):
        st.markdown(
            "- **Constant flow** is assumed between start and selected times; no pump stops or transients.\n"
            "- **Positions are volume-based**: batch boundaries move by displaced volume; conversion to distance accounts for varying diameters per section.\n"
            "- **Internal diameter** is computed as (diameter − 2 × wall). Units: inch to meters.\n"
            "- Batches beyond the outlet are marked Delivered; partial segments are clipped to the pipeline length."
        )
