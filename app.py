import streamlit as st
import pandas as pd
import numpy as np
import math

st.set_page_config(page_title="Pipeline Batch Tracker", layout="wide")

st.title("🚀 Pipeline Batch Tracker")

# Input 1: Flow rate
flow_rate = st.number_input("Enter Flow Rate (m³/h):", min_value=0.0, step=10.0)

# Input 2: Pumping Plan (Batches)
st.subheader("Batch Pumping Plan")
batch_rows = st.number_input("Number of batches:", min_value=1, step=1, value=3)

# Default batch data
default_batches = pd.DataFrame({
    "Batch": [f"Batch {i+1}" for i in range(batch_rows)],
    "Volume (m³)": [0.0]*batch_rows,
    "Type": ["LS"]*batch_rows,
    "Density (kg/m³)": [850.0]*batch_rows,
    "Viscosity (cSt)": [1.0]*batch_rows
})

# Editable batch table
batch_df = st.data_editor(default_batches, num_rows="dynamic")

# Add pumping time calculation
if not batch_df.empty and flow_rate > 0:
    batch_df["Time (h)"] = batch_df["Volume (m³)"] / flow_rate
else:
    batch_df["Time (h)"] = 0

if not batch_df.empty:
    total_time = batch_df["Time (h)"].sum()
    st.info(f"⏱️ Total Pumping Time: {total_time:.2f} hours")

# Input 3: Pipeline details
st.subheader("Pipeline Details")
pipe_sections = st.number_input("Number of pipeline sections:", min_value=1, step=1, value=3)

# Default pipeline data
default_pipes = pd.DataFrame({
    "Section": [f"Section {j+1}" for j in range(pipe_sections)],
    "Diameter (inch)": [24.0]*pipe_sections,
    "Wall Thickness (inch)": [0.5]*pipe_sections,
    "Length (km)": [10.0]*pipe_sections
})

# Editable pipeline table
pipe_df = st.data_editor(default_pipes, num_rows="dynamic")

# Calculate pipeline properties
pipe_results = []
for _, row in pipe_df.iterrows():
    dia = row["Diameter (inch)"]
    thk = row["Wall Thickness (inch)"]
    length = row["Length (km)"]

    # Convert to meters
    dia_m = dia * 0.0254
    length_m = length * 1000

    # Volume in cubic meters
    vol_pipe = math.pi * (dia_m/2)**2 * length_m

    # Average density from batches (fallback if empty)
    avg_density = batch_df["Density (kg/m³)"].mean() if not batch_df.empty else 850

    # Flow velocity
    area = math.pi * (dia_m/2)**2
    flow_m3s = flow_rate / 3600 if flow_rate > 0 else 0
    velocity = flow_m3s / area if area > 0 else 0

    # Pressure drop (simplified)
    f = 0.02  # assumed friction factor
    dp = f * (length_m/dia_m) * (avg_density * velocity**2 / 2) if dia_m > 0 else 0

    pipe_results.append([dia, thk, length, vol_pipe, dp/1e5])  # store pressure in bar approx

pipe_results_df = pd.DataFrame(pipe_results, columns=["Diameter (inch)", "Wall Thickness (inch)", "Length (km)", "Volume (m³)", "ΔP (bar)"])

st.write("### Pipeline Table with Calculations")
st.dataframe(pipe_results_df)

if not pipe_results_df.empty:
    total_pipe_vol = pipe_results_df["Volume (m³)"].sum()
    st.info(f"🛢️ Total Pipeline Volume: {total_pipe_vol:.2f} m³")
    total_dp = pipe_results_df["ΔP (bar)"].sum()
    st.info(f"📉 Estimated Total Pressure Drop: {total_dp:.2f} bar")

# Visual Representation
st.subheader("📊 Visual Representation")

if not batch_df.empty:
    st.bar_chart(batch_df.set_index("Batch")["Volume (m³)"])

if not pipe_results_df.empty:
    st.line_chart(pipe_results_df["Length (km)"])

st.success("✅ Data captured and calculations done!")
