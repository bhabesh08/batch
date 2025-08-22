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
batch_rows = st.number_input("Number of batches:", min_value=1, step=1, value=2)

batch_data = []
for i in range(batch_rows):
    st.markdown(f"### Batch {i+1}")
    vol = st.number_input(f"Volume (m³) - Batch {i+1}", min_value=0.0, step=10.0, key=f"vol_{i}")
    typ = st.selectbox(f"Type - Batch {i+1}", ["LS", "HS", "MS", "HSD", "ATF"], key=f"typ_{i}")
    dens = st.number_input(f"Density (kg/m³) - Batch {i+1}", min_value=0.0, step=10.0, key=f"dens_{i}")
    visc = st.number_input(f"Viscosity (cSt) - Batch {i+1}", min_value=0.0, step=1.0, key=f"visc_{i}")
    time_hr = vol / flow_rate if flow_rate > 0 else 0
    batch_data.append([vol, typ, dens, visc, time_hr])

batch_df = pd.DataFrame(batch_data, columns=["Volume (m³)", "Type", "Density (kg/m³)", "Viscosity (cSt)", "Time (h)"])

st.write("### Pumping Plan Table")
st.dataframe(batch_df)

if not batch_df.empty:
    total_time = batch_df["Time (h)"].sum()
    st.info(f"⏱️ Total Pumping Time: {total_time:.2f} hours")

# Input 3: Pipeline details
st.subheader("Pipeline Details")
pipe_sections = st.number_input("Number of pipeline sections:", min_value=1, step=1, value=3)

pipe_data = []
for j in range(pipe_sections):
    st.markdown(f"### Section {j+1}")
    dia = st.number_input(f"Diameter (inch) - Section {j+1}", min_value=0.0, step=0.5, key=f"dia_{j}")
    thk = st.number_input(f"Wall Thickness (inch) - Section {j+1}", min_value=0.0, step=0.1, key=f"thk_{j}")
    length = st.number_input(f"Length (km) - Section {j+1}", min_value=0.0, step=1.0, key=f"len_{j}")

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

    pipe_data.append([dia, thk, length, vol_pipe, dp/1e5])  # store pressure in bar approx

pipe_df = pd.DataFrame(pipe_data, columns=["Diameter (inch)", "Wall Thickness (inch)", "Length (km)", "Volume (m³)", "ΔP (bar)"])

st.write("### Pipeline Table")
st.dataframe(pipe_df)

if not pipe_df.empty:
    total_pipe_vol = pipe_df["Volume (m³)"].sum()
    st.info(f"🛢️ Total Pipeline Volume: {total_pipe_vol:.2f} m³")
    total_dp = pipe_df["ΔP (bar)"].sum()
    st.info(f"📉 Estimated Total Pressure Drop: {total_dp:.2f} bar")

# Visual Representation
st.subheader("📊 Visual Representation")

if not batch_df.empty:
    st.bar_chart(batch_df.set_index("Type")["Volume (m³)"])

if not pipe_df.empty:
    st.line_chart(pipe_df["Length (km)"])

st.success("✅ Data captured and calculations done!")
