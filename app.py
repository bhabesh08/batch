import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(
    page_title="Pipeline Batch Tracker",
    page_icon="🛤️",
    layout="wide",
)

# --- Helper Functions ---

def convert_to_m(value, unit):
    """Converts a value from a given unit to meters."""
    if unit.lower() == 'inch':
        return value * 0.0254
    if unit.lower() == 'km':
        return value * 1000
    return value

def get_distance_from_volume(volume_m3, cumulative_volumes, cumulative_lengths_km):
    """
    Calculates the distance along the pipeline for a given pumped volume.
    Returns distance in km.
    """
    total_pipeline_volume = cumulative_volumes[-1]
    total_pipeline_length = cumulative_lengths_km[-1]

    if volume_m3 <= 0:
        return 0
    if volume_m3 >= total_pipeline_volume:
        return total_pipeline_length

    # Find which section the volume falls into
    section_index = np.searchsorted(cumulative_volumes, volume_m3, side='right')

    # Volume and length of the pipeline up to the previous section
    vol_before = cumulative_volumes[section_index - 1] if section_index > 0 else 0
    len_before = cumulative_lengths_km[section_index - 1] if section_index > 0 else 0

    # Volume and length of the current section
    vol_section = cumulative_volumes[section_index] - vol_before
    len_section = cumulative_lengths_km[section_index] - len_before

    # Volume within the current section
    vol_in_section = volume_m3 - vol_before

    # Calculate distance within the current section
    dist_in_section = (vol_in_section / vol_section) * len_section if vol_section > 0 else 0

    return len_before + dist_in_section


def create_visual_representation(locations_df, pipeline_df, total_length_km):
    """Creates a Plotly figure to visualize batch locations."""
    
    # Define a color map for product types
    product_types = locations_df['Type'].unique()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    color_map = {ptype: colors[i % len(colors)] for i, ptype in enumerate(product_types)}

    fig = go.Figure()

    # Add a line for the main pipeline axis
    fig.add_shape(
        type="line",
        x0=0, y0=0, x1=total_length_km, y1=0,
        line=dict(color="Gray", width=5)
    )

    # Add rectangles for each batch
    for index, row in locations_df.iterrows():
        start_km = row['Start Location (km)']
        end_km = row['End Location (km)']
        
        if end_km > start_km: # Only draw if the batch has a length
            fig.add_shape(
                type="rect",
                x0=start_km,
                y0=-0.5,
                x1=end_km,
                y1=0.5,
                fillcolor=color_map.get(row['Type'], 'lightgrey'),
                line=dict(width=0),
                opacity=0.8,
            )
            # Add annotation for the batch
            fig.add_annotation(
                x=(start_km + end_km) / 2,
                y=0,
                text=f"<b>{row['Batch']}</b><br>({row['Type']})",
                showarrow=False,
                font=dict(color="white", size=10),
                yshift=0
            )

    # Add markers for pipeline section joints
    cumulative_length = 0
    for index, row in pipeline_df.iterrows():
        cumulative_length += row['Length (km)']
        if cumulative_length < total_length_km:
            fig.add_shape(
                type="line",
                x0=cumulative_length, y0=-0.7, x1=cumulative_length, y1=0.7,
                line=dict(color="Black", width=2, dash="dash")
            )
            fig.add_annotation(
                x=cumulative_length, y=-0.8,
                text=f"Sec {index+1}-{index+2}",
                showarrow=False,
                yshift=-10
            )

    fig.update_layout(
        title="Pipeline Batch Visualization",
        xaxis_title="Distance (km)",
        yaxis_title="",
        yaxis=dict(showticklabels=False, range=[-2, 2]),
        xaxis=dict(range=[0, total_length_km * 1.05]),
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False
    )
    return fig

# --- Main Application UI ---
st.title("🛤️ Pipeline Batch Tracker")
st.markdown("An interactive tool to simulate and track the position of different product batches within a pipeline system at a specific time.")

# --- User Inputs ---
st.header("1. Define System Parameters")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Pumping Plan")
    # Sample data for the pumping plan
    pumping_data = {
        "Volume (m3)": [1500, 2000, 1200, 2500],
        "Type": ["HSD", "ATF", "MS", "HSD"],
        "Density (kg/m3)": [830, 790, 740, 835],
        "Viscosity (cSt)": [3.5, 1.2, 0.6, 3.8],
    }
    pumping_plan_df = pd.DataFrame(pumping_data)
    pumping_plan_df.index = [f"Batch {i+1}" for i in pumping_plan_df.index]
    
    edited_pumping_plan = st.data_editor(
        pumping_plan_df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Type": st.column_config.SelectboxColumn(
                "Type",
                options=["LS", "HS", "MS", "HSD", "ATF", "Kero"],
                required=True,
            )
        }
    )

with col2:
    st.subheader("Pipeline Details")
    # Sample data for pipeline details
    pipeline_data = {
        "Diameter (inch)": [16, 14, 16],
        "Wall Thickness (inch)": [0.250, 0.250, 0.312],
        "Length (km)": [120, 80, 150],
    }
    pipeline_df = pd.DataFrame(pipeline_data)
    pipeline_df.index = [f"Section {i+1}" for i in pipeline_df.index]
    
    edited_pipeline_df = st.data_editor(
        pipeline_df,
        num_rows="dynamic",
        use_container_width=True
    )

st.header("2. Set Operational Conditions")

col3, col4 = st.columns(2)

with col3:
    flow_rate_m3h = st.number_input("Flow Rate (m3/h)", min_value=1, value=500)

with col4:
    st.subheader("Select Time")
    start_date = st.date_input("Pumping Start Date", datetime.date(2024, 7, 20))
    start_time = st.time_input("Pumping Start Time", datetime.time(6, 0))
    start_datetime = datetime.datetime.combine(start_date, start_time)

    st.markdown("---")

    current_date = st.date_input("Date for Location Check", datetime.datetime.now().date())
    current_time = st.time_input("Time for Location Check", datetime.datetime.now().time())
    current_datetime = datetime.datetime.combine(current_date, current_time)

# --- Calculation and Display ---
st.header("3. Batch Location Results")

# Validate inputs before proceeding
if edited_pumping_plan.empty or edited_pipeline_df.empty or flow_rate_m3h <= 0:
    st.warning("Please ensure the Pumping Plan, Pipeline Details, and a valid Flow Rate are provided.")
elif current_datetime < start_datetime:
    st.error("The 'Time for Location Check' cannot be before the 'Pumping Start Time'.")
else:
    try:
        # --- Pipeline Calculations ---
        pipeline_calc_df = edited_pipeline_df.copy()
        pipeline_calc_df['Inner Diameter (m)'] = convert_to_m(pipeline_calc_df['Diameter (inch)'] - 2 * pipeline_calc_df['Wall Thickness (inch)'], 'inch')
        pipeline_calc_df['Area (m2)'] = np.pi * (pipeline_calc_df['Inner Diameter (m)'] / 2)**2
        pipeline_calc_df['Volume (m3)'] = pipeline_calc_df['Area (m2)'] * convert_to_m(pipeline_calc_df['Length (km)'], 'km')
        pipeline_calc_df['Cumulative Volume (m3)'] = pipeline_calc_df['Volume (m3)'].cumsum()
        pipeline_calc_df['Cumulative Length (km)'] = pipeline_calc_df['Length (km)'].cumsum()
        
        total_pipeline_length_km = pipeline_calc_df['Length (km)'].sum()
        total_pipeline_volume_m3 = pipeline_calc_df['Volume (m3)'].sum()

        # --- Batch Location Calculations ---
        time_diff_seconds = (current_datetime - start_datetime).total_seconds()
        time_diff_hours = time_diff_seconds / 3600
        total_pumped_volume = flow_rate_m3h * time_diff_hours

        # Create a copy to avoid SettingWithCopyWarning
        batch_locations_df = edited_pumping_plan.copy()
        batch_locations_df.reset_index(inplace=True)
        batch_locations_df.rename(columns={'index': 'Batch'}, inplace=True)
        
        batch_locations_df['Cumulative Volume (m3)'] = batch_locations_df['Volume (m3)'].cumsum()

        locations = []
        
        for i, batch in batch_locations_df.iterrows():
            vol_at_batch_end = total_pumped_volume - (batch['Cumulative Volume (m3)'] - batch['Volume (m3)'])
            vol_at_batch_start = total_pumped_volume - batch['Cumulative Volume (m3)']

            end_km = get_distance_from_volume(
                vol_at_batch_end,
                pipeline_calc_df['Cumulative Volume (m3)'].values,
                pipeline_calc_df['Cumulative Length (km)'].values
            )
            
            start_km = get_distance_from_volume(
                vol_at_batch_start,
                pipeline_calc_df['Cumulative Volume (m3)'].values,
                pipeline_calc_df['Cumulative Length (km)'].values
            )
            
            locations.append({
                "Batch": batch['Batch'],
                "Type": batch['Type'],
                "Start Location (km)": round(start_km, 2),
                "End Location (km)": round(end_km, 2),
                "Length in Pipe (km)": round(end_km - start_km, 2)
            })

        final_locations_df = pd.DataFrame(locations)

        # --- Display Results ---
        st.subheader("Visual Representation")
        st.info(f"Showing batch locations for **{current_datetime.strftime('%Y-%m-%d %H:%M')}**. "
                f"Total pumped volume: **{total_pumped_volume:,.2f} m³**.")

        # Filter out batches that haven't entered the pipeline yet
        visible_batches_df = final_locations_df[final_locations_df['End Location (km)'] > 0]
        
        if visible_batches_df.empty:
            st.warning("No batches have entered the pipeline at the selected time.")
        else:
            fig = create_visual_representation(visible_batches_df, pipeline_calc_df, total_pipeline_length_km)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Location Data Table")
        st.dataframe(final_locations_df, use_container_width=True, hide_index=True)

        # --- Expander for Detailed Calculations ---
        with st.expander("View Detailed Pipeline Calculations"):
            st.dataframe(pipeline_calc_df)
            st.markdown(f"**Total Pipeline Length:** {total_pipeline_length_km:,.2f} km")
            st.markdown(f"**Total Pipeline Volume:** {total_pipeline_volume_m3:,.2f} m³")

    except Exception as e:
        st.error(f"An error occurred during calculation: {e}")
        st.error("Please check your input data. All numerical columns must contain valid numbers.")

