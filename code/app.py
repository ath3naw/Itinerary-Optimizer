import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import optimize, clean_dataset, functions

st.title("Itinerary Optimizer for NYC Attractions 📝")
st.image("https://static.independent.co.uk/2025/03/04/14/03/iStock-2054112501.jpeg", caption="New York City")
st.write("Plan an optimized route for NY attractions based on your preferences.")
st.write("Developed by Athena Wang")
df = clean_dataset.clean_dataset()
# create map
fig = px.scatter_mapbox(df, lat='lat', lon='lon',
                        hover_name='name', # Info shown on hover
                        hover_data=['indoor', 'museum'], # More info
                        zoom=9, height=500)
fig.update_layout(mapbox_style="carto-positron") # Set map style
st.plotly_chart(fig, use_container_width=True)

# whether we calculate loop or a straight line itinerary
loop = st.radio("Do you want to start and end at the same location?", ('Yes', 'No'))

st.divider()

if loop == 'Yes':
    start_lat = st.number_input(
        "Enter your starting address latitude (N):",
        min_value=40.5000, max_value=41.0000,
        value=40.7424,
        placeholder="Between 40.5 and 41.0, e.g. 40.7424",
        format="%0.4f"
    )
    start_lon = st.number_input(
        "Enter your starting address longitude (E):",
        min_value=-74.2000, max_value=-73.7000,
        value=-74.0061,
        placeholder="Between -74.2 and -73.7, e.g. -74.0061",
        format="%0.4f"
    )

    num_locations = st.number_input(
        "Enter the number of locations to visit after starting point:",
        1, 15,
        placeholder="e.g., 9",
        format="%d"
    )
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        pass
    with col2:
        generate = st.button("Generate Itinerary", width="stretch", type="primary")
    with col3:
        pass
    if generate:
        st.divider()
        df = clean_dataset.add_start(start_lat, start_lon)
        distance_mat = functions.create_distance_matrix(df)
        itinerary, total_dist = optimize.pick_k_nearest(df, num_locations, distance_mat)
        st.subheader("Interactive Map of Itinerary")
        st.write("Can toggle types of points on/off in the legend and zoom in using top panel (+) button. Hover over points to see details.")
        functions.create_itinerary(df, itinerary, total_dist)
else:
    start_lat = st.number_input(
        "Enter your starting address latitude (N):",
        min_value=40.5000, max_value=41.0000,
        value=40.7424,
        placeholder="e.g., 40.7424",
        format="%0.4f"
    )
    start_lon = st.number_input(
        "Enter your starting address longitude (E):",
        min_value=-74.2000, max_value=-73.7000,
        value=-74.0061,
        placeholder="e.g., -74.0061",
        format="%0.4f"
    )
    end_lat = st.number_input(
        "Enter your ending address latitude (N):",
        min_value=40.5000, max_value=41.0000,
        value=40.7021,
        placeholder="e.g., 40.7021",
        format="%0.4f"
    )
    end_lon = st.number_input(
        "Enter your ending address longitude (E):",
        min_value=-74.2000, max_value=-73.7000,
        value=-73.9921,
        placeholder="e.g., -73.9921",
        format="%0.4f"
    )

    num_locations = st.number_input(
        "Enter the number of locations to visit between starting and ending points:",
        1, 15,
        placeholder="e.g., 9",
        format="%d"
    )
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        pass
    with col2:
        generate = st.button("Generate Itinerary", width="stretch", type="primary")
    with col3:
        pass
    if generate:
        st.divider()
        df = clean_dataset.add_start_end(start_lat, start_lon, end_lat, end_lon)
        distance_mat = functions.create_distance_matrix(df)
        itinerary, total_dist = optimize.greedy_dir_alg(df, num_locations, distance_mat)
        st.subheader("Interactive Map of Itinerary")
        st.write("Can toggle types of points on/off in the legend and zoom in using top panel (+) button. Hover over points to see details.")
        functions.create_itinerary(df, itinerary, total_dist)