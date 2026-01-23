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
expanded = False
df_selected = df.copy()

# User preference section
st.subheader("Your Preferences")
if "selected_types" not in st.session_state:
    st.session_state.selected_types = []

# User picks types of museums they are interested in
st.write("Select museum types you are interested in:")

types = sorted(df['type'].unique())
cols = st.columns(3)
selected_types = []
for i, t in enumerate(types):
    with cols[i % 3]:
        if st.checkbox(t, key=f"type_{t}"):
            selected_types.append(t)

# if did not select anything, show warning
if selected_types == []:
    st.write("Please select at least one type of attraction.")
# if selected types chosen, move on to next step
else:
    st.divider()
    st.session_state.selected_types = selected_types
    selected_categories = set()
    if st.session_state.selected_types:
        expanded = False
        for t in st.session_state.selected_types:
            cats = sorted(
                    {cat for cats in df[df['type']==t]['category'] if isinstance(cats, list) for cat in cats}
                )
            if cats != []:
                st.write(f"Select the themes of {t} museums you are interested in:")
                expanded = True
            cols = st.columns(3)
            for i, cat in enumerate(cats):
                with cols[i % 3]:
                    if st.checkbox(cat, key=f"cat_{t}_{cat}"):
                        selected_categories.add(cat)
    df_selected = df[
        (df['type'].isin(st.session_state.selected_types)) &
        (
            df['category'].isna() |
            df['category'].apply(
                lambda cats: isinstance(cats, list) and bool(set(cats) & selected_categories)
            )
        )
    ]
if expanded:
    st.divider()
st.subheader("Your Inputs")
enough = True
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
    if df_selected.shape[0] < num_locations:
        enough = False
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        pass
    with col2:
        generate = st.button("Generate Itinerary", width="stretch", type="primary")
    with col3:
        pass
    if generate and not enough:
        st.write(f"Not enough locations (max {df_selected.shape[0]}) match your preferences. Please select more types of attractions or adjust the number of locations to visit.")
    elif generate:
        st.divider()
        df_selected = clean_dataset.add_start(df_selected, start_lat, start_lon)
        distance_mat = functions.create_distance_matrix(df_selected)
        itinerary, total_dist = optimize.pick_k_nearest(df_selected, num_locations, distance_mat)
        itinerary2, total_dist2 = optimize.pick_loop_points(df_selected, num_locations)
        if total_dist2 < total_dist:
            itinerary = itinerary2
            total_dist = total_dist2
        st.subheader("Interactive Map of Itinerary")
        st.write("Can toggle types of points on/off in the legend and zoom in using top panel (+) button. Hover over points to see details.")
        functions.create_itinerary(df, df_selected, itinerary, total_dist)
else:
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
    end_lat = st.number_input(
        "Enter your ending address latitude (N):",
        min_value=40.5000, max_value=41.0000,
        value=40.7021,
        placeholder="Between 40.5 and 41.0, e.g. 40.7021",
        format="%0.4f"
    )
    end_lon = st.number_input(
        "Enter your ending address longitude (E):",
        min_value=-74.2000, max_value=-73.7000,
        value=-73.9921,
        placeholder="Between -74.2 and -73.7, e.g. -73.9921",
        format="%0.4f"
    )
    num_locations = st.number_input(
        "Enter the number of locations to visit between starting and ending points:",
        1, 15,
        placeholder="e.g., 9",
        format="%d"
    )
    if df_selected.shape[0] < num_locations:
        enough = False
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        pass
    with col2:
        generate = st.button("Generate Itinerary", width="stretch", type="primary")
    with col3:
        pass
    if generate and not enough:
        st.write(f"Not enough locations (max {df_selected.shape[0]}) match your preferences. Please select more types of attractions or adjust the number of locations to visit.")
    elif generate:
        st.divider()
        df_selected = clean_dataset.add_start_end(df_selected, start_lat, start_lon, end_lat, end_lon)
        distance_mat = functions.create_distance_matrix(df_selected)
        itinerary, total_dist = optimize.greedy_dir_alg(df_selected, num_locations, distance_mat)
        st.subheader("Interactive Map of Itinerary")
        st.write("Can toggle types of points on/off in the legend and zoom in using top panel (+) button. Hover over points to see details.")
        functions.create_itinerary(df, df_selected, itinerary, total_dist)