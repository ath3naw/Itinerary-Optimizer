import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go
def create_distance_matrix(df):
    X = df[["easting", "northing"]].to_numpy(dtype=float)

    diff = X[:, None, :] - X[None, :, :]   # shape (n, n, 2)
    return np.linalg.norm(diff, axis=2)

def plot_path(df, order, title="NYC Museums"):
    start_idx = df.index[(df["name"] == "start")][0]
    loop = False
    if "end" in df["name"].values:
        end_idx = df.index[df["name"] == "end"][0]
    else:
        end_idx = None
        loop = True
    fig, ax = plt.subplots(figsize = (9,6))
    plt.scatter(df["easting"], df["northing"], c='blue', marker='o')
    if loop:
        lim_x = min(df.loc[order, "easting"]) - 500, max(df.loc[order, "easting"]) + 500
        lim_y = min(df.loc[order, "northing"]) - 1000, max(df.loc[order, "northing"]) + 1000
    else:
        lim_x = min(df.loc[order, "easting"], df.loc[end_idx, "easting"]) - 500, max(df.loc[order, "easting"], df.loc[end_idx, "easting"]) + 500
        lim_y = min(df.loc[order, "northing"], df.loc[end_idx, "northing"]) - 1000, max(df.loc[order, "northing"], df.loc[end_idx, "northing"]) + 1000
    plt.xlim(lim_x)
    plt.ylim(lim_y)
    plt.plot(df["easting"].iloc[order], df["northing"].iloc[order], c='red', marker='o')
    plt.xlabel('Easting')
    plt.ylabel('Northing')
    plt.title(title)
    ax.annotate('Start', xy=(df["easting"].iloc[start_idx]-0.0001, df["northing"].iloc[start_idx]), xytext=(df["easting"].iloc[start_idx]-0.001, df["northing"].iloc[start_idx]-0.002))
    if not loop:
        ax.annotate('End', xy=(df["easting"].iloc[end_idx]+0.0001, df["northing"].iloc[end_idx]), xytext=(df["easting"].iloc[end_idx]+0.0003, df["northing"].iloc[end_idx]+0.001))
    plt.grid()
    plt.show()

def check_path(df, order, total_dist):
    full_itinerary = df.iloc[order]["name"]
    print("Itinerary:\n", full_itinerary)
    print("Total distance (m):", total_dist)

def plot_clusters(df, labels):
    start_idx = df.index[(df["name"] == "start")][0]
    fig, ax = plt.subplots(figsize = (9,6))
    plt.scatter(df["easting"], df["northing"], c=labels, marker='o', cmap = "tab20")
    plt.colorbar(label = "cluster")
    ax.annotate('Start', xy=(df["easting"].iloc[start_idx]-0.0001, df["northing"].iloc[start_idx]), xytext=(df["easting"].iloc[start_idx]-0.001, df["northing"].iloc[start_idx]-0.002))
    plt.xlabel('Easting')
    plt.ylabel('Northing')
    plt.title('NYC Museums for', len(np.unique(labels)), ' Day Itinerary')
    plt.grid()
    plt.show()

def create_itinerary(df, df_selected, itinerary, total_dist):
    fig1 = go.Figure()
    fig1.add_trace(go.Scattermapbox(
        mode="markers", # Only markers
        lon=df['lon'],
        lat=df['lat'],
        marker=dict(size=10, color='blue', opacity=0.8),
        text=df['name'], # Text for hover
        name="Attractions"
    ))
    fig1.add_trace(go.Scattermapbox(
        mode="lines",
        lat=df_selected["lat"].iloc[itinerary],
        lon=df_selected["lon"].iloc[itinerary],
        line=dict(width=4, color="red"),
        text=df['name'],
        name="Route"
    ))
    route_lats = df_selected['lat'].iloc[itinerary]
    route_lons = df_selected['lon'].iloc[itinerary]
    lat_min, lat_max = df['lat'].min(), df['lat'].max()
    lon_min, lon_max = df['lon'].min(), df['lon'].max()
    center_lat = (lat_min + lat_max) / 2
    center_lon = (lon_min + lon_max) / 2
    fig1.add_trace(go.Scattermapbox(
        mode="markers+text",
        lat=route_lats,
        lon=route_lons,
        marker=dict(size=14, color="red"),
        text=[str(i+1) for i in range(len(itinerary))],  # order labels
        textposition="top right",
        hovertext=df_selected["name"].iloc[itinerary],
        hoverinfo="text",
        name="Route Order"
    ))
    fig1.update_layout(mapbox = dict(style="carto-positron",
                                    center=dict(lat=center_lat, lon=center_lon),
                                    zoom=9), height=500) # Set map style
    st.plotly_chart(fig1, use_container_width=True)
    st.subheader("New York Itinerary")
    for i, loc in enumerate(itinerary):
        if i == 0:
            st.write(f"{i+1}. {'Start'}")
        elif i != len(itinerary) - 1:
            st.write(f"{i+1}. {df_selected['name'].iloc[loc]}")
        else:
            st.write(f"{i+1}. {'End'}")
        if pd.notna(df_selected['image'].iloc[loc]):
            col1, col2, col3 = st.columns([1,2,1])
            with col1:
                pass
            with col2:
                st.image(df_selected['image'].iloc[loc], width='stretch')
            with col3:
                pass
        if pd.notna(df_selected['description'].iloc[loc]):
            st.write(df_selected['description'].iloc[loc])
    col1, col2, col3 = st.columns([1,3,1])
    with col1:
        pass
    with col2:
        st.subheader(f"Total distance (m): {total_dist:.2f}")
    with col3:
        pass
    st.badge("Note: Distances are approximate and calculated using UTM coordinates.")