import numpy as np
import matplotlib.pyplot as plt
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