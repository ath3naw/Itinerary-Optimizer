import optimize, clean_dataset, functions
if __name__ == "__main__":
    start_lat = float(input("Enter your starting address latitude: "))
    start_lon = float(input("Enter your starting address longitude: "))
    end_lat = float(input("Enter your ending address latitude: "))
    end_lon = float(input("Enter your ending address longitude: "))
    num_locations = int(input("Enter the number of locations to visit between start and end: "))
    days = int(input("Enter the number of days you plan to stay: "))
    df = clean_dataset.clean_dataset(start_lat, start_lon, end_lat, end_lon)
    distance_mat = functions.create_distance_matrix(df)
    itinerary, total_dist = optimize.greedy_dir_alg(df, num_locations, distance_mat)

    functions.check_path(df, itinerary, total_dist)
    functions.plot_path(df, itinerary)

    end_idx = df.index[(df["name"] == "end")][0]
    locations = df.drop(index=[end_idx])
    lab, cen = optimize.penal_k_means(locations, days)