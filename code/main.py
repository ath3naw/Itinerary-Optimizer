import optimize, clean_dataset, functions
if __name__ == "__main__":
    loop = input("Is your starting location the same as your ending location? (yes/no): ").strip().lower()
    again = True
    while again:
        again = False
        if loop == 'yes':
            start_lat = float(input("Enter your starting address latitude: "))
            start_lon = float(input("Enter your starting address longitude: "))
            num_locations = int(input("Enter the number of locations to visit between start and end: "))
            days = int(input("Enter the number of days you plan to stay: "))
            df = clean_dataset.add_start(start_lat, start_lon)
            print(df.head())
            distance_mat = functions.create_distance_matrix(df)
            itinerary, total_dist = optimize.pick_k_nearest(df, num_locations, distance_mat)
            functions.check_path(df, itinerary, total_dist)
            functions.plot_path(df, itinerary)
        elif loop == 'no':
            itinerary, total_dist = optimize.greedy_dir_alg(df, num_locations, distance_mat)
            start_lat = float(input("Enter your starting address latitude: "))
            start_lon = float(input("Enter your starting address longitude: "))
            end_lat = float(input("Enter your ending address latitude: "))
            end_lon = float(input("Enter your ending address longitude: "))
            num_locations = int(input("Enter the number of locations to visit between start and end: "))
            days = int(input("Enter the number of days you plan to stay: "))
            df = clean_dataset.add_start_end(start_lat, start_lon, end_lat, end_lon)
            distance_mat = functions.create_distance_matrix(df)
            itinerary, total_dist = optimize.greedy_dir_alg(df, num_locations, distance_mat)

            functions.check_path(df, itinerary, total_dist)
            functions.plot_path(df, itinerary)

            end_idx = df.index[(df["name"] == "end")][0]
            locations = df.drop(index=[end_idx])
            lab, cen = optimize.penal_k_means(locations, days)
            functions.plot_clusters(locations, lab)
        else:
            again = True
            print("Invalid input. Please enter 'yes' or 'no'.")