import numpy as np
from itertools import combinations
# want to implement one greedy algorithm based on directions
def greedy_dir_alg(df, num_locations, distance_mat, alpha=0.5, beta=2.0, gamma=0.5):
    """
    A greedy algorithm calculating the best itinerary to minimize travel distances based on
    direction from start to end point
    
    :param df: dataframe of NYC attractions, including start and end points
    :param num_locations: number of locations user plans to visit in a day
    :param distance_mat: distance from all NYC attractions to all other NYC attractions
    :param alpha: optimization parameter controlling for weight of distance to previous attraction (adjust higher if would like to visit attractions closer together)
    :param beta: optimization parameter controlling for shorter path (more specifically, alignment with the direction user must travel in to reach destination)
    :param gamma: optimization parameter controlling for weight of distance to destination (adjust higher if would like to visit more attractions closer to destination),
    make alpha and gamma equal if the weight of the distances to previous attractions and endpoint don't matter that much
    """
    start_idx = df.index[(df["name"] == "start")][0]
    end_idx = df.index[df["name"] == "end"][0]
    itinerary = [start_idx]
    visited = [start_idx, end_idx]

    start = df.iloc[start_idx][["easting", "northing"]].to_numpy(dtype=float)
    end = df.iloc[end_idx][["easting", "northing"]].to_numpy(dtype=float)

    current_idx = start_idx
    curr = start
    total_dist = 0

    for _ in range(num_locations):
        best_idx = None

        dir = end - curr
        dir /= np.linalg.norm(dir)

        curr = df.iloc[current_idx][["easting", "northing"]].to_numpy(dtype=float)
        points = df[["easting", "northing"]].to_numpy(dtype=float)

        vect = points-curr
        dists = np.linalg.norm(vect, axis=1)
        dists[dists == 0] = np.inf  # avoid division by zero, makes it impossible to go on same coordinate if starting at that coordinate
        align = np.dot(vect / dists[:, None], dir)
        remain = np.linalg.norm(end - points, axis=1)

        # closer distances to current point are better, higher alignment is better, closer to end is better (can adjust)
        # default is no preference for closer points vs farther points on a route
        score = alpha * dists - beta * align + gamma * remain

        score[visited] = np.inf
        best_idx = np.argmin(score)
        total_dist += dists[best_idx]
        visited.append(best_idx)
        itinerary.append(best_idx)
        current_idx = best_idx

    itinerary.append(end_idx)
    total_dist += np.linalg.norm(
        df.iloc[current_idx][["easting", "northing"]].values - end
    )
    itinerary, total_dist = organize_path_dp(itinerary, distance_mat)
    return itinerary, total_dist

def organize_path_dp(order, distance_mat):
    n = len(order)
    map = {}
    idx_map = {i: i-1 for i in range(1,n-1)}  # exclude start and end

    # base case, only visited 1 node
    for i in range(1,n-1):
        map[(1 << idx_map[i], i)] = (0, distance_mat[order[0], order[i]])  # map(visited nodes, last_node) = (last node, distance)

    # for any visited set
    for set_len in range(2, n-1):
        for subset in combinations(range(1, n-1), set_len):
            visited = sum(1 << idx_map[j] for j in subset)
            for node in subset:
                prev_visited = visited ^ (1 << idx_map[node]) # visited without that one node
                best_dist = np.inf
                best_prev = None
                # pick a different node from prev_visited to come from
                for k in subset:
                    if k == node:
                        continue
                    _, prev_dist = map[(prev_visited, k)]
                    new_dist = prev_dist + distance_mat[order[k], order[node]]
                    if new_dist < best_dist:
                        best_dist = new_dist
                        best_prev = k
                map[(visited, node)] = (best_prev, best_dist)
                #print(f"Visited: {visited}, Last node: {node}, Prev: {best_prev}, Dist: {best_dist}")
    # finish at end node
    full_visited = (1 << (n-2)) - 1  # all nodes except start and end
    best_dist = np.inf
    end_prev = None
    for i in range(1, n-1):
        _, prev_dist = map[(full_visited, i)]
        new_dist = prev_dist + distance_mat[order[i], order[-1]]
        if new_dist < best_dist:
            best_dist = new_dist
            end_prev = i

    # reconstruct path by going from end to beginning
    path = [order[-1], order[end_prev]]
    visited = full_visited
    curr = end_prev

    while visited:
        prev, _ = map[(visited, curr)]
        path.append(order[prev])
        visited = visited ^ (1 << idx_map[curr])
        curr = prev
    path.reverse()
    return path, best_dist

# want to actually implement k-means based on distance from the starting idx, want to start with start and end idx appended
def kmeans_init_from_start(df, days):
    """
    Initializing cluster centers for k_means++ algorithm, with fixed cluster around starting point
    
    :param df: dataframe of all locations of NYC attractions and starting location in eastings and northings
    :param days: number of days to spend doing the itinerary
    """
    locs = df[["easting", "northing"]]
    start_idx = df.index[(df["name"] == "start")][0]
    n, d = locs.shape[0], locs.shape[1] # d = number of features considered
    n = n-1
    locations = locs.to_numpy(dtype = float)
    # already define one of the centers as the starting idx
    centers = np.zeros((days+1, d), dtype=float)
    centers[0] = locations[start_idx]
    remain_index = list(range(n))

    # for all the clusters (not including the start), choose a random cluster center weighted proportionally to distance to other cluster centers
    for idx in range(1, days+1):
        dists = np.min(np.linalg.norm(locations[remain_index][:,None,:] - centers[None, :idx, :], axis = 2)**2, axis = 1)
        probs = dists / np.sum(dists)
        new_idx = np.random.choice(remain_index, p=probs)
        centers[idx] = locations[new_idx]
        remain_index.remove(new_idx)
    return centers

# penalized k-means, will use same k_means++ distribution for initiation (want cluster centers to be as spread out as possible initially)
def penal_k_means(df, days, lam=0.2, gam=1e-7, max_iters=50, eps=1e-5):
    """
    Soft penalty optimization algorithm using k-means++ with fixed cluster around starting point, also adjusting for size of clusters and
    distance to cluster centers

    :param df: dataframe of NYC attractions, including starting location in eastings and northings
    :param days: number of days to spend doing the itinerary
    :param lam: optimization parameter controlling for significance of equally sized clusters
    :param gam: optimization parameter controlling for distance of cluster centers to starting point - note that it's often small because of
    the small scale of eastings and northings in NYC
    :param max_iters: maximum iterations of clustering, if it doesn't converge to a change less than eps
    :param eps: the limit at which we can stop iterating and changing points from different clusters
    """
    start_idx = df.index[(df["name"] == "start")][0]
    locs = df[["easting", "northing"]]
    centers = kmeans_init_from_start(locs, days, start_idx)
    # don't actually want to include end point in locations to map out
    n,d = locations.shape[0], locations.shape[1]
    locations = locations.to_numpy(dtype = float)
    costs = np.zeros((n, d))
    delt = np.zeros(days+1)
    delt[0] = 500
    # start loop for max_iters times
    for _ in range(max_iters):
        # assign clusters
        dists = np.linalg.norm(locations[:,None,:]-centers[None,:,:], axis = 2)
        sizes = np.zeros(days+1)
        tar = [(n-1)/days if a != 0 else 1 for a in range(days+1)]
        labels = np.empty(n)
        # want to have initial assignments and then adjust to locally optimal minima/maxima like in gradient descent
        dist_pen = gam*(np.linalg.norm(centers-centers[0], axis = 1))**2
        # sort from minimum distance to centers to maximum distance to centers, greedy algorithm to calculate costs/penalties
        # and assign clusters based on costs (the closer to the cluster center, the better)
        for i in np.argsort(np.min(dists, axis = 1)):
            costs = []
            for j in range(days+1):
                bal_pen = lam*(sizes[j]-tar[j])**2
                costs.append(dists[i,j]+bal_pen+dist_pen[j]+delt[j])
            idx = np.argmin(costs)
            labels[i] = idx
            sizes[idx] += 1

        # recompute centers
        new_centers = centers.copy()
        for j in range(days+1):
            if j == 0:
                continue
            elif np.any(labels == j):
                new_centers[j] = locations[labels == j].mean(axis = 0)
        # if total change is less than epsilon, stop loop
        if np.linalg.norm(new_centers - centers) < eps:
            break

        centers = new_centers
    return labels, centers