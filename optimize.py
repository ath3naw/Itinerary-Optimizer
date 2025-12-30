import numpy as np
from itertools import combinations
# want to implement one greedy algorithm based on directions
def greedy_dir_alg(df, num_locations, distance_mat, alpha=0.5, beta=2.0, gamma=0.5):
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
        dists[dists == 0] = np.inf  # avoid division by zero
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

