import pandas as pd
import numpy as np

# load the data
dist_df = pd.read_csv('/Users/ykk/Downloads/distance.csv')

# six nodes to optimize
cities = ['City_24', 'City_47', 'City_31', 'City_54', 'City_53', 'City_19']

# Mathematical Modeling (Matrix)
n = len(cities)
matrix = np.zeros((n, n))

# Distnace
for i in range(n):
    for j in range(n):
        if i == j:
            matrix[i][j] = 0
        # find the distance betwenn A,B or B,A
        else:
            # find the distance between i,j
            d = dist_df[(dist_df['Source'] == cities[i]) & (dist_df['Destination'] == cities[j])]
            # if there is no i,j, then find the route from opposite direction
            if d.empty:
                d = dist_df[(dist_df['Source'] == cities[j]) & (dist_df['Destination'] == cities[i])]
            matrix[i][j] = d['Distance(M)'].values[0]
print(matrix)

#--DP--
dp = {}
def solve_dp():
    # Initialization
    # Bitmask (Still confused)
    # r=1
    for i in range(1, n):
        dp[(1 << i) | 1, i] = (matrix[0][i], [0, i]) #(distance, [path])

    # r=2 to n-1
    for r in range(2, n):
        from itertools import combinations
        for subset in combinations(range(1, n), r):
            # mask will become 111 ig 0,1,2 have been visited
            mask = 1
            for bit in subset:
                mask |= (1 << bit)

            #
            for next_city in subset:
                prev_mask = mask & ~(1 << next_city)  #

                best_dist = float('inf') #initial solution is infinite
                best_prev_path = []

                # find which city in the subset is the nearest
                for prev_city in subset:
                    if prev_city == next_city: continue #cannot be the same city
                    # check if the state has been saved into dp
                    if (prev_mask, prev_city) in dp:
                        # calculate the new distance
                        dist = dp[prev_mask, prev_city][0] + matrix[prev_city][next_city]
                        # if the distance now is better than best_dist, then renew
                        if dist < best_dist:
                            best_dist = dist
                            best_prev_path = dp[prev_mask, prev_city][1]

                # save the data (confused)
                if best_dist != float('inf'):
                    dp[mask, next_city] = (best_dist, best_prev_path + [next_city])

    # Final
    full_mask = (1 << n) - 1 # all cities have to be 1
    min_dist = float('inf') # infinite
    final_path = []

    for i in range(1, n):
        dist = dp[full_mask, i][0] + matrix[i][0]
        if dist < min_dist:
            min_dist = dist
            # add city 0
            final_path = dp[full_mask, i][1] + [0]

    return min_dist, final_path


# Final
min_dist, best_path = solve_dp()

print(f"The best route (DP) is: {[cities[i] for i in best_path]}")
print(f"The total distance is {min_dist} m")