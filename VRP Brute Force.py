import pandas as pd
import numpy as np

# load the data
dist_df = pd.read_csv('/Users/ykk/Downloads/distance.csv')

# Use six nodes to optimize
cities = ['City_24', 'City_47', 'City_31', 'City_54', 'City_53', 'City_19']

# Mathematical Modeling (Matrix)
n = len(cities)
matrix = np.zeros((n, n))

# 4. Distance
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

from itertools import permutations

# let city_24 is the origin
other_cities = list(range(1, n)) #list the other_cities
min_dist = float('inf') #set the initial min_dist = infinite
best = []

# Try all the possible path
for p in permutations(other_cities):
    current_path = [0] + list(p) + [0]  # 0 to p to 0
    current = 0

    for i in range(len(current_path) - 1):
        current += matrix[current_path[i]][current_path[i + 1]]

    if current < min_dist:
        min_dist = current
        best_path = current_path

print(f"The best route is: {[cities[i] for i in best_path]}")
print(f"The total distance is {min_dist} m")