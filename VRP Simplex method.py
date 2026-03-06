import pandas as pd
import numpy as np

# Data Loading
dist_df = pd.read_csv('/Users/ykk/Downloads/distance.csv')

# Select six nodes to optimize
cities = ['City_24', 'City_47', 'City_31', 'City_54', 'City_53', 'City_19']

# Mathematical Modeling (Matrix)
n = len(cities)
matrix = np.zeros((n, n))

# Distance
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

import pulp

# Define the Problem as minimize problem
prob = pulp.LpProblem("TSP_Optimization", pulp.LpMinimize)

# Decision variable x[i][j] is Binary variable
x = pulp.LpVariable.dicts("x", (range(n), range(n)), cat='Binary')

# Using u[i] to eliminate subtour
u = pulp.LpVariable.dicts("u", range(n), lowBound=0, upBound=n-1, cat='Continuous')

# objective function
prob += pulp.lpSum(matrix[i][j] * x[i][j] for i in range(n) for j in range(n) if i != j)

# Constraints
for i in range(n):
    # each cities can only leave once
    prob += pulp.lpSum(x[i][j] for j in range(n) if i != j) == 1
    # each citeis can only visit once
    prob += pulp.lpSum(x[j][i] for j in range(n) if i != j) == 1

# MTZ subtour elimination
for i in range(1, n):
    for j in range(1, n):
        if i != j:
            prob += u[i] - u[j] + n * x[i][j] <= n - 1

# 6. Solve the Problem
prob.solve(pulp.PULP_CBC_CMD(msg=0))

# Optimal solution
print(f"The porblem is {pulp.LpStatus[prob.status]}")
print(f"Total distance is {pulp.value(prob.objective)} m")

# Find the best path
curr = 0
path = [cities[curr]]
while len(path) < n: # if smaller than n, then keep going
    # if the curr and next_city been selected(=1)
    for next_city in range(n):
        if curr != next_city and pulp.value(x[curr][next_city]) == 1:
            path.append(cities[next_city]) #add the "next city" into the path
            curr = next_city # curr become 'next city' (loop)
            break
path.append(cities[0]) # back to city 0
print(f"The best path is: {path}")