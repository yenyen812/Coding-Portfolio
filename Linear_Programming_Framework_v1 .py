from pulp import *

lp = LpProblem('Backery_Problem',LpMaximize) #Maximize my problem

# decision variable
var_keys = [1,2,3,4,5]
x = LpVariable.dicts("Bakery_item", var_keys,lowBound=0, cat="Integer") # Variables are non_negative integer
print (x)

# define objective fnction
lp += 10*x[1]+5*x[2]+6*x[3]+7*x[4]+6*x[5]
print(lp.objective)

# constraints
resources = {"oven": 180,"food_processor": 300,"boiler": 145} #RHS
coeff = {"oven":{1:5,2:1,3:4,4:2,5:3},"food_processor": {1:1,2:1,3:6,4:3,5:7},"boiler":{1:4,2:6,3:2,4:6,5:1}} #LHS
for r in resources.keys():
    lp += (lpSum(coeff[r][i] * x[i] for i in var_keys) <= resources[r], f"{r}_constraint")
print(lp.constraints)

# solve the problem with CBC algorithm and don't show the porcess
status = lp.solve(PULP_CBC_CMD(msg=0))
print ("Status:", status) #1:optimal

# print the objective variables value
for var in lp.variables():
    print(var, "=", value(var))
print('OPT =', value(lp.objective))