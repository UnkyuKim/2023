import numpy as np
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import math
import random
import time
import pandas as pd
from data_new import *

data = optimization_data(seed=100)
start = time.time()
data.DRO_parameter(n_scenario=15, lamb=1, Type='DRO')

T = data.T
Delta_ch = data.Delta_ch
Delta_dch = data.Delta_dch
SOE_min = data.SOE_min
SOE_max = data.SOE_max
SOE_first = data.SOE_first
E_ch_min = data.E_ch_min
E_ch_max = data.E_ch_max
E_dch_min = data.E_dch_min
E_dch_max = data.E_dch_max
E_PV = data.E_PV
TOU = data.TOU
E_Load = data.E_Load
E_Peak = data.E_Peak
E_EV = data.E_EV
E_shave = data.E_shave
E_ev = data.E_ev
E_before_net = data.E_before_net
net_data = data.net_data
for t in range(T):
    if E_shave[t] <= 0:
        E_shave[t] = 0
    else:
        E_shave[t] = E_shave[t]

E_EV_min = data.E_EV_min
E_EV_max = data.E_EV_max

N_EV = data.n_scenario
epsilon = data.radius
E_EV_sample = data.historical_sample

zero = np.array([0.0]*T)

model = gp.Model("DRO_with_WM")

C_t = model.addVars(T, lb=0, vtype=gp.GRB.CONTINUOUS, name="C")
E_Net = model.addVars(T, lb=0, vtype=gp.GRB.CONTINUOUS, name="E_Net")
E_ch = model.addVars(T, vtype=gp.GRB.CONTINUOUS, name="E_ch")
E_dch = model.addVars(T, vtype=gp.GRB.CONTINUOUS, name="E_dch")
E_shave = model.addVars(T, lb=-E_Peak, vtype=gp.GRB.CONTINUOUS, name="E_shave")
b_ESS = model.addVars(T, lb=0, vtype=gp.GRB.BINARY, name="b_ESS")
SOE = model.addVars(T+1, vtype=gp.GRB.CONTINUOUS, name="SOE")
lambda_EV = model.addVars(T, lb=0, vtype=gp.GRB.CONTINUOUS, name="lambda_EV")
E_total_cost = model.addVars(T, lb=0, vtype=gp.GRB.CONTINUOUS, name="Total_cost")

model.addConstr(SOE[0] == SOE_first)

for t in range(T+1):
    model.addConstr(SOE[t] <= SOE_max)
    model.addConstr(SOE[t] >= SOE_min)


for t in range(T):
    model.addConstr(C_t[t] == TOU[t] * E_Net[t])
    model.addConstr(E_Net[t] == E_Load[t] + E_EV[t] + (E_ch[t] - E_dch[t]) - E_PV[t])
    model.addConstr(SOE[t+1] == SOE[t] + (Delta_ch * E_ch[t]) - (E_dch[t] / Delta_dch))

    model.addConstr(0.0 <= E_Net[t])
    model.addConstr(E_Net[t] <= E_Peak)
    model.addConstr(E_shave[t] >= 0.0)
    model.addConstr(E_dch[t] <= E_shave[t])
    model.addConstr(E_ch[t] >= E_ch_min * b_ESS[t])
    model.addConstr(E_ch[t] <= E_ch_max * b_ESS[t])
    model.addConstr(E_dch[t] >= E_dch_min * (1 - b_ESS[t]))
    model.addConstr(E_dch[t] <= E_dch_max * (1 - b_ESS[t]))

    model.addConstr(E_total_cost[t] == E_Net[t] * TOU[t])
model.addConstr(SOE[T] >= SOE_first)
s_EV = {}

for t in range(T):
    for m in range(N_EV):
        s_EV[t, m] = model.addVar(vtype=gp.GRB.CONTINUOUS, name="s_EV({}, {})".format(t+1, m+1))



J = sum(lambda_EV[t] * epsilon[t] + (1/N_EV) * sum(s_EV[t, m] for m in range(N_EV)) for t in range(T))

for t in range(T):
    for m in range(N_EV):

        model.addConstr(s_EV[t, m] >= TOU[t] * (E_Net[t] - E_EV[t] + E_EV_min[t]) + lambda_EV[t] * (E_EV_min[t] - E_EV_sample[t][m]))
        model.addConstr(s_EV[t, m] >= TOU[t] * (E_Net[t] - E_EV[t] + E_EV_max[t]) - lambda_EV[t] * (E_EV_max[t] - E_EV_sample[t][m]))
        model.addConstr(s_EV[t, m] >= TOU[t] * E_EV_sample[t][m])

model.setObjective(J, gp.GRB.MINIMIZE)
model.optimize()



print([E_Net[t].x for t in range(T)])
print([E_ch[t].x for t in range(T)])
print([E_dch[t].x for t in range(T)])
# print(E_Load)
# print(E_ev)
# print([SOE[t].x for t in range(T+1)])
# print([E_Net[t].x for t in range(T)])
# print([[E_EV_sample[t, m] for m in range(N_EV)] for t in range(T)])
# print([b_ESS[t].x for t in range(T)])
print([E_total_cost[t].x for t in range(T)])

