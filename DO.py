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


for t in range(T):
    if E_shave[t] <= 0:
        E_shave[t] = 0
    else:
        E_shave[t] = E_shave[t]

model = gp.Model("DO")

C_t = model.addVars(T, lb=0, vtype=gp.GRB.CONTINUOUS, name="C")
E_Net = model.addVars(T, lb=0, vtype=gp.GRB.CONTINUOUS, name="E_Net")
E_ch = model.addVars(T, vtype=gp.GRB.CONTINUOUS, name="E_ch")
E_dch = model.addVars(T, vtype=gp.GRB.CONTINUOUS, name="E_dch")
E_shave_max = model.addVars(T, lb=0, vtype=gp.GRB.CONTINUOUS, name="E_shave_max")
zero = model.addVars(T, vtype=gp.GRB.CONTINUOUS, name="zero")
E_before_shave = model.addVars(T, vtype=gp.GRB.CONTINUOUS, name="E_before_shave")
b_ESS = model.addVars(T, lb=0, vtype=gp.GRB.BINARY, name="b_ESS")
SOE = model.addVars(T+1, vtype=gp.GRB.CONTINUOUS, name="SOE")
model.addConstr(SOE[0] == SOE_first)

for t in range(T):
    model.addConstr(C_t[t] == TOU[t] * E_Net[t])
    model.addConstr(E_Net[t] == E_Load[t] + E_EV[t] + (E_ch[t] - E_dch[t]) - E_PV[t])
    model.addConstr(SOE[t+1] == SOE[t] + (Delta_ch * E_ch[t]) - (E_dch[t] / Delta_dch))

    model.addConstr(E_before_shave[t] == E_Load[t] + E_EV[t] - E_PV[t])
    model.addConstr(E_dch[t] <= E_shave[t])
    model.addConstr(0 <= E_Net[t])
    model.addConstr(E_Net[t] <= E_Peak)
    model.addConstr(E_ch[t] >= E_ch_min * b_ESS[t])
    model.addConstr(E_ch[t] <= E_ch_max * b_ESS[t])
    model.addConstr(E_dch[t] >= E_dch_min * (1 - b_ESS[t]))
    model.addConstr(E_dch[t] <= E_dch_max * (1 - b_ESS[t]))
    model.addConstr(SOE[t] <= SOE_max)
    model.addConstr(SOE[t] >= SOE_min)

model.addConstr(SOE[T] >= SOE_first)

obj = sum(C_t[t] for t in range(T))

model.setObjective(obj, gp.GRB.MINIMIZE)
model.optimize()

print([E_Net[t].x for t in range(T)])
print([E_dch[t].x for t in range(T)])