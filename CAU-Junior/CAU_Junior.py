import numpy as np
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import math
import random
import time
import pandas as pd
from DATA_DO import *

################################### 사용 데이터 ###################################

data = optimization_data(seed=100)

C_ESS = data.C_ESS
beta = data.beta
T = data.T
pi = data.TOU
E_grid_user = data.E_grid_user
E_pv = data.E_PV
Delta_ch = data.Delta_ch
Delta_dch = data.Delta_dch
SOE_min = data.SOE_min
SOE_zero = data.SOE_zero
E_ch_min = data.E_ch_min
E_ch_max = data.E_ch_max
E_dch_min = data.E_dch_min
E_dch_max = data.E_dch_max
SOH_max = data.SOH_max
SOH_min = data.SOH_min
ESS_capacity = data.ESS_capacity
N = data.N
ESS_life = data.ESS_life
D = data.D

################################### 모델 설정 ###################################

model = gp.Model("DO")

C_t = model.addVars(T, lb=0, vtype=gp.GRB.CONTINUOUS, name="C")
C_change = model.addVars(1, lb=0, vtype=GRB.CONTINUOUS, name="C_change")
E_net = model.addVars(T, lb=0, vtype=gp.GRB.CONTINUOUS, name="E_net")

E_ch = model.addVars(T, vtype=gp.GRB.CONTINUOUS, name="E_ch")
E_grid_ch = model.addVars(T, lb=0, vtype=gp.GRB.CONTINUOUS, name="E_grid_ch")
E_pv_ch = model.addVars(T, lb=0, vtype=gp.GRB.CONTINUOUS, name="E_pv_ch")

E_dch = model.addVars(T, vtype=gp.GRB.CONTINUOUS, name="E_dch")

E_pv_waste = model.addVars(T, lb=0, vtype=gp.GRB.CONTINUOUS, name="E_pv_waste")

b_ch = model.addVars(T, lb=0, vtype=gp.GRB.BINARY, name="b_ch")
b_ESS = model.addVars(T, lb=0, vtype=gp.GRB.BINARY, name="b_ESS")
SOE = model.addVars(T+1, vtype=gp.GRB.CONTINUOUS, name="SOE")
SOE_max = model.addVars(T, lb=0, vtype=gp.GRB.CONTINUOUS, name="SOE_max")

SOH = model.addVars(T+1, vtype=gp.GRB.CONTINUOUS, name="SOH")
n_t = model.addVars(T, lb=0, vtype=gp.GRB.CONTINUOUS, name="n_t")

model.addConstr(SOE[0] == SOE_zero)
model.addConstr(SOH[0] == SOH_max)

model.addConstr(C_change[0] == C_ESS * (SOH_max - SOH[T]) / (SOH_max - SOH_min))
for t in range(T):
    model.addConstr(C_t[t] == pi[t] * E_net[t])
    model.addConstr(E_net[t] == E_grid_user[t] - E_dch[t])
    model.addConstr(E_pv[t] == E_pv_waste[t] + E_pv_ch[t])
    model.addConstr(E_ch[t] == b_ch[t] * E_grid_ch[t] + (1 - b_ch[t]) * E_pv_ch[t])
    model.addConstr(SOE[t+1] == SOE[t] + ((Delta_ch * E_ch[t]) - (E_dch[t] / Delta_dch)))
    model.addConstr(SOH[t] == SOE_max[t] / ESS_capacity)
    model.addConstr(n_t[t] == E_ch[t] + E_dch[t])
    model.addConstr(SOH[t+1] == SOH[t] - ((SOH_max - SOH_min) / D) * n_t[t] )

    model.addConstr(0 <= E_pv_waste[t])
    model.addConstr(0 <= E_pv_ch[t])
    model.addConstr(E_pv_waste[t] <= E_pv[t])
    model.addConstr(E_pv_ch[t] <= E_pv[t])


    model.addConstr(E_ch[t] >= E_ch_min * b_ESS[t])
    model.addConstr(E_ch[t] <= E_ch_max * b_ESS[t])
    model.addConstr(E_dch[t] >= E_dch_min * (1 - b_ESS[t]))
    model.addConstr(E_dch[t] <= E_dch_max * (1 - b_ESS[t]))
    model.addConstr(SOE[t] <= SOE_max[t])
    model.addConstr(SOE[t] >= SOE_min)
    model.addConstr(SOH[t] >= SOH_min)
    model.addConstr(SOH[t] <= SOH_max)

obj = sum(C_t[t] for t in range(T)) + C_change[0]
model.setObjective(obj, gp.GRB.MINIMIZE)
model.optimize()
print([E_pv_ch[t].x for t in range(T)])
print([E_dch[t].x for t in range(T)])
print(C_change[0])
print(SOH[T])
print([C_t[t].x for t in range(T)])