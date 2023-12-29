import gurobipy
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import math
import random
import time
from data_t import *
start = time.time()
opt_data = data(time=24,unit=30,seed=100)
opt_data.DRO_parameter(n_scenario=40, lamb=1, Type= 'DRO')
opt_data.MPC_parameter(horizon=36,time_step=1)

N =opt_data.n_scenario
Total_time = opt_data.T
Horizon = opt_data.horizon
L = opt_data.time_step

eps_E_sell =opt_data.radius_E_sell
eps_E_buy=opt_data.radius_E_buy
ch_eff = opt_data.eff
dch_eff = opt_data.eff

SOE_ini = opt_data.SOE_ini
SOE_min = opt_data.SOE_min
SOE_max = opt_data.SOE_max
Ech_min = opt_data.Ch_min
Ech_max = opt_data.Ch_max
Edch_min = opt_data.Dch_min
Edch_max = opt_data.Dch_max
E_sell_min = opt_data.E_sell_min
E_sell_max = opt_data.E_sell_max
E_buy_min = opt_data.E_buy_min
E_buy_max = opt_data.E_buy_max
random_E_sell = np.array(opt_data.historical_sample_E_sell)
random_E_buy = np.array(opt_data.historical_sample_E_buy)
pi_sell = opt_data.pi_sell
pi_buy = opt_data.pi_buy
E_PV = opt_data.E_PV
E_fixed = opt_data.E_fixed

area_attribute_min = 0.5
area_attribute_max = 1.5
E_net_final = []
E_ch_final = []
E_dch_final = []

profit = 0.0
for l_idx in range(math.ceil(Total_time/L)):
    time_idx = l_idx
    current_start = l_idx*L
    current_end = l_idx*L + Horizon
    next_start = (l_idx+1)*L
    if current_end > Total_time:
        current_end = Total_time
        Horizon = Total_time - current_start
    else:
        pass

    model = gp.Model(name="DRO+MPC")

    E_Net = model.addVars(Horizon, vtype=gp.GRB.CONTINUOUS,  name="Energy_consumption")
    E_ch = model.addVars(Horizon, vtype=gp.GRB.CONTINUOUS, lb=Ech_min, ub=Ech_max, name="ESS_Charging")
    E_dch = model.addVars(Horizon, vtype=gp.GRB.CONTINUOUS, lb=Edch_min, ub=Edch_max, name="ESS_Discharging")
    SOE = model.addVars(Horizon + 1, vtype=gp.GRB.CONTINUOUS, lb=SOE_min, ub=SOE_max, name="SOE")
    b_ess = model.addVars(Horizon, vtype=gp.GRB.BINARY, name="b_ESS")
    s_E_sell =np.array([[model.addVar(name="s_E_sell({},{})".format(horizon_idx, data_idx)) for data_idx in range(N)] for horizon_idx in
         range(Horizon)])
    s_E_buy =  np.array([[model.addVar(name="s_E_net({},{})".format(horizon_idx, data_idx)) for data_idx in range(N)] for horizon_idx in
        range(Horizon)])
    DRO_lambda_sell = model.addVars(Horizon, vtype=gp.GRB.CONTINUOUS, name="DRO_lambda_sell")
    DRO_lambda_buy = model.addVars(Horizon, vtype=gp.GRB.CONTINUOUS, name="DRO_lambda_buy")
    model.addConstr(SOE[0] == SOE_ini)

    for horizon_idx in range(Horizon):
        model.addConstr(SOE[horizon_idx + 1] >= SOE_min)
        model.addConstr(SOE[horizon_idx + 1] <= SOE_max)
        model.addConstr(E_ch[horizon_idx] >= Ech_min * b_ess[horizon_idx])
        model.addConstr(E_ch[horizon_idx] <= Ech_max * b_ess[horizon_idx])
        model.addConstr(E_dch[horizon_idx] >= Edch_min * (1 - b_ess[horizon_idx]))
        model.addConstr(E_dch[horizon_idx] <= Edch_max * (1 - b_ess[horizon_idx]))

        for data_idx in range(0, N):
            model.addConstr(s_E_sell[horizon_idx][data_idx] <= pi_sell[time_idx] * E_sell_min[time_idx]
                            - DRO_lambda_sell[horizon_idx] * (E_sell_min[time_idx] - random_E_sell[time_idx][data_idx]))
            model.addConstr(s_E_sell[horizon_idx][data_idx] <= pi_sell[time_idx] * random_E_sell[time_idx][data_idx])

            model.addConstr(s_E_buy[horizon_idx, data_idx] >= pi_buy[time_idx] * (E_buy_min[time_idx] + E_Net[horizon_idx])
                            + DRO_lambda_buy[horizon_idx] * (E_buy_min[time_idx] - random_E_buy[time_idx, data_idx]))
            model.addConstr(s_E_buy[horizon_idx, data_idx] >= pi_buy[time_idx] * (E_buy_max[time_idx] + E_Net[horizon_idx])
                            - DRO_lambda_buy[horizon_idx] * (E_buy_max[time_idx] - random_E_buy[time_idx, data_idx]))
            model.addConstr(s_E_buy[horizon_idx, data_idx] >= pi_buy[time_idx] * (random_E_buy[time_idx, data_idx] + E_Net[horizon_idx]))

            model.addConstr(E_Net[horizon_idx] == E_fixed[time_idx] +(E_ch[horizon_idx] - E_dch[horizon_idx]))
            model.addConstr(SOE[horizon_idx + 1] == SOE[horizon_idx] + ch_eff * E_ch[horizon_idx] - (E_dch[horizon_idx] / dch_eff) + E_PV[time_idx])
            model.addConstr( E_Net[horizon_idx] >= -1 * random_E_buy[time_idx][data_idx])
            #model.addConstr(E_Net[horizon_idx] >= 0)


        time_idx = time_idx + 1

    J1 = [0]*Horizon
    J2 = [0]*Horizon
    #time_idx = l_idx

    for horizon_idx in range(Horizon):
        J1[horizon_idx] = -DRO_lambda_sell[horizon_idx] * eps_E_sell[current_start+horizon_idx] + (1 / N) * sum(
                     s_E_sell[horizon_idx, data_idx] for data_idx in range(N))

        J2[horizon_idx] = DRO_lambda_buy[horizon_idx] * eps_E_buy[current_start+horizon_idx] + (1 / N) * sum(
                    s_E_buy[horizon_idx, data_idx] for data_idx in range(N))

    obj = sum(J1[horizon_idx] - J2[horizon_idx] for horizon_idx in range(Horizon))

    model.setObjective(obj, GRB.MAXIMIZE)
    model.optimize()


    if l_idx != (math.ceil(Total_time/L)-1):
        for time_idx in range(L):
            profit += pi_sell[current_start]*opt_data.E_sell[current_start] - pi_buy[current_start]*(opt_data.E_buy[current_start] + E_Net[time_idx].x)
            current_start += 1
            E_net_final = np.append(E_net_final, E_Net[time_idx].x)
            E_ch_final = np.append(E_ch_final, E_ch[time_idx].x)
            E_dch_final = np.append(E_dch_final, E_dch[time_idx].x)
        SOE_ini = SOE[L+1].x

    else:
        for horizon_idx in range(Horizon):
            profit += pi_sell[current_start]*opt_data.E_sell[current_start]- pi_buy[current_start]*(opt_data.E_buy[current_start] + E_Net[horizon_idx].x)
            current_start +=1
            E_net_final = np.append(E_net_final, E_Net[horizon_idx].x)
            E_ch_final = np.append(E_ch_final, E_ch[horizon_idx].x)
            E_dch_final = np.append(E_dch_final, E_dch[horizon_idx].x)

end = time.time()
print("********************")
print(profit)
print("********************")
print(E_net_final)
print(np.array(E_ch_final))
print(np.array(E_dch_final))
print("*******************************************************")
print(np.array(E_ch_final) - np.array(E_dch_final))

print("++++++++++++++++++++++++++++")
print(end-start)