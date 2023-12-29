import numpy as np
import random
import math

class optimization_data(object):
    def __init__(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        self.day = 30
        self.beta = 0.9

        self.T = 4
        self.T_month = self.T * self.day

        self.TOU = np.array(
            [0.06, 0.12, 0.14, 0.12])
        self.TOU_month = np.tile(self.TOU, self.day)

        self.E_Load = np.array([10, 10, 10, 10])
        self.E_Load_month = np.tile(self.E_Load, self.day)
        self.E_EV = np.array([50, 50, 55, 60, 70, 80, 110, 120, 140, 150, 170, 180, 155, 140, 130, 120, 110, 105, 100, 95, 90, 80, 70, 60])
        self.E_EV_month = np.tile(self.E_EV, 5)
        self.E_grid_user_month = self.E_Load_month + self.E_EV_month

        self.E_PV = 0.6*np.array([0, 102, 178, 0])
        self.E_PV_month = np.tile(self.E_PV, self.day)

        self.Delta_ch = 0.99
        self.Delta_dch = 0.99
        self.ESS_capacity = 60
        self.SOE_min = 0.1 * self.ESS_capacity
        self.SOE_max = 1 * self.ESS_capacity
        self.SOE_zero = 0.5 * self.ESS_capacity
        self.E_ch_min = 0.0
        self.E_ch_max = 10
        self.E_dch_min = 0.0
        self.E_dch_max = 10
        self.SOH_max = 0.8
        self.SOH_min = 0.7
        self.N = 1000
        self.C_ESS = 3000
        self.ESS_life = 10*365
        self.D = 2 * self.N * (self.SOE_max - self.SOE_min)