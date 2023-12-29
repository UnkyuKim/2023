import numpy as np
import random
import math

class optimization_data(object):
    def __init__(self, seed):
        np.random.seed(seed)
        random.seed(seed)

        self.T = 24
        self.Delta_ch = 0.999
        self.Delta_dch = 0.999
        self.ESS_capacity = 350
        self.SOE_min = 0.1 * self.ESS_capacity
        self.SOE_max = 1 * self.ESS_capacity
        self.SOE_first = 0.5 * self.ESS_capacity
        self.E_ch_min = 0.0
        self.E_ch_max = 150
        self.E_dch_min = 0.0
        self.E_dch_max = 150
        self.E_EV = np.array([25,  32,  34,  32,  45,  60,  81,  88, 104, 131, 184, 212, 197, 206, 181, 170, 140, 115,  97,  82,  85,  68,  56,  40])
        self.E_PV = np.array([0,  0,  0,  0,  0,  0,  2,  6, 12, 20, 28, 34, 38, 40, 36, 30, 20, 10,  4,  0,  0,  0,  0,  0])
        self.TOU = np.array([0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.12, 0.14, 0.14, 0.12, 0.14, 0.14, 0.14, 0.14, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.06, 0.06])
        self.E_Load = np.array([135, 133, 131, 128, 125, 120, 110, 115, 112, 109, 100, 95, 102, 109, 112, 115, 120, 130, 135, 138, 140, 142, 144, 150])
        self.E_Peak = 250
        self.E_shave = self.E_Load + self.E_EV - self.E_PV - self.E_Peak
        self.E_before_net = self.E_Load + self.E_EV - self.E_PV
        self.E_before_peak = np.array([160, 165, 165, 160, 170, 180, 190, 200, 210, 230, 270, 290, 280, 295, 275, 270, 250, 240, 230, 220, 225, 210, 200, 190])
        self.E_ev = self.E_before_peak + self.E_PV - self.E_Load
        self.lambda_EV = 1.0
        self.net_data = np.array([250.0, 250.00000000000028, 165.0, 160.1751751751749, 170.0, 180.0, 190.0, 200.0, 210.0, 230.00000000000117, 250.0, 250.0, 250.0, 250.0, 250.0, 250.0, 250.0, 240.0, 230.0, 220.0, 225.0, 250.0, 250.0, 190.0])


        self.E_EV_min = np.zeros_like(self.E_EV)
        self.E_EV_max = np.zeros_like(self.E_EV)
    def DRO_parameter(self, n_scenario, lamb, Type = 'DRO'):
        self.n_scenario = n_scenario
        self.lamb = lamb

        # if Type == 'SO':
        #     self.n_scenario = 200
        self.historical_sample = self.generating_sampling_data(n_scenario=self.n_scenario, data=self.E_EV)

        self.D, _ = self.Calculating_D(sampled_data=self.historical_sample, mean=self.E_EV)
        self.radius = self.Calculating_eps(D_list=self.D, n_scenario=self.n_scenario)

        if Type == 'RO':
            self.radius = 0.2*np.ones(len(self.radius))
        elif Type == 'SO':
            self.radius = 0.0*np.ones(len(self.radius))
        else: pass


    def MPC_parameter(self, horizon):
        self.horizon = horizon

    def generating_sampling_data(self, n_scenario, data):
        sampled_data = []
        std = 0.01
        for data_idx in range(len(data)):
            samples = np.random.normal(data[data_idx], std, n_scenario) # 평균, 표준편차, 개수
            samples = np.clip(samples, a_min=0.0, a_max=20000) # samples의 값을 a_min보다 작은 것을 a_min값으로, a_max보다 큰 것을 a_max로 바꾸자.
            sampled_data.append(samples)
            for data_idx in range(len(sampled_data)):
                self.E_EV_min[data_idx] = 0.95*np.min(sampled_data[data_idx])
                self.E_EV_max[data_idx] = 1.05*np.max(sampled_data[data_idx])

        return sampled_data


    def Calculating_D(self, sampled_data, mean):
        # 황 논문 수식 4의 루트 내부 계산.
        def y_function(x1, x2, mean):
            #rho = x1, 크사이_m = x2
            result_y = np.power(np.linalg.norm(x2 - mean*np.ones((1, self.n_scenario)), axis=0), 2)
            value = 0.0
            for i in range(self.n_scenario):
                value += np.exp(x1*np.array(result_y[i]))
            result_y = np.log10(value*(1/self.n_scenario)) + 1 #log 10??
            result_y = (2/x1)*result_y
            return result_y


        D_list, rho_list = [], []
        for time_idx in range(len(sampled_data)):
        # for data_idx in range(len(sampled_data[0])):
            x_list = np.linspace(0, 15, 1000)  #0부터 15까지 1000개 채우기
            y_list = []
            for x_idx in range(len(x_list)):
                y = y_function(x1=x_list[x_idx], x2=sampled_data[time_idx], mean=sampled_data[time_idx].mean())
                y_list.append(y)

            min_x = np.array(y_list).argmin() # y_list를 최소로 만드는 값이 어디있는지.

            rho = x_list[min_x]
            D = y_function(x1=rho, x2=sampled_data[time_idx], mean=sampled_data[time_idx].mean())

            D_list.append(D)
            rho_list.append(rho)

        return D_list, rho_list

    def Calculating_eps(self, D_list, n_scenario):
        confidence = np.log10(1/0.95)
        # print(confidence)
        eps_list = []
        for D_idx in range(len(D_list)):
            eps=D_list[D_idx]*math.sqrt((
                                         (1/n_scenario)*confidence))
            eps_list.append(eps)
        # print(eps_list)
        return eps_list

