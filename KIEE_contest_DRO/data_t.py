import numpy as np
import random
import math
from docplex.mp.model import Model
from scipy.special import gamma
from scipy.stats import rv_histogram, wasserstein_distance, rv_continuous

class data(object):
    def __init__(self, time, unit, seed):
        np.random.seed(seed)
        random.seed(seed)

        self.Time = time
        self.unit = unit
        self.T = self.time_resol_cal(UNIT=self.unit, HOUR=self.Time)
        self.pi_24 = 1300 * np.array([0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.12, 0.14, 0.14, 0.12,
                           0.14, 0.14, 0.14, 0.14, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.06, 0.06])
        self.pi_buy = self.No_amp_time_scale(DATA=self.pi_24, NEW_lst_num=self.T)
        self.pi_sell = 1.5*self.No_amp_time_scale(DATA=self.pi_24, NEW_lst_num=self.T)

        self.E_PV_24 = 4 * np.array([0, 0, 0, 0, 0, 0, 0.05, 0.45, 1.37, 2.71, 3.91, 5.08, 5.19, 4.90, 4.33, 3.27, 1.95, 0.78, 0.26, 0.06, 0.00, 0.00, 0.00, 0.00])
        self.E_PV = self.interpolation(LOAD_lst=self.E_PV_24, TIME_slot=self.T)

        self.E_fixed_24  = 4 * np.array([3.09, 3.29, 3.17, 3.28, 3.15, 2.79, 2.83, 3.18, 5.69, 15.03, 17.10, 17.05, 13.40, 11.24, 16.21, 15.16,15.24, 8.71, 5.55, 6.38, 6.77, 6.32, 5.10, 4.26])
        self.E_fixed = self.interpolation(LOAD_lst=self.E_fixed_24, TIME_slot=self.T)


        self.E_sell_24 = (1 / 65) * np.array([2800, 3200, 2320, 1600, 1520, 2240, 3120, 3800, 8040, 9200, 10880, 10920, 11120, 12040, 12040, 11080,12320, 11800, 9800, 8160, 6880, 6080, 4360, 3800])
        self.E_sell  = self.interpolation(LOAD_lst=self.E_sell_24, TIME_slot=self.T)


        self.E_buy_24 = 0.26 * self.E_sell_24
        self.E_buy = self.interpolation(LOAD_lst=self.E_buy_24, TIME_slot=self.T)

        self.E_set = np.mean(self.E_fixed)

        self.ESS_capacity = 200
        self.SOE_max = 0.95*self.ESS_capacity
        self.SOE_min = 0.1*self.ESS_capacity
        self.SOE_ini = 85
        self.Ch_min = 0
        self.Ch_max = 50
        self.Dch_min = 0
        self.Dch_max = 50
        self.eff = 0.999
        self.eps = 1.0

        self.E_sell_min=[]
        self.E_sell_max = []
        self.E_buy_min = []
        self.E_buy_max = []
        self.E_PV_min = []
        self.E_PV_max = []

    def DRO_parameter(self, n_scenario, lamb, Type = 'DRO'):
        self.n_scenario = n_scenario
        self.lamb = lamb

        self.historical_sample_E_sell = self.generating_sampling_data_E_sell(n_scenario=self.n_scenario, data=self.E_sell)
        self.historical_sample_E_buy = self.generating_sampling_data_E_sell(n_scenario=self.n_scenario, data=self.E_buy)
        self.historical_sample_E_PV = self.generating_sampling_data_E_PV(n_scenario=self.n_scenario, data=self.E_PV)

        for data_idx in range(len(self.historical_sample_E_sell)):
            self.E_sell_min = np.append(self.E_sell_min, 0.8 * np.min(self.historical_sample_E_sell[data_idx]))
            self.E_sell_max = np.append(self.E_sell_max, 1.2 * np.max(self.historical_sample_E_sell[data_idx]))

        for data_idx in range(len(self.historical_sample_E_buy)):
            self.E_buy_min = np.append(self.E_buy_min,0.8 * np.min(self.historical_sample_E_buy[data_idx]))
            self.E_buy_max = np.append(self.E_buy_max,1.2 * np.max(self.historical_sample_E_buy[data_idx]))

        for data_idx in range(len(self.historical_sample_E_PV)):
            self.E_PV_min = np.append(self.E_PV_min,0.8 * np.min(self.historical_sample_E_PV[data_idx]))
            self.E_PV_max = np.append(self.E_PV_max,1.2 * np.max(self.historical_sample_E_PV[data_idx]))

        #E_sell
        self.D_E_sell, _ = self.Calculating_D(sampled_data=self.historical_sample_E_sell, mean=self.E_sell)
        self.radius_E_sell = self.Calculating_eps(D_list=self.D_E_sell, n_scenario=self.n_scenario)

        #E_buy
        self.D_E_buy, _ = self.Calculating_D(sampled_data=self.historical_sample_E_buy, mean=self.E_buy)
        self.radius_E_buy = self.Calculating_eps(D_list=self.D_E_buy, n_scenario=self.n_scenario)

        # E_PV
        self.D_E_PV, _ = self.Calculating_D(sampled_data=self.historical_sample_E_PV, mean=self.E_PV)
        self.radius_E_PV = self.Calculating_eps(D_list=self.D_E_PV, n_scenario=self.n_scenario)

        if Type == 'RO':
            self.radius_E_sell = 9999999999999999999999999999999999999999999999999999999999 * np.ones(len(self.radius_E_sell))
            self.radius_E_buy =  9999999999999999999999999999999999999999999999999999999999 * np.ones(len(self.radius_E_buy))
        elif Type == 'SO':
            self.radius_E_sell = 0 * np.ones(len(self.radius_E_sell))
            self.radius_E_buy = 0*  np.ones(len(self.radius_E_buy))
        else: pass

    def generating_sampling_data_E_sell(self, n_scenario, data):
        sampled_data = []
        #std = 0.01
        mean = 22
        standard = 2
        for data_idx in range(len(data)):
            #samples = np.random.normal(data[data_idx], std, n_scenario)
            samples = np.random.normal(mean, standard, n_scenario)
            samples = np.clip(samples, a_min=0.0, a_max=20000)
            sampled_data.append(samples)

        return sampled_data

    def generating_sampling_data_E_PV(self, n_scenario, data):
        sampled_data = []
        std = 0.01
        for data_idx in range(len(data)):
            samples = np.random.normal(data[data_idx], std, n_scenario)
            samples = np.clip(samples, a_min=0.0, a_max=20000)
            sampled_data.append(samples)

        return sampled_data

    def Calculating_D(self, sampled_data, mean):
        # 황 논문 수식 4의 루트 내부 계산.
        def y_function(x1, x2, mean):
            #rho = x1, 크사이_m = x2
            result_y = np.power(np.linalg.norm(x2 - mean*np.ones((1,self.n_scenario)), axis=0), 2)
            value = 0.0
            for i in range(self.n_scenario):
                value += np.exp(x1*np.array(result_y[i]))
            result_y = np.log10(value*(1/self.n_scenario)) + 1
            result_y = (2/x1)*result_y
            return  result_y


        D_list, rho_list = [], []
        for time_idx in range(len(sampled_data)):
        # for data_idx in range(len(sampled_data[0])):
            x_list = np.linspace(0, 3, 1000)
            y_list = []
            for x_idx in range(len(x_list)):
                y = y_function(x1=x_list[x_idx], x2=sampled_data[time_idx], mean=sampled_data[time_idx].mean())
                y_list.append(y)

            min_x = np.array(y_list).argmin()

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
                                         (1/n_scenario)*confidence)
                                                                    )
            eps_list.append(eps)
        # print(eps_list)
        return eps_list

    def MPC_parameter(self, horizon, time_step):
        self.horizon = horizon
        self.time_step = time_step
#쪼갠시간에 맞게 Total time period 계산해주는 것.
    def time_resol_cal(self, UNIT=int, HOUR=int):
        num_hour = int(60/UNIT)
        total_time = HOUR*num_hour
        return total_time

    #total_time = time_resol_cal(UNIT=30, HOUR=24)
    #print(total_time)

    #분할없이 변경된 Total time period에 맞게 바꿔주는 것.
    def No_amp_time_scale(self, DATA, NEW_lst_num):
        NEW_lst = np.ones(NEW_lst_num)
        mul = int(NEW_lst_num / len(DATA))
        for i in range(len(DATA)):
            for j in range(mul):
                NEW_lst[(i * mul) + j] = DATA[i]
        return NEW_lst


    def interpolation(self, LOAD_lst, TIME_slot):
        # 예를 들어, LOAD_lst를 1X3 이라고 하자.
        # 내가 1X15로 바꾸고 싶다고 해보자.

        origin_Time_slot = len(LOAD_lst)
        new_lst = np.ones(TIME_slot)
        amp = int(TIME_slot / origin_Time_slot)
        alpha_denomi = amp - 1

        for i in range(len(LOAD_lst)):
            # 알파 구하기
            if i < (len(LOAD_lst) - 1):
                diff = LOAD_lst[i + 1] - LOAD_lst[i]
            else:
                diff = 0.0 - LOAD_lst[i]

            # 새 리스트에 추가하기
            for j in range(amp):
                new_lst[(i * amp) + j] = LOAD_lst[i] + (diff / amp) * j

        result_lst = (1 / amp) * new_lst

        return result_lst

    #b=interpolation(LOAD_lst=data.E_PV,TIME_slot=total_time)
    #print(b)
    #print(len(b))
