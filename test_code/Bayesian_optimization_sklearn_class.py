#!/usr/bin/env python
# coding: utf-8
import os

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF, WhiteKernel as Wh
from sklearn.preprocessing import StandardScaler
from skopt import gp_minimize


class BayesianOptimization:
    def __init__(self, datapath, savepath="result", flag_maximize=True, n_calls=100):
        self.datapath = datapath
        self.savepath = savepath
        self.flag_maximize = flag_maximize
        self.n_calls = n_calls

    def preprocess(self, skiprows=True):
        # load data
        if skiprows:
            data = np.loadtxt(self.datapath, delimiter=",", dtype=float, encoding="utf-8", skiprows=1)
        else:
            data = np.loadtxt(self.datapath, delimiter=",", dtype=float, encoding="utf-8", skiprows=0)
        self.x = data[:,:6] # 5 inputs(Temperature, Humidity, CO2, Illumination, Time) + 1(RGR at previous time step)
        self.y = data[:,6].reshape(-1,1) # 1 output(RGR at present time step)
        self.y_t = self.y[-1] # extract RGR at latest time step
        # scaling(using standard scaler)
        self.x_sc = StandardScaler()
        self.y_sc = StandardScaler()
        x_std = self.x_sc.fit_transform(self.x); self.x_std = x_std
        y_std = self.y_sc.fit_transform(self.y); self.y_std = y_std
        return self.x_std, self.y_std

    def fit(self, kernel):
        self.reg = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=30)
        self.reg.fit(self.x_std, self.y_std)

    def f(self, x):
        x = np.array(x).reshape(1,-1)
        x_std = self.x_sc.transform(x)
        y_std = self.reg.predict(x_std); y = self.y_sc.inverse_transform(y_std)[0][0]
        if self.flag_maximize:
            return y * (-1)
        else:
            return y

    def optimization(self):
        # range of parameter
        spaces = [
            (20.0, 35.0, "uniform"),  # Temperature
            (40.0, 80.0, "uniform"),  # Humidity
            (400.0, 1200.0, "uniform"),  # CO2
            (100.0, 255.0, "uniform"),  # Illumination
            (8.0, 23.0, "uniform"),  # Time
            (self.y_t)  # RGR at the latest time step(fixed)
        ]
        self.res = gp_minimize(self.f, spaces, acq_func="EI", n_points=10000, n_calls=self.n_calls,
                               model_queue_size=1, n_jobs=-1, verbose=False)
        return self.res

    def show_result(self):
        # extract best output(max or min)
        if self.flag_maximize:
            opt_fx = self.res.fun * (-1)
        else:
            opt_fx = self.res.fun
        # extract best input
        opt_x = self.res.x
        print("Best value is {}".format(opt_fx))
        print("Best input is {}".format(opt_x))

    def save_result(self):
        # save results to a csv file
        columns_name_list = ["Temperature", "Humidity", "CO2", "Illumination", "Time", "Y_t"]  # column name
        columns_name_list_str = ",".join(columns_name_list)  # put a comma between elements to a string
        columns_name_list_str = columns_name_list_str + ",Y_t+1" + "\n"  # insert the target name and line feed code
        if not os.path.exists(self.savepath):
            os.mkdir(self.savepath)
        with open(self.savepath + "/opt_data.csv", "w") as f:
            f.write(columns_name_list_str)
            buf_list = []
            for X in self.res.x:
                buf_list.append(X)
            buf_list.append(self.res.fun)
            opt_list = list(map(str, buf_list))
            opt_str = ','.join(opt_list)
            f.write(opt_str + "\n")
        # save history to a csv file
        history_X = self.res.x_iters[0:self.n_calls]
        if self.flag_maximize:
            history_Y = self.res.func_vals[0:self.n_calls] * (-1)
        else:
            history_Y = self.res.func_vals[0:self.n_calls]
        history_Y_list = history_Y.tolist()
        history_X_list = history_X
        with open(self.savepath + "/history_data.csv", "w") as f:
            f.write(columns_name_list_str)
            for X, Y in zip(history_X_list, history_Y_list):
                X.append(Y)
                buf_list = list(map(str, X))
                buf_str = ','.join(buf_list)
                f.write(buf_str + '\n')


if __name__ == "__main__":
    datapath = "data.csv"
    BO = BayesianOptimization(datapath)
    BO.preprocess()
    kernel = C(1.0, (1e-2,1e2)) * RBF(1.0, (1e-2,1e2)) + Wh(0.01, (1e-2,1e2))
    BO.fit(kernel=kernel)
    BO.optimization()
    BO.show_result()
    BO.save_result()

