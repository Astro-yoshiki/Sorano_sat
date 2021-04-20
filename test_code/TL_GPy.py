#!/usr/bin/env python
# coding: utf-8
import os

import numpy as np
from joblib import load
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF, WhiteKernel as Wh
from sklearn.preprocessing import StandardScaler
from skopt import gp_minimize

from BO_GPy import BayesianOptimization, cut_table


class TransferLearning(BayesianOptimization):
    def __init__(self, base_path, target_path, model_path, x_scaler_path, y_scaler_path, save_path=None):
        super().__init__(save_path, max_iter=100, idx=0)
        self.data_path = base_path
        self.base_path = base_path
        self.target_path = target_path
        self.model_path = model_path
        self.x_scaler_path = x_scaler_path
        self.y_scaler_path = y_scaler_path
        if save_path is None:
            save_path = "result/TL/"
            if not os.path.exists(save_path):
                os.mkdir(save_path)
        self.save_path = save_path

        self.x = None
        self.y = None
        self.x_std = None
        self.y_std = None
        self.x_scaler_base = None
        self.y_scaler = None
        self.y_scaler_base = None
        self.model_base = None

    def calculate_difference(self, skiprows=1):
        base_data = np.loadtxt(self.base_path, delimiter=",", dtype=float, encoding="utf-8", skiprows=skiprows)
        target_data = np.loadtxt(self.target_path, delimiter=",", dtype=float, encoding="utf-8", skiprows=skiprows)
        diff_data = target_data.copy()
        for row in range(diff_data.shape[0]):
            # Growth rateの差分を計算
            diff_data[row, -1] = target_data[row, -1] - base_data[row, -1]
        return diff_data

    def load_model_and_scaler(self):
        try:
            self.model_base = load(self.model_path)
            self.x_scaler_base = load(self.x_scaler_path)
            self.y_scaler_base = load(self.y_scaler_path)
        except (FileNotFoundError, TypeError) as e:
            print(e)

    def preprocess_data(self, data):
        # 5 inputs(Temperature, Humidity, CO2, Illumination, Time)
        self.x = data[:, :5]
        self.y = data[:, 5].reshape(-1, 1)  # 1 output(Growth rate)
        # scaling(using StandardScaler)
        self.y_scaler = StandardScaler()
        self.x_std = self.x_scaler_base.fit_transform(self.x)
        self.y_std = self.y_scaler.fit_transform(self.y)
        print("***** Preprocess finished ({0}) *****".format(str(self.idx)))

    def objective_function_diff(self, x):
        """
        目的関数(最適化の際に最大化 or 最小化をする関数)を与える
        ここでは, RGRの最大化を目的関数として設定している
        (注意：skoptは最小化問題しか扱えないため, 最大化問題は-1を掛けることで最小化問題に変換している)
        """
        x = np.array(x).reshape(1, -1)
        x_std = self.x_scaler_base.transform(x)
        y_diff_std = self.model.predict(x_std)  # 差分を予測
        y_diff = self.y_scaler.inverse_transform(y_diff_std)[0][0]
        y_base_std = self.model_base.predict(x_std)  # 転移前のモデルを用いた予測
        y_base = self.y_scaler_base.inverse_transform(y_base_std)[0][0]
        y_tl = y_base + y_diff  # 和を取ることで転移学習を行う
        if self.flag_maximize:
            return y_tl * (-1)
        else:
            return y_tl

    def run_optimization(self):
        """
        ベイズ最適化を実行する関数
        gp_minimizeの引数についてはコチラを参照(https://scikit-optimize.github.io/stable/modules/generated/skopt.gp_minimize.html#skopt.gp_minimize)
        """
        # range of parameter
        spaces = [
            (20.0, 35.0, "uniform"),  # Temperature
            (45.0, 80.0, "uniform"),  # Humidity
            (400.0, 1200.0, "uniform"),  # CO2
            (100.0, 255.0, "uniform"),  # Illumination
            (8.0, 23.0, "uniform"),  # Time
        ]
        # run optimization
        print("***** Optimization start ({0}) *****".format(str(self.idx)))
        self.res = gp_minimize(self.objective_function_diff, spaces, acq_func="gp_hedge", acq_optimizer="sampling",
                               n_points=50000, n_calls=self.max_iter, model_queue_size=1, n_jobs=-1, verbose=True)


if __name__ == "__main__":
    _base_path = "data/SoranoSat_Data.csv"
    _target_path = "data/SoranoSat_Data_Noise.csv"
    _model_path = "result/BO/gp.pkl"
    _x_scaler_path = "result/BO/x_scaler.joblib"
    _y_scaler_path = "result/BO/y_scaler.joblib"

    # 転移学習の実行
    TL = TransferLearning(base_path=_base_path, target_path=_target_path, model_path=_model_path,
                          x_scaler_path=_x_scaler_path, y_scaler_path=_y_scaler_path)
    _data = TL.calculate_difference(skiprows=1)
    TL.load_model_and_scaler()
    TL.preprocess_data(data=_data)
    _kernel = C(1.0, (1e-2, 1e2)) * RBF(1.0, (1e-2, 1e2)) + Wh(0.01, (1e-2, 1e2))
    TL.gaussian_process(kernel=_kernel, save_model=False)
    TL.plot_prediction()
    TL.run_optimization()
    TL.save_result(save_history=False)
    cut_table(data_path=_base_path, line_to_cut_off=1)
