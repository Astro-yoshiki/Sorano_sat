#!/usr/bin/env python
# coding: utf-8
import numpy as np
from joblib import load
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF, WhiteKernel as Wh
from sklearn.preprocessing import StandardScaler

from BO import BayesianOptimization, cut_table


class TransferLearning(BayesianOptimization):
    def __init__(self, base_path, target_path, model_path):
        super().__init__()
        self.base_path = base_path
        self.target_path = target_path
        self.model_path = model_path

        self.x = None
        self.y = None
        self.x_std = None
        self.y_std = None
        self.x_scaler = None
        self.y_scaler = None

    def calculate_difference(self, skiprows=1):
        base_data = np.loadtxt(self.base_path, delimiter=",", dtype=float, encoding="utf-8", skiprows=skiprows)
        target_data = np.loadtxt(self.target_path, delimiter=",", dtype=float, encoding="utf-8", skiprows=skiprows)
        diff_data = np.zeros(target_data.shape)
        for row in range(diff_data.shape[0]):
            diff_data[row] = target_data[row, :] - base_data[row, :]
        return diff_data

    def load_model(self):
        model = load(self.model_path)
        return model

    def preprocess(self, data):
        # 5 inputs(Temperature, Humidity, CO2, Illumination, Time)
        self.x = data[:, :5]
        self.y = data[:, 5].reshape(-1, 1)  # 1 output(growth rate)
        # scaling(using StandardScaler)
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.x_std = self.x_scaler.fit_transform(self.x)
        self.y_std = self.y_scaler.fit_transform(self.y)
        print("***** Preprocess finished ({0}) *****".format(str(self.idx)))


if __name__ == "__main__":
    # TODO: ノイズを加えたデータを作成
    _base_path = "data/SoranoSat_Data.csv"
    _target_path = "data/SoranoSat_Data_Noise.csv"
    _model_path = "result/BO/result/gp.pkl"

    # 転移学習の実行
    TL = TransferLearning(base_path=_base_path, target_path=_target_path, model_path=_model_path)
    _data = TL.calculate_difference(skiprows=1)
    TL.preprocess(data=_data)
    _kernel = C(1.0, (1e-2, 1e2)) * RBF(1.0, (1e-2, 1e2)) + Wh(0.01, (1e-2, 1e2))
    TL.gaussian_process(kernel=_kernel, save_model=False)
    TL.run_optimization()
    TL.save_result(save_history=False)
    cut_table(data_path=_base_path, line_to_cut_off=1)
