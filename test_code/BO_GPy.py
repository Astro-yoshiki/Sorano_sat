#!/usr/bin/env python
# coding: utf-8
import datetime
import os
import shutil

import GPy
import GPyOpt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from GPy.kern import RBF
from joblib import dump
from sklearn.preprocessing import StandardScaler


def cut_table(data_path, line_to_cut_off=None):
    """Function to arrange table(delete the last three lines)"""
    df = pd.read_csv(data_path)
    df_original = df.iloc[:line_to_cut_off * (-1), :]
    df_original.to_csv(data_path, index=False)


class DateTimeFlag:
    """Class to get the current time"""
    def __init__(self):
        self.now = datetime.datetime.now()

    def get_flag(self):
        return "{0:%Y%m%d}".format(self.now)


class BayesianOptimization:
    """
    ベイズ最適化を行うクラス. 流れは, データの前処理 → ガウス過程回帰 → 最適化 → 結果の表示 → 結果の保存 となっている
    (必須)
    data_path: データファイルを指定
    (任意)
    save_path: 最適化結果の保存場所を指定
    flag_maximize: Trueであれば最大化, Falseであれば最小化
    max_iter: 最適化を何回繰り返すか(デフォルトは100回)
    """

    def __init__(self, data_path=None, save_path=None, flag_maximize=True, max_iter=100, idx=None):
        self.data_path = data_path
        if save_path is None:
            save_path = "result/BO/"
            if not os.path.exists(save_path):
                os.mkdir(save_path)
        self.save_path = save_path
        self.flag_maximize = flag_maximize
        self.max_iter = max_iter
        self.idx = idx + 1

        date = DateTimeFlag()
        self.date = date.get_flag()

        self.x = None
        self.y = None
        self.dim = None
        self.x_std = None
        self.y_std = None
        self.x_scaler = None
        self.y_scaler = None
        self.model = None
        self.res = None

    def preprocess(self, skiprows=True):
        """
        データの前処理(入力と出力を分割・正規化)を行う関数
        (任意)
        skiprows: 行のラベルが含まれている場合はTrue, そうでない場合はFalseを指定
        """
        # load data
        if skiprows:
            data = np.loadtxt(self.data_path, delimiter=",", dtype=float, encoding="utf-8", skiprows=1)
        else:
            data = np.loadtxt(self.data_path, delimiter=",", dtype=float, encoding="utf-8", skiprows=0)

        # 5 inputs(Temperature, Humidity, CO2, Illumination, Time)
        self.x = data[:, :5]
        self.y = data[:, 5].reshape(-1, 1)  # 1 output(growth rate)
        self.dim = self.x.shape[1]
        # scaling(using StandardScaler)
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.x_std = self.x_scaler.fit_transform(self.x)
        self.y_std = self.y_scaler.fit_transform(self.y)
        # save scaler(for transfer learning)
        dump(self.x_scaler, self.save_path + "x_scaler.joblib")
        dump(self.y_scaler, self.save_path + "y_scaler.joblib")
        print("***** Preprocess finished ({0}) *****".format(str(self.idx)))

    def gaussian_process(self, kernel=None, save_model=False):
        """
        ガウス過程回帰を行う関数
        (必須)
        kernel: ガウス過程におけるカーネルを設定する
        save_model: 作成したモデルを保存する場合はTrue, しない場合はFalseを指定
        """
        if kernel is None:
            kernel = RBF(self.dim)
        self.model = GPy.models.GPRegression(self.x_std, self.y_std, kernel=kernel)
        self.model.optimize(messages=True, max_iters=10000)
        if save_model:
            dump(self.model, self.save_path + "gp.pkl")
        print("***** Gaussian Process finished ({0}) *****".format(str(self.idx)))

    def plot_prediction(self):
        y_pred_std, _ = self.model.predict(self.x_std)
        y_pred = self.y_scaler.inverse_transform(y_pred_std)

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(self.y, y_pred, "bo", label="Expected value")
        ax.plot([min(self.y), max(self.y)], [min(self.y), max(self.y)], color="black", linestyle="dashed")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Prediction")
        ax.set_title("Prediction by GPR")
        ax.legend(loc="best")

        # show plots
        fig.tight_layout()
        plt.savefig(self.save_path + "prediction.png", dpi=100)
        plt.close()

    def objective_function(self, x):
        """
        目的関数(最適化の際に最大化 or 最小化をする関数)を与える
        ここでは, RGRの最大化を目的関数として設定している
        (注意：skoptは最小化問題しか扱えないため, 最大化問題は-1を掛けることで最小化問題に変換している)
        """
        x = np.array(x).reshape(1, -1)
        x_std = self.x_scaler.transform(x)
        y_std = self.model.predict(x_std)
        y = self.y_scaler.inverse_transform(y_std)[0]
        return y

    def run_optimization(self):
        """
        ベイズ最適化を実行する関数
        """
        print("***** Optimization start ({0}) *****".format(str(self.idx)))
        # range of parameter
        bounds = [
            # If continuous value, set "type" to "continuous" and set the value as (lower limit, upper limit).
            {"name": "Temperature", "type": "continuous", "domain": (20.0, 35.0)},
            {"name": "Humidity", "type": "continuous", "domain": (45.0, 80.0)},
            {"name": "CO2", "type": "continuous", "domain": (400.0, 1200.0)},
            {"name": "Illumination", "type": "continuous", "domain": (100.0, 255.0)},
            {"name": "Time", "type": "continuous", "domain": (8.0, 23.0)}
        ]
        res = GPyOpt.methods.BayesianOptimization(
            f=self.objective_function,
            domain=bounds,
            # constraints = constraints, # in case constrains needed
            maximize=self.flag_maximize,  # maximize or minimize
            model_type="GP",  # default "GP"
            initial_design_type="latin",
            acquisition_type="EI",
            num_cores=-1
        )
        # run optimization
        res.run_optimization(max_iter=self.max_iter)
        self.res = res

    def show_result(self):
        """
        最適解を表示する関数. Best inputが最適な条件の組み合わせを, Best valueがその条件で実験したときの予測値をそれぞれ意味する
        """
        # extract best output
        if self.flag_maximize:
            opt_fx = self.res.fx_opt * (-1)
        else:
            opt_fx = self.res.fx_opt
        # extract best input
        opt_x = self.res.x_opt
        print("Best value is {}".format(opt_fx))
        print("Best input is {}".format(opt_x))

    def save_result(self, save_history=False):
        """
        最適解と最適化過程を保存する関数. ファイル形式は.csvとしている
        1. SoranoSat_recipe_ラズパイ番号_日付.csv: 各ラズパイに送る最適解
        2. SoranoSat_OptData_日付.csv：最適化された条件を含む教師データ
        3. SoranoSat_History_日付.csv: 最適化過程(任意)
        """
        # save results to a csv file
        columns_name_list = ["Temperature", "Humidity", "CO2", "Illumination", "Time"]  # column name
        columns_name_list_str = ",".join(columns_name_list)  # put a comma between elements to a string
        columns_name_list_str = columns_name_list_str + ",Growth_rate" + "\n"  # insert target name and line feed code
        with open(self.save_path + "SoranoSat_recipe_{0}_{1}.csv".format(str(self.idx), self.date), "w") as f:
            f.write(columns_name_list_str)
            buf_list = []
            for x in self.res.x_opt:
                buf_list.append(x)
            buf_list.append(self.res.fx_opt * (-1))
            opt_list = list(map(str, buf_list))
            opt_str = ','.join(opt_list)
            f.write(opt_str + "\n")

        # copy master data to save directory and add results
        with open(self.data_path, "a") as f:
            buf_list = []
            for x in self.res.x_opt:
                buf_list.append(x)
            buf_list.append(self.res.fx_opt * (-1))
            opt_list = list(map(str, buf_list))
            opt_str = ",".join(opt_list)
            f.write(opt_str + "\n")
        shutil.copy(self.data_path, self.save_path + "SoranoSat_OptData_{0}.csv".format(self.date))

        # save history to a csv file
        if save_history:
            history_x = self.res.X[0:self.max_iter]
            if self.flag_maximize:
                history_y = self.res.Y[0:self.max_iter] * (-1)
            else:
                history_y = self.res.Y[0:self.max_iter]
            history_y_list = history_y.tolist()
            history_x_list = history_x.tolist()
            with open(self.save_path + "SoranoSat_History_{0}.csv".format(self.date), "w") as f:
                f.write(columns_name_list_str)
                for X, Y in zip(history_x_list, history_y_list):
                    X.append(Y)
                    buf_list = list(map(str, X))
                    buf_str = ','.join(buf_list)
                    f.write(buf_str + '\n')
        print("***** Optimization finished ({0}) *****".format(str(self.idx)))


if __name__ == "__main__":
    _data_path = "data/SoranoSat_Data.csv"
    iteration = 3
    for i in range(iteration):
        BO = BayesianOptimization(data_path=_data_path, max_iter=100, idx=i)
        BO.preprocess()
        BO.gaussian_process(save_model=True)
        BO.plot_prediction()
        BO.run_optimization()
        BO.show_result()
        BO.save_result(save_history=False)
    cut_table(data_path=_data_path, line_to_cut_off=iteration)