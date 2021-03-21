#!/usr/bin/env python
# coding: utf-8
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import xgboost as xgb
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_crossover, get_mutation, get_sampling, get_termination
from pymoo.model.problem import Problem
from pymoo.optimize import minimize
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

plt.rcParams['font.size'] = 16
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.linewidth'] = 0.3


class DataAnalysis:
    """
    データ解析を行うクラス. 流れは, データの前処理 → ガウス過程回帰 → SHAP解析 → 結果の保存 となっている
    (必須)
    data_path: データファイルを指定
    (必須)
    """

    def __init__(self, data_path=None):
        self.data_path = data_path
        self.x = None
        self.x_index = None
        self.y = None
        self.model = None

    def preprocess(self):
        """
        データの前処理(入力と出力を分割・正規化)を行う関数
        (任意)
        skiprows: 行のラベルが含まれている場合はTrue, そうでない場合はFalseを指定
        """
        # load data
        data = np.loadtxt(self.data_path, delimiter=",", dtype=float, skiprows=1)
        # 5 inputs(Temperature, Humidity, CO2, Illumination, Time)
        self.x = data[:, :5]
        self.y = data[:, 5].reshape(-1, 1)  # 1 output(RGR at present time step)
        print("***** Preprocess finished *****")

    @staticmethod
    def rmse_score(y_true, y_pred):
        """RMSE (Root Mean Square Error: 平均二乗誤差平方根) を計算する関数"""
        mse = mean_squared_error(y_true, y_pred)
        rmse = math.sqrt(mse)
        return rmse

    def gbdt(self, return_model=True):
        """
        勾配ブースティングによる回帰とグリッドサーチによるモデルの探索を行う関数
        """
        model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=1000)
        # グリッドサーチの設定
        param_grid = {"max_depth": [4, 6], "learning_rate": [0.01, 0.02, 0.05, 0.1]}

        # ハイパーパラメータ探索
        model_cv = GridSearchCV(model, param_grid, cv=5, iid=True, verbose=0)
        model_cv.fit(self.x, self.y)

        # 改めて最適パラメータで学習
        self.model = xgb.XGBRegressor(objective="reg:squarederror", **model_cv.best_params_)
        self.model.fit(self.x, self.y)
        print("***** Modeling finished *****")

        if return_model:
            return self.model


def predict_growth(x, model):
    """
    遺伝的アルゴリズムを実行するために,インプットからアウトプットを予測する関数を用意しておく
    """
    x = np.array(x).reshape(1, -1)  # 入力されたパラメータをndarrayに変換
    y_pred = model.predict(x)
    return y_pred


class GA:
    def __init__(self, problem, pop_size=100, n_offsprings=100, n_gen=100, save_path=None):
        self.problem = problem
        self.pop_size = pop_size
        self.n_offsprings = n_offsprings
        self.n_gen = n_gen
        if save_path is None:
            save_path = "result/"
            if not os.path.exists(save_path):
                os.mkdir(save_path)
        self.save_path = save_path

        self.algorithm = None
        self.termination = None
        self.res = None

    def set_algorithm(self):
        self.algorithm = NSGA2(
            pop_size=self.pop_size,
            n_offsprings=self.n_offsprings,
            sampling=get_sampling("real_lhs"),
            crossover=get_crossover("real_sbx", prob=0.9, eta=15),
            mutation=get_mutation("real_pm", eta=20),
            eliminate_duplicates=True
        )
        self.termination = get_termination("n_gen", self.n_gen)

    def execute(self, verbose=True):
        print("***** Genetic Algorithm Start *****")
        self.res = minimize(self.problem, self.algorithm, self.termination, pf=None, save_history=True,
                            verbose=verbose, seed=0)
        print("***** Genetic Algorithm Finished *****")

    def plot_result(self):
        columns = ["Temperature", "Humidity", "CO2", "Illumination", "Time", "Growth_rate"]
        header = ",".join(columns)
        # 最適解をcsvファイルに保存
        res_x = np.array(self.res.X)
        res_f = np.array(self.res.F) * (-1)  # 元の値に戻す
        optimal = np.hstack([res_x, res_f])
        np.savetxt(self.save_path + "optimal_NSGA2.csv", optimal, delimiter=",", header=header)

        # 並行座標表示
        df = pd.DataFrame(optimal, index=None, columns=columns)
        fig = px.parallel_coordinates(df,
                                      dimensions=columns,
                                      color_continuous_scale=px.colors.diverging.Tealrose,
                                      height=400, width=800)
        fig.write_html(self.save_path + 'parallel_coordinates_px.html', auto_open=True)
        plt.close()


# define your problem
class MyProblem(Problem):
    def __init__(self, model):
        self.model = model
        super().__init__(n_var=5,
                         n_obj=1,
                         n_constr=0,
                         xl=np.array([20.0, 45.0, 400.0, 100.0, 8.0]),
                         xu=np.array([35.0, 80.0, 1200.0, 255.0, 23.0]),
                         elementwise_evaluation=True)

    def _evaluate(self, x, out, *args, **kwargs):
        # set objective functions
        growth_rate = predict_growth(x, self.model)
        f1 = growth_rate * (-1)  # 最大化問題に変換するための処理

        # TODO: 制約条件についてディスカッションしたい(Timeに関しては制約つけられるよね！)
        # set constraints
        # g1 = 2 * (x[0] - 0.1) * (x[0] - 0.9) / 0.18
        # g2 = - 20 * (x[0] - 0.4) * (x[0] - 0.6) / 4.8

        # F is objective function, G is constraint
        out["F"] = [f1]
        # out["G"] = [g1, g2]


if __name__ == "__main__":
    _data_path = "data/SoranoSat_Recipe.csv"
    _save_path = "result/"
    # modeling part
    analyzer = DataAnalysis(data_path=_data_path)
    analyzer.preprocess()
    clf = analyzer.gbdt(return_model=True)

    # genetic algorithm part
    prob = MyProblem(clf)
    ga = GA(problem=prob, pop_size=100, n_offsprings=100, n_gen=100, save_path=_save_path)
    ga.set_algorithm()
    ga.execute(verbose=True)
    ga.plot_result()
