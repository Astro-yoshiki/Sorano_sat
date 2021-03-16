#!/usr/bin/env python
# coding: utf-8
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import shap
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler


class DataAnalysis:
    """
    データ解析を行うクラス. 流れは, データの前処理 → ガウス過程回帰 → SHAP解析 → 結果の保存 となっている
    (必須)
    data_path: データファイルを指定
    (任意)
    save_path: 最適化結果の保存場所を指定
    """

    def __init__(self, data_path=None, save_path=None):
        self.data_path = data_path
        if save_path is None:
            save_path = "result/"
            if not os.path.exists(save_path):
                os.mkdir(save_path)
        self.save_path = save_path

        self.x = None
        self.y = None
        self.x_std = None
        self.y_std = None
        self.y_t = None
        self.x_scaler = None
        self.y_scaler = None
        self.model = None

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

        # 5 inputs(Temperature, Humidity, CO2, Illumination, Time) + 1 input(RGR at previous time step)
        self.x = data[:, :5]
        self.y = data[:, 5].reshape(-1, 1)  # 1 output(RGR at present time step)
        # scaling(using StandardScaler)
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.x_std = self.x_scaler.fit_transform(self.x)
        self.y_std = self.y_scaler.fit_transform(self.y)
        print("***** Preprocess finished *****")

    @staticmethod
    def rmse_score(y_true, y_pred):
        """RMSE (Root Mean Square Error: 平均二乗誤差平方根) を計算する関数"""
        mse = mean_squared_error(y_true, y_pred)
        rmse = math.sqrt(mse)
        return rmse

    def gbdt(self):
        """
        勾配ブースティングによる回帰とグリッドサーチによるモデルの探索を行う関数
        """
        model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=1000)
        # グリッドサーチの設定
        param_grid = {"max_depth": [4, 6], "learning_rate": [0.01, 0.02, 0.05, 0.1]}

        # ハイパーパラメータ探索
        model_cv = GridSearchCV(model, param_grid, cv=5, verbose=0)
        model_cv.fit(self.x, self.y)

        # 改めて最適パラメータで学習
        self.model = xgb.XGBRegressor(objective="reg:squarederror", **model_cv.best_params_)
        self.model.fit(self.x, self.y)
        print("***** Modeling finished *****")

    # FIXME: 現状勾配ブースティングはSHAPに対応していない
    def shap_analysis(self, save_figure=False):
        shap.initjs()
        data = self.x
        # Create Gradient explainer
        explainer = shap.GradientExplainer(self.model, data)
        # Extract SHAP values to explain the model predictions
        shap_values = explainer.shap_values(data)

        # Plot Feature Importance
        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(1, 2, 1)
        shap.summary_plot(shap_values, self.x, plot_type="bar", auto_size_plot=False, show=False, ax=ax1)
        # Plot Feature Importance - 'Dot' type
        ax2 = fig.add_subplot(1, 2, 2)
        shap.summary_plot(shap_values, self.x, plot_type="dot", auto_size_plot=False, show=False, ax=ax2)

        if save_figure:
            plt.savefig(self.save_path + "summary_plot.png", dpi=100, bbox_inched="tight")
        plt.close()

    def plot_feature_importance(self, save_figure=False):
        self.model.get_booster().feature_names = ["Temperature", "Humidity", "CO2", "Illumination", "Time"]
        fig, ax = plt.subplots(figsize=(10, 5))
        xgb.plot_importance(self.model.get_booster(), ax=ax)
        if save_figure:
            plt.savefig(self.save_path + "feature_importance.png", dpi=100, bbox_inched="tight")
        plt.close()


if __name__ == "__main__":
    path = "data/SoranoSat_Recipe.csv"
    analyzer = DataAnalysis(data_path=path)
    analyzer.preprocess()
    analyzer.gbdt()
    analyzer.plot_feature_importance(save_figure=True)
