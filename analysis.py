#!/usr/bin/env python
# coding: utf-8
import math
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import shap
import xgboost as xgb
from interpret import show
from interpret.data import Marginal
from interpret.glassbox import ExplainableBoostingRegressor, LinearRegression, RegressionTree
from interpret.perf import RegressionPerf
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

# graph setting
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
        self.x_index = None
        self.y = None
        self.model = None

    def preprocess(self, data_sum=False):
        """
        データの前処理(入力と出力を分割・正規化)を行う関数
        (任意)
        skiprows: 行のラベルが含まれている場合はTrue, そうでない場合はFalseを指定
        """
        # load data
        data = pd.read_csv(self.data_path)
        # 5 inputs(Temperature, Humidity, CO2, Illumination, Time)
        self.x = data.iloc[:, :5]
        self.x_index = self.x.columns.to_list()
        self.y = data.iloc[:, [5]]  # 1 output(RGR at present time step)

        if data_sum:
            column_names = self.x.columns.to_list()
            add_name = [name + "_sum" for name in column_names]
            x_sum = pd.DataFrame(np.zeros([len(data), len(self.x.columns)]), columns=add_name)
            for row in range(1, len(data)+1):
                x_extracted = self.x.iloc[:row].values
                x_sum.iloc[row-1] = np.sum(x_extracted, axis=0)
            self.x = pd.concat([self.x, x_sum], axis=1)
            self.x_index += add_name
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
        model_cv = GridSearchCV(model, param_grid, cv=5, iid=True, verbose=0)
        model_cv.fit(self.x, self.y)

        # 改めて最適パラメータで学習
        self.model = xgb.XGBRegressor(objective="reg:squarederror", **model_cv.best_params_)
        self.model.fit(self.x, self.y)
        print("***** Modeling finished *****")

    def ga2m(self):
        # Explore the Data
        marginal = Marginal().explain_data(self.x, self.y, name="Raw Data")

        # Train the Explainable Boosting Machine(EBM)
        lr = LinearRegression()
        lr.fit(self.x, self.y)

        rt = RegressionTree()
        rt.fit(self.x, self.y)

        ebm = ExplainableBoostingRegressor()  # For Classifier, use ebm = ExplainableBoostingClassifier()
        ebm.fit(self.x, self.y)

        # How Does the EBM Model Perform?
        ebm_perf = RegressionPerf(ebm.predict).explain_perf(self.x, self.y, name="EBM")
        lr_perf = RegressionPerf(lr.predict).explain_perf(self.x, self.y, name="Linear Regression")
        rt_perf = RegressionPerf(rt.predict).explain_perf(self.x, self.y, name="Regression Tree")

        # Global Interpretability - What the Model says for All Data
        ebm_global = ebm.explain_global(name="EBM")
        lr_global = lr.explain_global(name="LinearRegression")
        rt_global = rt.explain_global(name="Regression Tree")

        # Put All in a Dashboard - This is the best
        show([marginal, lr_global, lr_perf, rt_global, rt_perf, ebm_perf, ebm_global])

    def shap_analysis(self, save_figure=False):
        shap.initjs()
        data = self.x
        # Create Tree explainer
        explainer = shap.TreeExplainer(self.model, data)
        # Extract SHAP values to explain the model predictions
        shap_values = explainer.shap_values(data)

        # Plot Feature Importance - 'violin' type
        shap.summary_plot(shap_values, self.x, plot_type="violin", plot_size=(13, 5), show=False)

        if save_figure:
            plt.savefig(self.save_path + "summary_plot.png", dpi=100, bbox_inched="tight")
        plt.close()

    def plot_feature_importance(self, save_figure=False):
        self.model.get_booster().feature_names = self.x_index
        fig, ax = plt.subplots(figsize=(20, 5))
        xgb.plot_importance(self.model.get_booster(), ax=ax)
        if save_figure:
            plt.savefig(self.save_path + "feature_importance.png", dpi=100, bbox_inched="tight")
        plt.close()


if __name__ == "__main__":
    path = "data/SoranoSat_Recipe.csv"
    analyzer = DataAnalysis(data_path=path)
    analyzer.preprocess(data_sum=True)
    analyzer.gbdt()
    analyzer.shap_analysis(save_figure=True)
    analyzer.plot_feature_importance(save_figure=True)
