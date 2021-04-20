#!/usr/bin/env python
# coding: utf-8
import numpy as np

data_path = "../data/SoranoSat_Data.csv"
save_path = "../data/SoranoSat_Data_Noise.csv"

label = np.loadtxt(data_path, delimiter=",", dtype=str, encoding="utf-8", skiprows=0)[0]
header = ",".join(label)
data = np.loadtxt(data_path, delimiter=",", dtype=float, encoding="utf-8", skiprows=1)
nrows = data.shape[0]

# ノイズを付加
# TODO: ノイズを付加するのは成長速度だけで良い
noise_Growth = np.random.normal(loc=0, scale=0.3, size=(nrows,))
data[:, -1] += noise_Growth
# 作成したテーブルの保存
np.savetxt(save_path, data, delimiter=",", header=header, fmt="%.1f")
