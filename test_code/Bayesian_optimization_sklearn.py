#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
from skopt import gp_minimize
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel as Wh
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

#-----loading data-----
df = pd.read_excel("../data/Concrete_Data.xls")
x = df.iloc[:, :8]; x = x.values # 入力
y = df.iloc[:, [8]]; y = y.values # 出力(成長速度)
dim = x.shape[1] # 次元数
#-----normalization(using standard scaler)------
x_sc = StandardScaler()
y_sc = StandardScaler()
x_std = x_sc.fit_transform(x)
y_std = y_sc.fit_transform(y)

#-----regression-----
kernel = C(1.0, (1e-2,1e2)) * RBF(1.0, (1e-2,1e2)) + Wh(0.01, (1e-2,1e2))
reg = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=30)
reg.fit(x_std, y_std)

#-----prediction-----
y_true_std = y_std[:]
y_true = y_sc.inverse_transform(y_true_std)
y_pred_std = reg.predict(x_std)
y_pred = y_sc.inverse_transform(y_pred_std)

#-----parity-plotの出力-----   
# 成長速度
plt.plot(y_true, y_pred, 'o')
plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='r')
plt.xlabel("Actual Growth")
plt.ylabel("Predicted Growth")
# show plots
plt.tight_layout()
plt.savefig('../result/parity-plot_gr.png', dpi=100)
plt.show()
# calculate R2
R2_gr = r2_score(y_true, y_pred)
print('R2 score : {:.3f}'.format(R2_gr))


# In[2]:


#-----setting of Bayesian Optimization----- 
spaces = [
    (102.0, 540.0, 'uniform'), # x0,
    (0.0, 359.0, 'uniform'), # x1,
    (0.0, 200.0, 'uniform'), # x2,
    (121.75, 247.0, 'uniform'), # x3,
    (0.0, 32.0, 'uniform'), # x4,
    (801.0, 1145.0, 'uniform'), # x5,
    (1.0, 992.0, 'uniform'), # x6,
    [365.0] # x7
]

flag_maximize = True # maxmize:True, minizize:False
n_calls = 50 # iteration

# setting of objective function
def f(x):
    global reg, x_sc, y_sc
    x = np.array(x).reshape(1, -1)
    x_std = x_sc.transform(x)
    y_std = reg.predict(x_std); y = y_sc.inverse_transform(y_std)[0][0]
    if flag_maximize:
        return -y
    else:
        return y

#-----run optimization-----
res = gp_minimize(
    f, spaces,
    acq_func="EI",
    n_points=10000,
    n_calls=n_calls,
    model_queue_size=1,
    n_jobs=2,
    verbose=False)

# extract best output(max or min)
if flag_maximize:
    opt_fx = res.fun * (-1)
else:
    opt_fx = res.fun
# extract best input
opt_x = res.x
print("Best value is {}".format(opt_fx))
print("Best input is {}".format(opt_x))


# In[3]:


#-----save results to a csv file-----
columns_name_list = ["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7"] # column name
columns_name_list_str = ",".join(columns_name_list) # put a comma between elements to a string
columns_name_list_str = columns_name_list_str + ",Y" + "\n" # insert the target name and line feed code
with open("../result/opt_data.csv", "w") as f:
    f.write(columns_name_list_str)
    buf_list = []
    for X in res.x:
        buf_list.append(X)
    buf_list.append(res.fun)
    opt_list = list(map(str, buf_list))
    opt_str = ','.join(opt_list)
    f.write(opt_str + "\n")

#-----save history to a csv file-----
history_X = res.x_iters[0:n_calls]
if flag_maximize:
    history_Y = res.func_vals[0:n_calls] * (-1)
else:
    history_Y = res.func_vals[0:n_calls]
history_Y_list = history_Y.tolist()
history_X_list = history_X
with open("../result/history_data.csv", "w") as f:
    f.write(columns_name_list_str)
    for X, Y in zip(history_X_list, history_Y_list):
        X.append(Y)
        buf_list = list(map(str, X))
        buf_str = ','.join(buf_list)
        f.write(buf_str + '\n')

