#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8
import GPy
import GPyOpt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

#-----loading data-----
df = pd.read_excel("../data/Concrete_Data.xls")
x = df.iloc[:, :8]; x = x.values # 入力
y = df.iloc[:, [8]]; y = y.values # 出力(成長速度)
dim = x.shape[1] # 次元数
#-----normalization(using standard scaler)------
x_stdsc = StandardScaler()
y_stdsc = StandardScaler()
x_std = x_stdsc.fit_transform(x)
y_std = y_stdsc.fit_transform(y)

#-----setting of kernel-----
kernel = GPy.kern.RBF(dim) + GPy.kern.Bias(dim) + GPy.kern.Linear(dim)
# kernel = GPy.kern.RBF(1) + GPy.kern.Bias(1) + GPy.kern.Linear(1)

#-----fitting-----
model = GPy.models.GPRegression(x_std, y_std, kernel=kernel)
model.optimize() # max_iters=1e5
# model.plot()

#-----prediction-----
y_true_std = y_std[:]
y_true = y_stdsc.inverse_transform(y_true_std)
y_pred_std = model.predict(x_std)
y_pred = y_stdsc.inverse_transform(y_pred_std)

#-----parity-plotの出力-----   
# 成長速度
plt.plot(y_true, y_pred[0], 'o')
plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='r')
plt.xlabel("Actual Growth")
plt.ylabel("Predicted Growth")
# show plots
plt.tight_layout()
plt.savefig('../result/parity-plot_gr.png', dpi=100)
plt.show()
# calculate R2
R2_gr = r2_score(y_true, y_pred[0])
print('R2 score : {:.3f}'.format(R2_gr))


# In[2]:


#-----setting of input-----
bounds = [
# If continuous value, set "type" to "continuous" and set the value as (lower limit, upper limit).
{'name': 'x0', 'type': 'continuous', 'domain': (102, 540)},
{'name': 'x1', 'type': 'continuous', 'domain': (0, 359.4)},
{'name': 'x2', 'type': 'continuous', 'domain': (0, 200.1)},
{'name': 'x3', 'type': 'continuous', 'domain': (121.75, 247)},
{'name': 'x4', 'type': 'continuous', 'domain': (0, 32.2)},
{'name': 'x5', 'type': 'continuous', 'domain': (801, 1145)},
{'name': 'x6', 'type': 'continuous', 'domain': (1, 992.6)},
{'name': 'x7', 'type': 'discrete', 'domain': (365,)} # x7 must be 365(after 1year)
]

# setting of objective function
def f(x):
    global model, x_stdsc, y_stdsc
    x_std = x_stdsc.transform(x)
    y_pred_std = model.predict(x_std)[0]
    y_pred = y_stdsc.inverse_transform(y_pred_std)
    return y_pred

flag_maximize = True # maxmize:True, minizize:False
num = 30 # iteration
BO = GPyOpt.methods.BayesianOptimization(
    f=f,
    domain=bounds,
    #constraints = constraints, # in case constrains needed
    maximize = flag_maximize, # maximize or minimize
    model_type = "GP" # default "GP"
)

#-----run optimization-----
BO.run_optimization(max_iter=num)
# extract best output(max or min)
if flag_maximize:
    opt_fx = BO.fx_opt * (-1)
else:
    opt_fx = BO.fx_opt
# extract best input
opt_x = BO.x_opt
print("Best value is {}".format(opt_fx))
print("Best input is {}".format(opt_x))

#-----save results to a csv file-----
columns_name_list = ["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7"] # column name
columns_name_list_str = ",".join(columns_name_list) # put a comma between elements to a string
columns_name_list_str = columns_name_list_str + ",Y" + "\n" # insert the target name and line feed code
with open("../result/opt_data.csv", "w") as f:
    f.write(columns_name_list_str)
    buf_list = []
    for X in opt_x.tolist():
        buf_list.append(X)
    buf_list.append(opt_fx)
    opt_list = list(map(str, buf_list))
    opt_str = ','.join(opt_list)
    f.write(opt_str + "\n")

#-----save history to a csv file-----
history_X = BO.X[0:num]
if flag_maximize:
    history_Y = BO.Y[0:num] * (-1)
else:
    history_Y = BO.Y[0:num]
history_Y_list = history_Y.tolist()
history_X_list = history_X.tolist()
with open("../result/history_data.csv", "w") as f:
    f.write(columns_name_list_str)
    for X, Y in zip(history_X_list, history_Y_list):
        X.extend(Y)
        buf_list = list(map(str, X))
        buf_str = ','.join(buf_list)
        f.write(buf_str + '\n')

