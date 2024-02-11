import numpy as np
from sklearn.metrics import *


def calculate_ar2(real_data, pred_data):
    model_name = "a"
    info = {}

    info["MAE of "+model_name] = mean_absolute_error(real_data, pred_data)
    # mae = mean_absolute_error(real_data, pred_data)
    info["MSE of "+model_name] = mean_squared_error(real_data, pred_data)
    # mse = mean_squared_error(real_data, pred_data)
    info["RSME of "+model_name] = np.sqrt(info["MSE of "+model_name])
    # rsme = np.sqrt(info["MSE of "+model_name])
    info["R-Sqaure of "+model_name] = r2_score(real_data, pred_data)
    # r2 = r2_score(real_data, pred_data)
    if isinstance(max(real_data), np.ndarray):
        info["Adjusted R-Square of " + model_name] = 1 - (1 - info["R-Sqaure of "+model_name]) * (len(pred_data)-1) / (len(pred_data)-max(real_data)[0]-1)
        # ar2 = 1 - (1 - info["R-Sqaure of "+model_name]) * (len(pred_data)-1) / (len(pred_data)-max(real_data)[0]-1)
    else:
        info["Adjusted R-Square of " + model_name] = 1 - (1 - info["R-Sqaure of " + model_name]) * (len(pred_data) - 1) / (len(pred_data) - max(real_data) - 1)
        # ar2 = 1 - (1 - info["R-Sqaure of " + model_name]) * (len(pred_data) - 1) / (len(pred_data) - max(real_data) - 1)

    return info["Adjusted R-Square of " + model_name]


def calculate_regression_metrics(pred_data, real_data, model_name):
    info = {}

    info["MAE of "+model_name] = mean_absolute_error(real_data, pred_data)
    # mae = mean_absolute_error(real_data, pred_data)
    info["MSE of "+model_name] = mean_squared_error(real_data, pred_data)
    # mse = mean_squared_error(real_data, pred_data)
    info["RSME of "+model_name] = np.sqrt(info["MSE of "+model_name])
    # rsme = np.sqrt(info["MSE of "+model_name])
    info["R-Sqaure of "+model_name] = r2_score(real_data, pred_data)
    # r2 = r2_score(real_data, pred_data)
    if isinstance(max(real_data), np.ndarray):
        info["Adjusted R-Square of " + model_name] = 1 - (1 - info["R-Sqaure of "+model_name]) * (len(pred_data)-1) / (len(pred_data)-max(real_data)[0]-1)
        # ar2 = 1 - (1 - info["R-Sqaure of "+model_name]) * (len(pred_data)-1) / (len(pred_data)-max(real_data)[0]-1)
    else:
        info["Adjusted R-Square of " + model_name] = 1 - (1 - info["R-Sqaure of " + model_name]) * (len(pred_data) - 1) / (len(pred_data) - max(real_data) - 1)
        # ar2 = 1 - (1 - info["R-Sqaure of " + model_name]) * (len(pred_data) - 1) / (len(pred_data) - max(real_data) - 1)

    return info


