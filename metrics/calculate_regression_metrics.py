import numpy as np
from sklearn.metrics import *


class RegressionMetrics:
    @classmethod
    def get_metrics(cls):
        return ["MAE", "MSE", "RSME", "R-Sqaure", "Adjusted R-Square"]


def calculate_regression_metrics(pred_data, real_data):
    info = {}

    info["MAE"] = mean_absolute_error(real_data, pred_data)
    # mae = mean_absolute_error(real_data, pred_data)
    info["MSE"] = mean_squared_error(real_data, pred_data)
    # mse = mean_squared_error(real_data, pred_data)
    info["RSME"] = np.sqrt(info["MSE"])
    # rsme = np.sqrt(info["MSE of "+model_name])
    info["R-Sqaure"] = r2_score(real_data, pred_data)
    # r2 = r2_score(real_data, pred_data)
    if isinstance(max(real_data), np.ndarray):
        info["Adjusted R-Square"] = 1 - (1 - info["R-Sqaure"]) * (len(pred_data)-1) / (len(pred_data)-max(real_data)[0]-1)
        # ar2 = 1 - (1 - info["R-Sqaure of "+model_name]) * (len(pred_data)-1) / (len(pred_data)-max(real_data)[0]-1)
    else:
        info["Adjusted R-Square"] = 1 - (1 - info["R-Sqaure"]) * (len(pred_data) - 1) / (len(pred_data) - max(real_data) - 1)
        # ar2 = 1 - (1 - info["R-Sqaure of " + model_name]) * (len(pred_data) - 1) / (len(pred_data) - max(real_data) - 1)

    return info


