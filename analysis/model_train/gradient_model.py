import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import learning_curve

from functions.process import transform_params_list, get_values_from_container_class
from metrics.calculate_regression_metrics import calculate_regression_metrics
from analysis.others.hyperparam_optimize import *
from classes.static_custom_class import StaticValue


class GradientBoostingParams:
    @classmethod
    def get_params_type(cls):
        return {
            'n_estimators': StaticValue.INT,
            'learning_rate': StaticValue.FLOAT,
            'max_depth': StaticValue.INT,
            'min_samples_split': StaticValue.INT,
            'min_samples_leaf': StaticValue.INT,
            'random_state': StaticValue.INT
        }

    @classmethod
    def get_params(cls):
        return {
            'n_estimators': [50, 100, 150],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'random_state': [StaticValue.RANDOM_STATE]
        }


# 梯度提升回归
def gradient_boosting_regressor(container, params_list):
    x_train, y_train, x_test, y_test, hyper_params_optimize = get_values_from_container_class(container)
    info = {}

    params_list = transform_params_list(GradientBoostingParams, params_list)

    gradient_boosting_regression_model = GradientBoostingRegressor(random_state=StaticValue.RANDOM_STATE)
    params = params_list

    if hyper_params_optimize == "grid_search":
        best_model = grid_search(params, gradient_boosting_regression_model, x_train, y_train)
    elif hyper_params_optimize == "bayes_search":
        best_model = bayes_search(params, gradient_boosting_regression_model, x_train, y_train)
    else:
        best_model = gradient_boosting_regression_model
        best_model.fit(x_train, y_train)

    info["参数"] = best_model.get_params()

    y_pred = best_model.predict(x_test)
    # y_pred = best_model.predict(x_test).reshape(-1, 1)
    container.set_y_pred(y_pred)

    train_sizes, train_scores, test_scores = learning_curve(best_model, x_train, y_train, cv=5)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    container.set_learning_curve_values(train_sizes, train_scores_mean, train_scores_std, test_scores_mean,
                                        test_scores_std)

    info["指标"] = calculate_regression_metrics(y_pred, y_test)

    container.set_info(info)
    container.set_status("trained")
    container.set_model(best_model)

    return container
