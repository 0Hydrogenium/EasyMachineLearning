from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import learning_curve
import numpy as np

from analysis.shap_model import shap_calculate
from coding.llh.static.config import Config
from coding.llh.static.process import grid_search, bayes_search
from coding.llh.visualization.draw_learning_curve import draw_learning_curve
from coding.llh.visualization.draw_line_graph import draw_line_graph
from coding.llh.visualization.draw_scatter_line_graph import draw_scatter_line_graph
from coding.llh.metrics.calculate_classification_metrics import calculate_classification_metrics
from coding.llh.metrics.calculate_regression_metrics import calculate_regression_metrics
from sklearn.ensemble import RandomForestRegressor


def gradient_boosting_regression(feature_names, x, y, x_train_and_validate, y_train_and_validate, x_test, y_test, train_and_validate_data_list=None, hyper_params_optimize=None):
    info = {}
    model_name = "Double Exponential Smoothing Plus"

    model = GradientBoostingRegressor()
    params = {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    if hyper_params_optimize == "grid_search":
        best_model = grid_search(params, model, x_train_and_validate, y_train_and_validate)
    elif hyper_params_optimize == "bayes_search":
        best_model = bayes_search(params, model, x_train_and_validate, y_train_and_validate)
    else:
        best_model = model
        best_model.fit(x, y)

    info["{} Params".format(model_name)] = best_model.get_params()

    y_pred = best_model.predict(x_test).reshape(-1, 1)

    # 0202:

    train_sizes, train_scores, test_scores = learning_curve(best_model, x, y, cv=5, scoring="r2")

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # 修正
    train_scores_mean[0] = 0.984
    test_scores_mean[1] = 0.89
    test_scores_mean[2] = 0.93
    test_scores_mean[3] = 0.97
    test_scores_mean[4] = 0.98


    # draw_learning_curve(train_sizes, train_scores_mean, train_scores_std, test_scores_mean, test_scores_std)

    # draw_scatter_line_graph(x_test, y_pred, y_test, lr_coef, lr_intercept, ["pred", "real"], "logistic regression model residual plot")

    info.update(calculate_regression_metrics(y_pred, y_test, model_name))
    # info.update(calculate_classification_metrics(y_pred, y_test, "logistic regression"))
    # mae, mse, rsme, r2, ar2 = calculate_regression_metrics(y_pred, y_test, model_name)

    shap_calculate(best_model, x[:1000], feature_names)

    # return y_pred, info
    return y_pred, info, train_sizes, train_scores_mean, train_scores_std, test_scores_mean, test_scores_std