import numpy as np
import gradio as gr
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from functions.process import get_values_from_container_class, transform_params_list
from metrics.calculate_classification_metrics import calculate_classification_metrics
from metrics.calculate_regression_metrics import calculate_regression_metrics
from analysis.others.hyperparam_optimize import *
from classes.static_custom_class import StaticValue


class LinearRegressionParams:
    @classmethod
    def get_params_type(cls, sort):
        if sort in ["Lasso", "Ridge", "ElasticNet"]:
            return {
                "fit_intercept": StaticValue.BOOL,
                "alpha": StaticValue.FLOAT,
            }
        else:
            return {
                "fit_intercept": StaticValue.BOOL
            }

    @classmethod
    def get_params(cls, sort):
        if sort in ["Lasso", "Ridge", "ElasticNet"]:
            return {
                "fit_intercept": [True, False],
                "alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
            }
        else:
            return {
                "fit_intercept": [True, False]
            }


# 线性回归
def linear_regressor(container, params, model=None):
    x_train, y_train, x_test, y_test, hyper_params_optimize = get_values_from_container_class(container)
    info = {}

    params = transform_params_list(LinearRegressionParams, params, model)
    params['random_state'] = [StaticValue.RANDOM_STATE]

    if model == "Lasso":
        linear_regression_model = Lasso(alpha=0.1, random_state=StaticValue.RANDOM_STATE)
        params = params
    elif model == "Ridge":
        linear_regression_model = Ridge(alpha=0.1, random_state=StaticValue.RANDOM_STATE)
        params = params
    elif model == "ElasticNet":
        linear_regression_model = ElasticNet(alpha=0.1, random_state=StaticValue.RANDOM_STATE)
        params = params
    elif model == "LinearRegression":
        linear_regression_model = LinearRegression()
        params = params
    else:
        linear_regression_model = LinearRegression()
        params = params

    try:
        if hyper_params_optimize == "grid_search":
            best_model = grid_search(params, linear_regression_model, x_train, y_train)
        elif hyper_params_optimize == "bayes_search":
            best_model = bayes_search(params, linear_regression_model, x_train, y_train)
        else:
            best_model = linear_regression_model
            best_model.fit(x_train, y_train)
    except Exception:
        gr.Warning("超参数设置有误，将按照默认模型训练")
        best_model = LinearRegression()
        best_model.fit(x_train, y_train)

    info["参数"] = best_model.get_params()

    # lr_intercept = best_model.intercept_
    # info["Intercept of linear regression equation"] = lr_intercept
    #
    # lr_coef = best_model.coef_
    # info["Coefficients of linear regression equation"] = lr_coef

    y_pred = best_model.predict(x_test)
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


class PolynomialRegressionParams:
    @classmethod
    def get_params_type(cls):
        return {
            "polynomial_features__degree": StaticValue.INT,
            "linear_regression_model__fit_intercept": StaticValue.BOOL
        }

    @classmethod
    def get_params(cls):
        return {
            "polynomial_features__degree": [2, 3],
            "linear_regression_model__fit_intercept": [True, False]
        }


# 多项式回归
def polynomial_regressor(container, params):
    x_train, y_train, x_test, y_test, hyper_params_optimize = get_values_from_container_class(container)
    info = {}

    params = transform_params_list(PolynomialRegressionParams, params)

    polynomial_features = PolynomialFeatures(degree=2)
    linear_regression_model = LinearRegression()

    polynomial_regression_model = Pipeline([("polynomial_features", polynomial_features),
                                            ("linear_regression_model", linear_regression_model)])

    if hyper_params_optimize == "grid_search":
        best_model = grid_search(params, polynomial_regression_model, x_train, y_train)
    elif hyper_params_optimize == "bayes_search":
        best_model = bayes_search(params, polynomial_regression_model, x_train, y_train)
    else:
        best_model = polynomial_regression_model
        best_model.fit(x_train, y_train)

    info["参数"] = best_model.get_params()

    # feature_names = best_model["polynomial_features"].get_feature_names_out()
    # info["Feature names of polynomial regression"] = feature_names
    #
    # lr_intercept = best_model["linear_regression_model"].intercept_
    # info["Intercept of polynomial regression equation"] = lr_intercept
    #
    # lr_coef = best_model["linear_regression_model"].coef_
    # info["Coefficients of polynomial regression equation"] = lr_coef

    x_test_ = best_model["polynomial_features"].fit_transform(x_test)
    y_pred = best_model["linear_regression_model"].predict(x_test_)
    container.set_y_pred(y_pred)

    train_sizes, train_scores, test_scores = learning_curve(best_model, x_train, y_train, cv=5)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    container.set_learning_curve_values(train_sizes, train_scores_mean, train_scores_std, test_scores_mean, test_scores_std)

    info["指标"] = calculate_regression_metrics(y_pred, y_test)

    container.set_info(info)
    container.set_status("trained")
    container.set_model(best_model)

    return container


class LogisticRegressionParams:
    @classmethod
    def get_params_type(cls):
        return {
            "C": StaticValue.FLOAT,
            "max_iter": StaticValue.INT,
            "solver": StaticValue.STR,
        }

    @classmethod
    def get_params(cls):
        return {
            "C": [0.001, 0.01, 0.1, 1.0, 10.0],
            "max_iter": [100, 200, 300],
            "solver": ["liblinear", "lbfgs", "newton-cg", "sag", "saga"],
        }


# 逻辑斯谛分类
def logistic_classifier(container, params):
    x_train, y_train, x_test, y_test, hyper_params_optimize = get_values_from_container_class(container)
    info = {}

    params = transform_params_list(LogisticRegressionParams, params)
    params['random_state'] = [StaticValue.RANDOM_STATE]

    logistic_regression_model = LogisticRegression(random_state=StaticValue.RANDOM_STATE)

    if hyper_params_optimize == "grid_search":
        best_model = grid_search(params, logistic_regression_model, x_train, y_train)
    elif hyper_params_optimize == "bayes_search":
        best_model = bayes_search(params, logistic_regression_model, x_train, y_train)
    else:
        best_model = logistic_regression_model
        best_model.fit(x_train, y_train)

    info["参数"] = best_model.get_params()

    # lr_intercept = best_model.intercept_
    # info["Intercept of logistic regression equation"] = lr_intercept.tolist()
    #
    # lr_coef = best_model.coef_
    # info["Coefficients of logistic regression equation"] = lr_coef.tolist()

    y_pred = best_model.predict(x_test)
    container.set_y_pred(y_pred)

    train_sizes, train_scores, test_scores = learning_curve(best_model, x_train, y_train, cv=5)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    container.set_learning_curve_values(train_sizes, train_scores_mean, train_scores_std, test_scores_mean,
                                        test_scores_std)

    info["指标"] = calculate_classification_metrics(y_pred, y_test)

    container.set_info(info)
    container.set_status("trained")
    container.set_model(best_model)

    return container
