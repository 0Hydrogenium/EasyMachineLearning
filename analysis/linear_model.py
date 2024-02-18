import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import learning_curve

from static.process import grid_search, bayes_search
from metrics.calculate_classification_metrics import calculate_classification_metrics
from metrics.calculate_regression_metrics import calculate_regression_metrics
from static.new_class import *
from static.config import Config


class LinearRegressionParams:
    @classmethod
    def get_params(cls, sort):
        if sort in ["Lasso", "Ridge", "ElasticNet"]:
            return {
                "fit_intercept": [True, False],
                "alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
                "random_state": [Config.RANDOM_STATE]
            }
        else:
            return {
                "fit_intercept": [True, False]
            }


# 线性回归
def linear_regression(container: Container, model=None):
    x_train = container.x_train
    y_train = container.y_train
    x_test = container.x_test
    y_test = container.y_test
    hyper_params_optimize = container.hyper_params_optimize
    info = {}

    if model == "Lasso":
        linear_regression_model = Lasso(alpha=0.1, random_state=Config.RANDOM_STATE)
        params = LinearRegressionParams.get_params(model)
    elif model == "Ridge":
        linear_regression_model = Ridge(alpha=0.1, random_state=Config.RANDOM_STATE)
        params = LinearRegressionParams.get_params(model)
    elif model == "ElasticNet":
        linear_regression_model = ElasticNet(alpha=0.1, random_state=Config.RANDOM_STATE)
        params = LinearRegressionParams.get_params(model)
    else:
        linear_regression_model = LinearRegression()
        params = LinearRegressionParams.get_params(model)

    if hyper_params_optimize == "grid_search":
        best_model = grid_search(params, linear_regression_model, x_train, y_train)
    elif hyper_params_optimize == "bayes_search":
        best_model = bayes_search(params, linear_regression_model, x_train, y_train)
    else:
        best_model = linear_regression_model
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

    info["参数"] = calculate_regression_metrics(y_pred, y_test)

    container.set_info(info)
    container.set_status("trained")
    container.set_model(best_model)

    return container


class PolynomialRegressionParams:
    @classmethod
    def get_params(cls):
        return {
            "polynomial_features__degree": [2, 3],
            "linear_regression_model__fit_intercept": [True, False]
        }


# 多项式回归
def polynomial_regression(container: Container):
    x_train = container.x_train
    y_train = container.y_train
    x_test = container.x_test
    y_test = container.y_test
    hyper_params_optimize = container.hyper_params_optimize
    info = {}

    polynomial_features = PolynomialFeatures(degree=2)
    linear_regression_model = LinearRegression()

    polynomial_regression_model = Pipeline([("polynomial_features", polynomial_features),
                                            ("linear_regression_model", linear_regression_model)])
    params = PolynomialRegressionParams.get_params()

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
    def get_params(cls):
        return {
            "C": [0.001, 0.01, 0.1, 1.0, 10.0],
            "max_iter": [100, 200, 300],
            "solver": ["liblinear", "lbfgs", "newton-cg", "sag", "saga"],
            "random_state": [Config.RANDOM_STATE]
        }


# 逻辑斯谛分类
def logistic_regression(container: Container):
    x_train = container.x_train
    y_train = container.y_train
    x_test = container.x_test
    y_test = container.y_test
    hyper_params_optimize = container.hyper_params_optimize
    info = {}

    logistic_regression_model = LogisticRegression(random_state=Config.RANDOM_STATE)
    params = LogisticRegressionParams.get_params()

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
