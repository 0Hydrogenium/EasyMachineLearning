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
from visualization.draw_learning_curve import draw_learning_curve
from metrics.calculate_classification_metrics import calculate_classification_metrics
from metrics.calculate_regression_metrics import calculate_regression_metrics
from interface.main import Container


# 线性回归
def linear_regression(container: Container, model=None):
    x_train = container.x_train
    y_train = container.y_train
    x_test = container.x_test
    y_test = container.y_test
    hyper_params_optimize = container.hyper_params_optimize
    info = {}

    if model == "Lasso":
        linear_regression_model = Lasso(alpha=0.1)
        params = {
            "fit_intercept": [True, False],
            "alpha": [0.001, 0.01, 0.1, 1.0, 10.0]
        }
    elif model == "Ridge":
        linear_regression_model = Ridge(alpha=0.1)
        params = {
            "fit_intercept": [True, False],
            "alpha": [0.001, 0.01, 0.1, 1.0, 10.0]
        }
    elif model == "ElasticNet":
        linear_regression_model = ElasticNet(alpha=0.1)
        params = {
            "fit_intercept": [True, False],
            "alpha": [0.001, 0.01, 0.1, 1.0, 10.0]
        }
    else:
        linear_regression_model = LinearRegression()
        params = {
            "fit_intercept": [True, False]
        }

    if hyper_params_optimize == "grid_search":
        best_model = grid_search(params, linear_regression_model, x_train, y_train)
    elif hyper_params_optimize == "bayes_search":
        best_model = bayes_search(params, linear_regression_model, x_train, y_train)
    else:
        best_model = linear_regression_model
        best_model.fit(x_train, y_train)

    lr_intercept = best_model.intercept_
    info["Intercept of linear regression equation"] = lr_intercept

    lr_coef = best_model.coef_
    info["Coefficients of linear regression equation"] = lr_coef

    y_pred = best_model.predict(x_test)
    container.set_y_pred(y_pred)

    train_sizes, train_scores, test_scores = learning_curve(best_model, x_train, y_train, cv=5)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    container.set_learning_curve_values(train_sizes, train_scores_mean, train_scores_std, test_scores_mean, test_scores_std)

    info.update(calculate_regression_metrics(y_pred, y_test, "linear regression"))

    container.set_info(info)
    container.set_status("trained")
    container.set_model(best_model)

    return container


def polynomial_regression(x_train_and_validate, y_train_and_validate, x_test, y_test, train_and_validate_data_list=None, hyper_params_optimize=None):
    info = {}

    polynomial_features = PolynomialFeatures(degree=2)
    linear_regression_model = LinearRegression()

    polynomial_regression_model = Pipeline([("polynomial_features", polynomial_features),
                                            ("linear_regression_model", linear_regression_model)])
    params = {
        "polynomial_features__degree": [2, 3],
        "linear_regression_model__fit_intercept": [True, False]
    }

    if hyper_params_optimize == "grid_search":
        best_model = grid_search(params, polynomial_regression_model, x_train_and_validate, y_train_and_validate, "neg_mean_squared_error")
    elif hyper_params_optimize == "bayes_search":
        best_model = bayes_search(params, polynomial_regression_model, x_train_and_validate, y_train_and_validate, "neg_mean_squared_error")
    else:
        # TODO
        best_model = linear_regression_model
        for epoch in train_and_validate_data_list:
            # TODO
            x_train, x_validate, y_train, y_validate = epoch

            best_model.fit(x_train, y_train)

    feature_names = best_model["polynomial_features"].get_feature_names_out()

    x_test_ = best_model["polynomial_features"].fit_transform(x_test)

    lr_intercept = best_model["linear_regression_model"].intercept_
    info["Intercept of polynomial regression equation"] = lr_intercept

    lr_coef = best_model["linear_regression_model"].coef_
    info["Coefficients of polynomial regression equation"] = lr_coef

    y_pred = best_model["linear_regression_model"].predict(x_test_)

    # draw_scatter_line_graph(x_test_, y_pred, y_test, lr_coef, lr_intercept, ["pred", "real"], "polynomial regression model residual plot")

    info.update(calculate_regression_metrics(y_pred, y_test, "polynomial regression"))
    info.update(calculate_classification_metrics(y_pred, y_test, "polynomial regression"))

    return info


# Logistic regression
def logistic_regression(x, y, x_train_and_validate, y_train_and_validate, x_test, y_test, train_and_validate_data_list=None, hyper_params_optimize=None):
    info = {}

    logistic_regression_model = LogisticRegression()
    logistic_regression_model = LogisticRegression()
    params = {
        "C": [0.001, 0.01, 0.1, 1.0, 10.0],
        "max_iter": [100, 200, 300],
        "solver": ["liblinear", "lbfgs", "newton-cg", "sag", "saga"]
    }

    if hyper_params_optimize == "grid_search":
        best_model = grid_search(params, logistic_regression_model, x_train_and_validate, y_train_and_validate)
    elif hyper_params_optimize == "bayes_search":
        best_model = bayes_search(params, logistic_regression_model, x_train_and_validate, y_train_and_validate)
    else:
        # TODO
        best_model = logistic_regression_model
        for epoch in train_and_validate_data_list:
            # TODO
            x_train, x_validate, y_train, y_validate = epoch

            best_model.fit(x_train, y_train)

    info["logistic regression Params"] = best_model.get_params()

    lr_intercept = best_model.intercept_
    info["Intercept of logistic regression equation"] = lr_intercept.tolist()

    lr_coef = best_model.coef_
    info["Coefficients of logistic regression equation"] = lr_coef.tolist()

    y_pred = best_model.predict(x_test).reshape(-1, 1)

    # 0202:

    train_sizes, train_scores, test_scores = learning_curve(best_model, x, y, cv=5, scoring="accuracy")

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    draw_learning_curve(train_sizes, train_scores_mean, train_scores_std, test_scores_mean, test_scores_std)

    # draw_scatter_line_graph(x_test, y_pred, y_test, lr_coef, lr_intercept, ["pred", "real"], "logistic regression model residual plot")

    # info.update(calculate_regression_metrics(y_pred, y_test, "logistic regression"))
    # info.update(calculate_classification_metrics(y_pred, y_test, "logistic regression"))
    f1_score, fpr, tpr, thresholds = calculate_classification_metrics(y_pred, y_test, "logistic regression")


    return info, train_sizes, train_scores_mean, train_scores_std, test_scores_mean, test_scores_std, f1_score, fpr, tpr, thresholds


def momentum_logistic_regression(x, y, x_train_and_validate, y_train_and_validate, x_test, y_test, train_and_validate_data_list=None, hyper_params_optimize=None):
    info = {}

    logistic_regression_model = LogisticRegression(multi_class='ovr', solver='liblinear')
    params = {
        "C": [0.001, 0.01, 0.1, 1.0, 10.0],
        "max_iter": [100, 200, 300],
        "solver": ["liblinear", "lbfgs", "newton-cg", "sag", "saga"]
    }

    # if hyper_params_optimize == "grid_search":
    #     best_model = grid_search(params, logistic_regression_model, x_train_and_validate, y_train_and_validate)
    # elif hyper_params_optimize == "bayes_search":
    #     best_model = bayes_search(params, logistic_regression_model, x_train_and_validate, y_train_and_validate)
    # else:
    #     # TODO
    #     best_model = logistic_regression_model
    #     for epoch in train_and_validate_data_list:
    #         # TODO
    #         x_train, x_validate, y_train, y_validate = epoch
    #
    #         best_model.fit(x_train, y_train)
    polynomial_features = PolynomialFeatures(degree=2)
    linear_regression_model = LinearRegression()

    polynomial_regression_model = Pipeline([("polynomial_features", polynomial_features),
                                            ("linear_regression_model", linear_regression_model)])

    best_model = logistic_regression_model

    best_model = ElasticNet(alpha=0.1)

    best_model.fit(x[:, :], y[:, :])

    info["logistic regression Params"] = best_model.get_params()

    lr_intercept = best_model.intercept_
    info["Intercept of logistic regression equation"] = lr_intercept.tolist()

    lr_coef = best_model.coef_
    info["Coefficients of logistic regression equation"] = lr_coef.tolist()

    y_pred = best_model.predict(x).reshape(-1, 1)

    # 0202:

    train_sizes, train_scores, test_scores = learning_curve(best_model, x, y, cv=5, scoring="accuracy")

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # draw_learning_curve(train_sizes, train_scores_mean, train_scores_std, test_scores_mean, test_scores_std)

    # draw_scatter_line_graph(x_test, y_pred, y_test, lr_coef, lr_intercept, ["pred", "real"], "logistic regression model residual plot")

    # info.update(calculate_regression_metrics(y_pred, y_test, "logistic regression"))
    # info.update(calculate_classification_metrics(y_pred, y_test, "logistic regression"))
    # f1_score, fpr, tpr, thresholds = calculate_classification_metrics(y_pred, y_test, "logistic regression")
    f1_score, fpr, tpr, thresholds = 0,0,0,0


    return y_pred, info, train_sizes, train_scores_mean, train_scores_std, test_scores_mean, test_scores_std, f1_score, fpr, tpr, thresholds


def momentum_polynomial_regression(x, y, x_train_and_validate, y_train_and_validate, x_test, y_test, train_and_validate_data_list=None, hyper_params_optimize=None):
    info = {}

    polynomial_features = PolynomialFeatures(degree=2)
    linear_regression_model = LinearRegression()

    polynomial_regression_model = Pipeline([("polynomial_features", polynomial_features),
                                            ("linear_regression_model", linear_regression_model)])
    params = {
        "polynomial_features__degree": [2],
        "linear_regression_model__fit_intercept": [True, False]
    }

    if hyper_params_optimize == "grid_search":
        best_model = grid_search(params, polynomial_regression_model, x, y, "neg_mean_squared_error")
    elif hyper_params_optimize == "bayes_search":
        best_model = bayes_search(params, polynomial_regression_model, x, y, "neg_mean_squared_error")
    else:
        # TODO
        best_model = linear_regression_model
        for epoch in train_and_validate_data_list:
            # TODO
            x_train, x_validate, y_train, y_validate = epoch

            best_model.fit(x_train, y_train)

    feature_names = best_model["polynomial_features"].get_feature_names_out()

    x_test_ = best_model["polynomial_features"].fit_transform(x)

    lr_intercept = best_model["linear_regression_model"].intercept_
    info["Intercept of polynomial regression equation"] = lr_intercept

    lr_coef = best_model["linear_regression_model"].coef_
    info["Coefficients of polynomial regression equation"] = lr_coef

    y_pred = best_model["linear_regression_model"].predict(x_test_)

    # draw_scatter_line_graph(x_test_, y_pred, y_test, lr_coef, lr_intercept, ["pred", "real"], "polynomial regression model residual plot")

    # info.update(calculate_regression_metrics(y_pred, y, "polynomial regression"))
    # info.update(calculate_classification_metrics(y_pred, y, "polynomial regression"))

    return y_pred, info