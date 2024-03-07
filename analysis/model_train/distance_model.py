from sklearn.model_selection import learning_curve
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from analysis.others.shap_model import *
from classes.static_custom_class import StaticValue
from functions.process import get_values_from_container_class, transform_params_list
from metrics.calculate_classification_metrics import calculate_classification_metrics
from metrics.calculate_regression_metrics import calculate_regression_metrics
from analysis.others.hyperparam_optimize import *


class KNNClassifierParams:
    @classmethod
    def get_params_type(cls):
        return {
            "n_neighbors": StaticValue.INT,
            "weights": StaticValue.STR,
            "p": StaticValue.INT
        }

    @classmethod
    def get_params(cls):
        return {
            "n_neighbors": [3, 5, 7, 9],
            "weights": ['uniform', 'distance'],
            "p": [1, 2]
        }


# KNN分类
def knn_classifier(container, params_list):
    x_train, y_train, x_test, y_test, hyper_params_optimize = get_values_from_container_class(container)
    info = {}

    params_list = transform_params_list(KNNClassifierParams, params_list)

    knn_classifier_model = KNeighborsClassifier()
    params = params_list

    if hyper_params_optimize == "grid_search":
        best_model = grid_search(params, knn_classifier_model, x_train, y_train)
    elif hyper_params_optimize == "bayes_search":
        best_model = bayes_search(params, knn_classifier_model, x_train, y_train)
    else:
        best_model = knn_classifier_model
        best_model.fit(x_train, y_train)

    info["参数"] = best_model.get_params()

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


class KNNRegressionParams:
    @classmethod
    def get_params_type(cls):
        return {
            "n_neighbors": StaticValue.INT,
            "weights": StaticValue.STR,
            "p": StaticValue.INT
        }

    @classmethod
    def get_params(cls):
        return {
            "n_neighbors": [3, 5, 7, 9],
            "weights": ['uniform', 'distance'],
            "p": [1, 2]
        }


# KNN回归
def knn_regressor(container, params_list):
    x_train, y_train, x_test, y_test, hyper_params_optimize = get_values_from_container_class(container)
    info = {}

    params_list = transform_params_list(KNNRegressionParams, params_list)

    knn_regression_model = KNeighborsRegressor()
    params = params_list

    if hyper_params_optimize == "grid_search":
        best_model = grid_search(params, knn_regression_model, x_train, y_train)
    elif hyper_params_optimize == "bayes_search":
        best_model = bayes_search(params, knn_regression_model, x_train, y_train)
    else:
        best_model = knn_regression_model
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