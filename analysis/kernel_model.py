import numpy as np
from sklearn.model_selection import learning_curve
from sklearn.svm import SVC
from sklearn.svm import SVR

from metrics.calculate_classification_metrics import calculate_classification_metrics
from metrics.calculate_regression_metrics import calculate_regression_metrics
from static.config import Config
from static.new_class import Container
from static.process import grid_search, bayes_search


class SVMRegressionParams:
    @classmethod
    def get_params(cls):
        return {
            'kernel': ['linear', 'rbf'],
            'C': [0.1, 1, 10, 100],
            'gamma': [0.01, 0.1, 1, 10],
            'epsilon': [0.01, 0.1, 1]
        }


# 支持向量机回归
def svm_regression(container: Container):
    x_train = container.x_train
    y_train = container.y_train
    x_test = container.x_test
    y_test = container.y_test
    hyper_params_optimize = container.hyper_params_optimize
    info = {}

    svm_regression_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
    params = SVMRegressionParams.get_params()

    if hyper_params_optimize == "grid_search":
        best_model = grid_search(params, svm_regression_model, x_train, y_train)
    elif hyper_params_optimize == "bayes_search":
        best_model = bayes_search(params, svm_regression_model, x_train, y_train)
    else:
        best_model = svm_regression_model
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


class SVMClassifierParams:
    @classmethod
    def get_params(cls):
        return {
            "C": [0.1, 1, 10, 100],
            "kernel": ['linear', 'rbf', 'poly'],
            "gamma": [0.1, 1, 10]
        }


# 支持向量机分类
def svm_classifier(container: Container):
    x_train = container.x_train
    y_train = container.y_train
    x_test = container.x_test
    y_test = container.y_test
    hyper_params_optimize = container.hyper_params_optimize
    info = {}

    svm_classifier_model = SVC(kernel="rbf")
    params = SVMClassifierParams.get_params()

    if hyper_params_optimize == "grid_search":
        best_model = grid_search(params, svm_classifier_model, x_train, y_train)
    elif hyper_params_optimize == "bayes_search":
        best_model = bayes_search(params, svm_classifier_model, x_train, y_train)
    else:
        best_model = svm_classifier_model
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

    info["指标"] = calculate_classification_metrics(y_pred, y_test)

    container.set_info(info)
    container.set_status("trained")
    container.set_model(best_model)

    return container
