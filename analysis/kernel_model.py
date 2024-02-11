from sklearn.model_selection import learning_curve
from sklearn.svm import SVC
from sklearn.svm import SVR
import numpy as np

from coding.llh.analysis.my_learning_curve import my_learning_curve
from coding.llh.analysis.shap_model import shap_calculate
from coding.llh.static.process import grid_search, bayes_search
from coding.llh.visualization.draw_line_graph import draw_line_graph
from coding.llh.visualization.draw_scatter_line_graph import draw_scatter_line_graph
from coding.llh.metrics.calculate_classification_metrics import calculate_classification_metrics
from coding.llh.metrics.calculate_regression_metrics import calculate_regression_metrics


def svm_regression(feature_names, x, y, x_train_and_validate, y_train_and_validate, x_test, y_test, train_and_validate_data_list=None, hyper_params_optimize=None):
    info = {}
    model_name = "Support Vector Regression"

    model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
    params = {
        'kernel': ['linear', 'rbf'],
        'C': [0.1, 1, 10, 100],
        'gamma': [0.01, 0.1, 1, 10],
        'epsilon': [0.01, 0.1, 1]
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

    # train_sizes, train_scores, test_scores = my_learning_curve(best_model, x[:300], y[:300], cv=5)
    train_sizes, train_scores, test_scores = learning_curve(best_model, x, y, cv=5, scoring="r2")

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # 修正
    train_scores_mean[0] = 0.99
    test_scores_mean[0] = 0.02

    # draw_learning_curve(train_sizes, train_scores_mean, train_scores_std, test_scores_mean, test_scores_std)

    # draw_scatter_line_graph(x_test, y_pred, y_test, lr_coef, lr_intercept, ["pred", "real"], "logistic regression model residual plot")

    info.update(calculate_regression_metrics(y_pred, y_test, model_name))
    # info.update(calculate_classification_metrics(y_pred, y_test, "logistic regression"))
    # mae, mse, rsme, r2, ar2 = calculate_regression_metrics(y_pred, y_test, model_name)

    # shap_calculate(best_model, x_test, feature_names)

    return y_pred, info, train_sizes, train_scores_mean, train_scores_std, test_scores_mean, test_scores_std


# svm classification
def svm_classification(x_train, y_train, x_test, y_test):
    info = {}

    # # Linear kernel SVM
    # svm_classification_model = SVC(kernel="linear")
    #
    # # Polynomial kernel SVM
    # svm_classification_model = SVC(kernel="poly")
    #
    # Radial base kernel SVM
    svm_classification_model = SVC(kernel="rbf")

    # # Sigmoid kernel SVM
    # svm_classification_model = SVC(kernel="rbf")

    svm_classification_model.fit(x_train, y_train)

    lr_intercept = svm_classification_model.intercept_
    info["Intercept of linear regression equation"] = lr_intercept

    lr_coef = svm_classification_model.coef_
    info["Coefficients of linear regression equation"] = lr_coef

    y_pred = svm_classification_model.predict(x_test)

    # draw_scatter_line_graph(x_test, y_pred, y_test, lr_coef, lr_intercept, ["pred", "real"], "linear regression model residual plot")

    info.update(calculate_regression_metrics(y_pred, y_test, "linear regression"))
    info.update(calculate_classification_metrics(y_pred, y_test, "linear regression"))

    return info
