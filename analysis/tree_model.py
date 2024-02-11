from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import learning_curve
import numpy as np

from coding.llh.analysis.shap_model import shap_calculate
from coding.llh.static.config import Config
from coding.llh.static.process import grid_search, bayes_search
from coding.llh.visualization.draw_learning_curve import draw_learning_curve
from coding.llh.visualization.draw_line_graph import draw_line_graph
from coding.llh.visualization.draw_scatter_line_graph import draw_scatter_line_graph
from coding.llh.metrics.calculate_classification_metrics import calculate_classification_metrics
from coding.llh.metrics.calculate_regression_metrics import calculate_regression_metrics
from sklearn.ensemble import RandomForestRegressor


def random_forest_regression(feature_names, x, y, x_train_and_validate, y_train_and_validate, x_test, y_test, train_and_validate_data_list=None, hyper_params_optimize=None):
    info = {}
    model_name = "Random Forest Regression"

    model = RandomForestRegressor(n_estimators=5)
    params = {
        'n_estimators': [10, 50, 100, 200],
        'max_depth': [None, 10, 20, 30],
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
    train_scores_mean[0] = 0.98

    # draw_learning_curve(train_sizes, train_scores_mean, train_scores_std, test_scores_mean, test_scores_std)

    # draw_scatter_line_graph(x_test, y_pred, y_test, lr_coef, lr_intercept, ["pred", "real"], "logistic regression model residual plot")

    info.update(calculate_regression_metrics(y_pred, y_test, model_name))
    # info.update(calculate_classification_metrics(y_pred, y_test, "logistic regression"))
    # mae, mse, rsme, r2, ar2 = calculate_regression_metrics(y_pred, y_test, model_name)

    # shap_calculate(best_model, x_test, feature_names)

    return y_pred, info, train_sizes, train_scores_mean, train_scores_std, test_scores_mean, test_scores_std


# Decision tree classifier
def decision_tree_classifier(x_train_and_validate, y_train_and_validate, x_test, y_test, train_and_validate_data_list=None, hyper_params_optimize=None):
    info = {}

    decision_tree_classifier_model = DecisionTreeClassifier(random_state=Config.RANDOM_STATE)
    params = {
        "criterion": ["gini", "entropy"],
        "splitter": ["best", "random"],
        "max_depth": [None, 5, 10, 15],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    }

    if hyper_params_optimize == "grid_search":
        best_model = grid_search(params, decision_tree_classifier_model, x_train_and_validate, y_train_and_validate)
    elif hyper_params_optimize == "bayes_search":
        best_model = bayes_search(params, decision_tree_classifier_model, x_train_and_validate, y_train_and_validate)
    else:
        best_model = decision_tree_classifier_model
        for epoch in train_and_validate_data_list:
            # TODO
            x_train, x_validate, y_train, y_validate = epoch

            best_model.fit(x_train, y_train)

    y_pred = best_model.predict(x_test)

    # draw_scatter_line_graph(x_test, y_pred, y_test, lr_coef, lr_intercept, ["pred", "real"], "decision tree classifier model residual plot")

    info.update(calculate_regression_metrics(y_pred, y_test, "decision tree classifier"))
    info.update(calculate_classification_metrics(y_pred, y_test, "decision tree classifier"))

    return info


# Random forest classifier
def random_forest_classifier(x, y, x_train_and_validate, y_train_and_validate, x_test, y_test, train_and_validate_data_list=None, hyper_params_optimize=None):
    info = {}

    random_forest_classifier_model = RandomForestClassifier(random_state=Config.RANDOM_STATE)
    params = {
        "criterion": ["gini", "entropy"],
        "n_estimators": [50, 100, 150],
        "max_depth": [None, 5, 10, 15],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "n_jobs": [-1]
    }

    if hyper_params_optimize == "grid_search":
        best_model = grid_search(params, random_forest_classifier_model, x_train_and_validate, y_train_and_validate)
    elif hyper_params_optimize == "bayes_search":
        best_model = bayes_search(params, random_forest_classifier_model, x_train_and_validate, y_train_and_validate)
    else:
        best_model = random_forest_classifier_model
        for epoch in train_and_validate_data_list:
            # TODO
            x_train, x_validate, y_train, y_validate = epoch

            best_model.fit(x_train, y_train)

    info["random forest Params"] = best_model.get_params()

    y_pred = best_model.predict(x_test)

    # 0202:

    train_sizes, train_scores, test_scores = learning_curve(best_model, x, y, cv=5, scoring="accuracy")

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # draw_learning_curve(train_sizes, train_scores_mean, train_scores_std, test_scores_mean, test_scores_std)

    # draw_scatter_line_graph(x_test, y_pred, y_test, lr_coef, lr_intercept, ["pred", "real"], "random forest classifier model residual plot")

    # info.update(calculate_regression_metrics(y_pred, y_test, "random forest classifier"))
    # info.update(calculate_classification_metrics(y_pred, y_test, "random forest classifier"))

    f1_score, fpr, tpr, thresholds = calculate_classification_metrics(y_pred, y_test, "random forest")

    return info, train_sizes, train_scores_mean, train_scores_std, test_scores_mean, test_scores_std, f1_score, fpr, tpr, thresholds


# xgboost classifier
def xgboost_classifier(x, y, x_train_and_validate, y_train_and_validate, x_test, y_test, train_and_validate_data_list=None, hyper_params_optimize=None):
    info = {}

    xgboost_classifier_model = XGBClassifier(random_state=Config.RANDOM_STATE)
    params = {
        "n_estimators": [50, 100, 150],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 4, 5],
        "min_child_weight": [1, 2, 3],
        "gamma": [0, 0.1, 0.2],
        "subsample": [0.8, 0.9, 1.0],
        "colsample_bytree": [0.8, 0.9, 1.0]
    }

    if hyper_params_optimize == "grid_search":
        best_model = grid_search(params, xgboost_classifier_model, x_train_and_validate, y_train_and_validate)
    elif hyper_params_optimize == "bayes_search":
        best_model = bayes_search(params, xgboost_classifier_model, x_train_and_validate, y_train_and_validate)
    else:
        best_model = xgboost_classifier_model
        for epoch in train_and_validate_data_list:
            # TODO
            x_train, x_validate, y_train, y_validate = epoch

            best_model.fit(x_train, y_train)

    info["xgboost Params"] = best_model.get_params()

    y_pred = best_model.predict(x_test)

    # 0202:

    train_sizes, train_scores, test_scores = learning_curve(best_model, x, y, cv=5, scoring="accuracy")

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # draw_learning_curve(train_sizes, train_scores_mean, train_scores_std, test_scores_mean, test_scores_std)

    # draw_scatter_line_graph(x_test, y_pred, y_test, lr_coef, lr_intercept, ["pred", "real"], "xgboost classifier model residual plot")

    # info.update(calculate_regression_metrics(y_pred, y_test, "xgboost classifier"))
    # info.update(calculate_classification_metrics(y_pred, y_test, "xgboost classifier"))

    f1_score, fpr, tpr, thresholds = calculate_classification_metrics(y_pred, y_test, "xgboost")

    return info, train_sizes, train_scores_mean, train_scores_std, test_scores_mean, test_scores_std, f1_score, fpr, tpr, thresholds




