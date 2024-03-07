import lightgbm as lightGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import learning_curve
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from analysis.others.shap_model import *
from functions.process import get_values_from_container_class, transform_params_list
from metrics.calculate_classification_metrics import calculate_classification_metrics
from metrics.calculate_regression_metrics import calculate_regression_metrics
from analysis.others.hyperparam_optimize import *
from classes.static_custom_class import StaticValue


class RandomForestRegressionParams:
    @classmethod
    def get_params_type(cls):
        return {
            'n_estimators': StaticValue.INT,
            'max_depth': StaticValue.INT,
            'min_samples_split': StaticValue.INT,
            'min_samples_leaf': StaticValue.INT,
        }

    @classmethod
    def get_params(cls):
        return {
            'n_estimators': [10, 50, 100, 200],
            'max_depth': [0, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
        }


# 随机森林回归
def random_forest_regressor(container, params):
    x_train, y_train, x_test, y_test, hyper_params_optimize = get_values_from_container_class(container)
    info = {}

    params = transform_params_list(RandomForestRegressionParams, params)
    params['random_state'] = [StaticValue.RANDOM_STATE]

    random_forest_regression_model = RandomForestRegressor(n_estimators=5, random_state=StaticValue.RANDOM_STATE)

    if hyper_params_optimize == "grid_search":
        best_model = grid_search(params, random_forest_regression_model, x_train, y_train)
    elif hyper_params_optimize == "bayes_search":
        best_model = bayes_search(params, random_forest_regression_model, x_train, y_train)
    else:
        best_model = random_forest_regression_model
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


class DecisionTreeClassifierParams:
    @classmethod
    def get_params_type(cls):
        return {
            "criterion": StaticValue.STR,
            "splitter": StaticValue.STR,
            "max_depth": StaticValue.INT,
            "min_samples_split": StaticValue.INT,
            "min_samples_leaf": StaticValue.INT,
        }

    @classmethod
    def get_params(cls):
        return {
            "criterion": ["gini", "entropy"],
            "splitter": ["best", "random"],
            "max_depth": [0, 5, 10, 15],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }


# 决策树分类
def decision_tree_classifier(container, params):
    x_train, y_train, x_test, y_test, hyper_params_optimize = get_values_from_container_class(container)
    info = {}

    params = transform_params_list(DecisionTreeClassifierParams, params)
    params['random_state'] = [StaticValue.RANDOM_STATE]

    random_forest_regression_model = DecisionTreeClassifier(random_state=StaticValue.RANDOM_STATE)

    if hyper_params_optimize == "grid_search":
        best_model = grid_search(params, random_forest_regression_model, x_train, y_train)
    elif hyper_params_optimize == "bayes_search":
        best_model = bayes_search(params, random_forest_regression_model, x_train, y_train)
    else:
        best_model = random_forest_regression_model
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


class RandomForestClassifierParams:
    @classmethod
    def get_params_type(cls):
        return {
            "criterion": StaticValue.STR,
            "n_estimators": StaticValue.INT,
            "max_depth": StaticValue.INT,
            "min_samples_split": StaticValue.INT,
            "min_samples_leaf": StaticValue.INT,
        }

    @classmethod
    def get_params(cls):
        return {
            "criterion": ["gini", "entropy"],
            "n_estimators": [50, 100, 150],
            "max_depth": [0, 5, 10, 15],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }


# 随机森林分类
def random_forest_classifier(container, params):
    x_train, y_train, x_test, y_test, hyper_params_optimize = get_values_from_container_class(container)
    info = {}

    params = transform_params_list(RandomForestClassifierParams, params)
    params['random_state'] = [StaticValue.RANDOM_STATE]

    random_forest_classifier_model = RandomForestClassifier(n_estimators=5, random_state=StaticValue.RANDOM_STATE)

    if hyper_params_optimize == "grid_search":
        best_model = grid_search(params, random_forest_classifier_model, x_train, y_train)
    elif hyper_params_optimize == "bayes_search":
        best_model = bayes_search(params, random_forest_classifier_model, x_train, y_train)
    else:
        best_model = random_forest_classifier_model
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


class XgboostClassifierParams:
    @classmethod
    def get_params_type(cls):
        return {
            "n_estimators": StaticValue.INT,
            "learning_rate": StaticValue.FLOAT,
            "max_depth": StaticValue.INT,
            "min_child_weight": StaticValue.INT,
            "gamma": StaticValue.FLOAT,
            "subsample": StaticValue.FLOAT,
            "colsample_bytree": StaticValue.FLOAT,
        }

    @classmethod
    def get_params(cls):
        return {
            "n_estimators": [50, 100, 150],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 4, 5],
            "min_child_weight": [1, 2, 3],
            "gamma": [0, 0.1, 0.2],
            "subsample": [0.5, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.8, 0.9, 1.0],
        }


# xgboost分类
def xgboost_classifier(container, params):
    x_train, y_train, x_test, y_test, hyper_params_optimize = get_values_from_container_class(container)
    info = {}

    params = transform_params_list(XgboostClassifierParams, params)
    params['random_state'] = [StaticValue.RANDOM_STATE]

    xgboost_classifier_model = XGBClassifier(random_state=StaticValue.RANDOM_STATE)

    if hyper_params_optimize == "grid_search":
        best_model = grid_search(params, xgboost_classifier_model, x_train, y_train)
    elif hyper_params_optimize == "bayes_search":
        best_model = bayes_search(params, xgboost_classifier_model, x_train, y_train)
    else:
        best_model = xgboost_classifier_model
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


class LightGBMClassifierParams:
    @classmethod
    def get_params(cls):
        return


# lightGBM分类
def lightGBM_classifier(container, params):
    x_train, y_train, x_test, y_test, hyper_params_optimize = get_values_from_container_class(container)
    info = {}

    params = transform_params_list(LightGBMClassifierParams, params)
    params['random_state'] = [StaticValue.RANDOM_STATE]

    lightgbm_classifier_model = lightGBMClassifier

    if hyper_params_optimize == "grid_search":
        best_model = grid_search(params, lightgbm_classifier_model, x_train, y_train)
    elif hyper_params_optimize == "bayes_search":
        best_model = bayes_search(params, lightgbm_classifier_model, x_train, y_train)
    else:
        best_model = lightgbm_classifier_model
        best_model.train(x_train, y_train)

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



