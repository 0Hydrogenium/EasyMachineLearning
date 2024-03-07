import numpy as np
from sklearn.model_selection import learning_curve
from sklearn.naive_bayes import *
from analysis.others.hyperparam_optimize import *
from classes.static_custom_class import StaticValue
from functions.process import transform_params_list, get_values_from_container_class

from metrics.calculate_classification_metrics import calculate_classification_metrics


class NaiveBayesClassifierParams:
    @classmethod
    def get_params_type(cls, sort):
        if sort == "MultinomialNB":
            return {
                "alpha": StaticValue.FLOAT
            }
        elif sort == "GaussianNB":
            return {}
        elif sort == "ComplementNB":
            return {
                "alpha": StaticValue.FLOAT,
                "fit_prior": StaticValue.BOOL,
                "norm": StaticValue.BOOL
            }

    @classmethod
    def get_params(cls, sort):
        if sort == "MultinomialNB":
            return {
                "alpha": [0.1, 0.5, 1.0, 2.0]
            }
        elif sort == "GaussianNB":
            return {}
        elif sort == "ComplementNB":
            return {
                "alpha": [0.1, 0.5, 1, 10],
                "fit_prior": [True, False],
                "norm": [True, False]
            }


# 朴素贝叶斯分类
def naive_bayes_classifier(container, params, model=None):
    x_train, y_train, x_test, y_test, hyper_params_optimize = get_values_from_container_class(container)
    info = {}

    params = transform_params_list(NaiveBayesClassifierParams, params, model)

    if model == "MultinomialNB":
        naive_bayes_model = MultinomialNB()
        params = params
    elif model == "GaussianNB":
        naive_bayes_model = GaussianNB()
        params = params
    elif model == "ComplementNB":
        naive_bayes_model = ComplementNB()
        params = params
    else:
        naive_bayes_model = GaussianNB()
        params = params

    if hyper_params_optimize == "grid_search":
        best_model = grid_search(params, naive_bayes_model, x_train, y_train)
    elif hyper_params_optimize == "bayes_search":
        best_model = bayes_search(params, naive_bayes_model, x_train, y_train)
    else:
        best_model = naive_bayes_model
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

