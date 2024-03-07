from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV


def grid_search(params, model, x_train, y_train, scoring=None):
    info = {}

    grid_search_model = GridSearchCV(model, params, cv=3, n_jobs=-1)

    grid_search_model.fit(x_train, y_train.ravel())

    info["Optimal hyperparameters"] = grid_search_model.best_params_

    best_model = grid_search_model.best_estimator_

    return best_model


def bayes_search(params, model, x_train, y_train, scoring=None):
    info = {}

    bayes_search_model = BayesSearchCV(model, params, cv=3, n_iter=50, n_jobs=-1)

    bayes_search_model.fit(x_train, y_train)

    info["Optimal hyperparameters"] = bayes_search_model.best_params_

    best_model = bayes_search_model.best_estimator_

    return best_model