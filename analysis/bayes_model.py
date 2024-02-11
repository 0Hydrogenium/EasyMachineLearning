from sklearn.naive_bayes import *

from coding.llh.visualization.draw_line_graph import draw_line_graph
from coding.llh.visualization.draw_scatter_line_graph import draw_scatter_line_graph
from coding.llh.metrics.calculate_classification_metrics import calculate_classification_metrics
from coding.llh.metrics.calculate_regression_metrics import calculate_regression_metrics


# Naive bayes classification
def naive_bayes_classification(x_train, y_train, x_test, y_test):
    info = {}

    # multinomial_naive_bayes_classification_model = MultinomialNB()
    Gaussian_naive_bayes_classification_model = GaussianNB()
    # bernoulli_naive_bayes_classification_model = BernoulliNB()
    # complement_naive_bayes_classification_model = ComplementNB()

    Gaussian_naive_bayes_classification_model.fit(x_train, y_train)

    y_pred = Gaussian_naive_bayes_classification_model.predict(x_test).reshape(-1, 1)

    # draw_scatter_line_graph(x_test, y_pred, y_test, lr_coef, lr_intercept, ["pred", "real"], "Gaussian naive bayes classification model residual plot")

    info.update(calculate_regression_metrics(y_pred, y_test, "Gaussian naive bayes classification"))
    info.update(calculate_classification_metrics(y_pred, y_test, "Gaussian naive bayes classification"))

    return info

