import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from coding.llh.metrics.calculate_regression_metrics import calculate_ar2


def my_learning_curve(estimator, X, y, cv=5):
    train_sizes = np.linspace(0.1, 1.0, 10)[:-1]
    train_scores = []
    val_scores = []

    for train_size in train_sizes:
        # Split the dataset into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=train_size, random_state=42)

        # Train the model on the training set
        # estimator.fit(X_train, y_train)

        # Evaluate the model on the training set
        y_train_pred = estimator.predict(X_train)
        train_accuracy = r2_score(y_train, y_train_pred)
        train_scores.append(train_accuracy)

        # Evaluate the model on the validation set
        y_val_pred = estimator.predict(X_val)
        val_accuracy = r2_score(y_val, y_val_pred)
        val_scores.append(val_accuracy)

    return train_sizes, train_scores, val_scores

