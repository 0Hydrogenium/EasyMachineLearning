import matplotlib.pyplot as plt

import shap


def shap_calculate(model, x, feature_names):
    explainer = shap.Explainer(model.predict, x)
    shap_values = explainer(x)

    shap.summary_plot(shap_values, x, feature_names=feature_names, show=False)

    return plt

    # title = "shap"




