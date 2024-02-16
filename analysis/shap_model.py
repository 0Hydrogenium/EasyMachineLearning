import matplotlib.pyplot as plt

import shap


def shap_calculate(model, x, feature_names, paint_object):
    explainer = shap.Explainer(model.predict, x)
    shap_values = explainer(x)

    shap.summary_plot(shap_values, x, feature_names=feature_names, show=False)

    plt.title(paint_object.get_name())

    return plt, paint_object




