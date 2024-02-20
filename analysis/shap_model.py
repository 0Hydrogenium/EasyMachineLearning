import matplotlib.pyplot as plt
import numpy as np
import shap


def draw_shap_beeswarm(model, x, feature_names, type, paint_object):
    explainer = shap.KernelExplainer(model.predict, x)
    shap_values = explainer(x)

    shap.summary_plot(shap_values, x, feature_names=feature_names, plot_type=type, show=False)

    plt.title(paint_object.get_name())
    plt.tight_layout()

    return plt, paint_object


def draw_waterfall(model, x, feature_names, number, paint_object):
    explainer = shap.KernelExplainer(model.predict, x, feature_names=feature_names)
    shap_values = explainer(x)

    shap.waterfall_plot(shap_values[number], show=False)

    plt.title(paint_object.get_name())
    plt.tight_layout()

    return plt, paint_object


def draw_force(model, x, feature_names, number, paint_object):
    explainer = shap.KernelExplainer(model.predict, x, feature_names=feature_names)
    shap_values = explainer(x[number])

    shap.force_plot(explainer.expected_value, shap_values.values, feature_names=feature_names, show=False, matplotlib=True)

    plt.title(paint_object.get_name())
    plt.tight_layout()

    return plt, paint_object


def draw_dependence(model, x, feature_names, col, paint_object):
    explainer = shap.KernelExplainer(model.predict, x, feature_names=feature_names)
    shap_values = explainer(x)

    shap.dependence_plot(feature_names.index(col), shap_values.values, x, feature_names=feature_names, show=False)

    plt.title(paint_object.get_name())
    plt.tight_layout()

    return plt, paint_object




