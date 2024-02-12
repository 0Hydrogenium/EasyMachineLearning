import shap
import matplotlib.pyplot as plt


def shap_calculate(model, x, feature_names):
    explainer = shap.Explainer(model.predict, x)
    shap_values = explainer(x)

    return shap.summary_plot(shap_values, x, feature_names=feature_names)

    # title = "shap"
    # cur_plt.savefig("./diagram/{}.png".format(title), dpi=300)




