import matplotlib.pyplot as plt
from sklearn.metrics import *

from classes.static_custom_class import *


def draw_roc_auc_curve_total(input_dict, type):
    plt.figure(figsize=(10, 6))

    if type == "train":
        i = 0
        for label_name, values in input_dict.items():
            fpr = values[0]
            tpr = values[1]
            thresholds = values[2]

            plt.plot(
                fpr,
                tpr,
                "o-",
                color=StaticValue.COLORS[i],
                label=label_name+str(round(auc(fpr, tpr), 2))
            )

            i += 1

        title = "Training roc-auc curve"
        plt.title(title)

    else:
        i = 0
        for label_name, values in input_dict.items():
            fpr = values[0]
            tpr = values[1]
            thresholds = values[2]

            plt.plot(
                fpr,
                tpr,
                "o-",
                color=StaticValue.COLORS[i],
                label=label_name + str(round(auc(fpr, tpr), 2))
            )

            i += 1

        title = "Cross-validation roc-auc curve"
        plt.title(title)

    plt.xlabel("tpr")
    plt.ylabel("fpr")
    plt.legend()

    plt.savefig("./diagram/{}.png".format(title), dpi=300)

    plt.show()