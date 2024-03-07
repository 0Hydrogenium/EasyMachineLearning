import matplotlib.pyplot as plt
import numpy as np

from classes.static_custom_class import *


# draw scatter line graph
def draw_scatter_line_graph(x_data, y_pred_data, y_real_data, coef, intercept, labels, title):
    # Manually adjust based on the data
    layout = """
        ABCDE
        FGHIJ
    """

    fig, ax = plt.subplot_mosaic(layout, figsize=(16, 16))

    for i in range(np.size(x_data, 1)):
        ax[str(chr(i+65))].scatter(x_data[:, i], y_pred_data.T, color=StaticValue.COLORS[0], s=4, label=labels[0])
        ax[str(chr(i+65))].scatter(x_data[:, i], y_real_data, color=StaticValue.COLORS[1], s=4, label=labels[1])
        ax[str(chr(i+65))].plot(x_data[:, i], x_data[:, i] * coef[i] + intercept, color=StaticValue.COLORS[2], markersize=4)
        ax[str(chr(i + 65))].legend()

    plt.suptitle(title)

    plt.savefig("./diagram/{}.png".format(title), dpi=300)

    plt.show()
