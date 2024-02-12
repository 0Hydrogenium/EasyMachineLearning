import numpy as np
import matplotlib.pyplot as plt

from static.config import Config


# draw line graph
def draw_line_graph(x_data, y_data: list, title):
    plt.figure(figsize=(10, 8))

    plt.plot(
        x_data,
        y_data,
        "-o",
        color=Config.COLORS[0]
    )

    plt.title(title)
    plt.savefig("./diagram/{}.png".format(title), dpi=300)

    plt.show()


def draw_line_graph_1(x_data, y_data: list, title, labels: list):
    plt.figure(figsize=(10, 8))

    for i, single_y_data in enumerate(y_data):
        plt.plot(
            x_data,
            single_y_data,
            "-o",
            color=Config.COLORS[i],
            label=labels[i]
        )

    plt.legend()
    plt.title(title)
    plt.savefig("./diagram/{}.png".format(title), dpi=300)

    plt.show()
