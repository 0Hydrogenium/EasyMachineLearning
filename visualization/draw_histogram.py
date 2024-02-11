import numpy as np
import matplotlib.pyplot as plt

from coding.llh.static.config import Config


# Plot bar charts
def draw_histogram(x_data, y_data, will_rotate, will_show_text, title):
    fig, ax = plt.subplots(figsize=(10, 8))

    bars = plt.bar(
        np.arange(0, len(x_data)),
        x_data,
        align="center",
        alpha=1,
        color=Config.COLORS,
        tick_label=y_data
    )

    # Bar annotation
    if will_show_text:
        for bar in bars:
            ax.annotate(
                str(bar.get_height()),
                xy=(bar.get_x() + bar.get_width() / 2,
                    bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                va="bottom",
                ha="center"
            )

    if will_rotate:
        plt.xticks(rotation=-90)

    plt.title(title)

    plt.savefig("./diagram/{}.png".format(title), dpi=300)

    plt.show()