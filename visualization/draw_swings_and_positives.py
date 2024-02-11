import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import *
from sklearn.preprocessing import label_binarize

from coding.llh.static.config import Config


def draw_swings_and_positives(df, p1_name, p2_name):
    plt.figure(figsize=(10, 6))

    plt.plot(
        df.loc[:, "elapsed_time"].values,
        df.loc[:, "swing"].values,
        "-",
        color=Config.COLORS_2[2],
        alpha=0.7,
        label="Swing of Play"
    )
    plt.plot(
        df.loc[:, "elapsed_time"].values,
        df.loc[:, "p1_remain_positive"].values,
        "-.",
        color=Config.COLORS_2[0],
        alpha=0.7,
        label=p1_name
    )
    plt.plot(
        df.loc[:, "elapsed_time"].values,
        df.loc[:, "p2_remain_positive"].values,
        "-.",
        color=Config.COLORS_2[1],
        alpha=0.7,
        label=p2_name
    )

    title = "Standard time interval"
    # plt.title(title)

    plt.xlabel("Elapsed time")
    plt.ylabel("Standard time interval")
    plt.legend()

    plt.savefig("./diagram/{}.png".format(title), dpi=300)

    plt.show()