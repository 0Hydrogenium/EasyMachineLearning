import numpy as np
from matplotlib import pyplot as plt

from coding.llh.static.config import Config


def draw_pred_total(input_dict):
    plt.figure(figsize=(10, 6))

    i = 0
    for name, cur_list in input_dict.items():
        mylist = cur_list
        plt.plot(
            np.array([x for x in range(len(cur_list[0]))]),
            cur_list[0],
            "-",
            color=Config.COLORS_4[i],
            alpha=0.9,
            label=name
        )
        i += 1

    plt.plot(
        np.array([x for x in range(len(mylist[1]))]),
        mylist[1],
        "--",
        color=Config.COLORS_4[1],
        alpha=0.9,
        label="actual data"
    )

    title = "pred curve"

    plt.xlabel("Sizes")
    plt.ylabel("Value")
    plt.legend()

    plt.savefig("./diagram/{}.png".format(title), dpi=300)

    plt.show()


