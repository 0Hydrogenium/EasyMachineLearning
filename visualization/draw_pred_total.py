import numpy as np
from matplotlib import pyplot as plt

from coding.llh.static.config import Config


def draw_pred_total(input_dict):
    plt.figure(figsize=(10, 6))

    for i, name, cur_list in enumerate(input_dict.items()):
        if i == len(input_dict.keys())-1:
            final_list = cur_list

        plt.plot(
            np.array([x for x in range(len(cur_list[0]))]),
            cur_list[0],
            "-",
            color=paint_object.get_color_cur_list()[i],
            alpha=0.9,
            label=paint_object.get_label_cur_list()[i]
        )

    plt.plot(
        np.array([x for x in range(len(final_list[1]))]),
        final_list[1],
        "--",
        color=paint_object.get_color_cur_list()[len(input_dict.keys())],
        alpha=0.9,
        label=paint_object.get_label_cur_list[len(input_dict.keys())]
    )

    plt.xlabel("Sizes")
    plt.ylabel("Value")
    plt.legend()

    plt.savefig("./diagram/{}.png".format(title), dpi=300)

    plt.show()


