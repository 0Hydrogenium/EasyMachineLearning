import numpy as np
from matplotlib import pyplot as plt

from static.new_class import PaintObject
from static.config import Config


def draw_data_fit_total(input_dict, paint_object: PaintObject):
    plt.figure(figsize=(10, 6), dpi=300)

    for i, input_dict_items in enumerate(input_dict.items()):
        name, cur_list = input_dict_items

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
        label=paint_object.get_label_cur_list()[len(input_dict.keys())]
    )

    plt.title(paint_object.get_name())

    plt.xlabel(paint_object.get_x_cur_label())
    plt.ylabel(paint_object.get_y_cur_label())
    plt.legend()

    # plt.savefig("./diagram/{}.png".format(title), dpi=300)
    # plt.show()

    paint_object.set_color_cur_num(len(input_dict.values())+1)
    paint_object.set_label_cur_num(len(input_dict.values())+1)

    return plt, paint_object

