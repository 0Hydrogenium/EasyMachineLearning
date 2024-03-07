import matplotlib.pyplot as plt
import numpy as np


def draw_histogram(nums, labels, paint_object, will_rotate=False, will_show_text=True):
    plt.clf()
    plt.figure(figsize=(10, 8), dpi=300)

    bars = plt.bar(
        np.arange(0, len(nums)),
        nums,
        align="center",
        alpha=1,
        color=paint_object.get_color_cur_list()[0],
        tick_label=labels
    )

    if will_show_text:
        for bar in bars:
            plt.annotate(
                str(bar.get_height()),
                xy=(bar.get_x() + bar.get_width() / 2,
                    bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                va="bottom",
                ha="center"
            )

    if will_rotate:
        plt.xticks(rotation=-45)

    plt.title(paint_object.get_name())

    plt.xlabel(paint_object.get_x_cur_label())
    plt.ylabel(paint_object.get_y_cur_label())

    paint_object.set_color_cur_num(1)

    return plt, paint_object

