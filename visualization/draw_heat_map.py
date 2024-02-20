import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from static.config import Config


def draw_heat_map(x_data, col_list, paint_object, will_rotate=False):
    plt.rcParams.update({'figure.autolayout': True})

    plt.figure(figsize=(10, 8), dpi=300)

    if isinstance(x_data, np.ndarray):
        np_data = np.around(x_data.astype("float64"), 2)
        pd_data = pd.DataFrame(x_data)
    elif isinstance(x_data, pd.DataFrame):
        np_data = np.around(x_data.to_numpy().astype("float64"), 2)
        pd_data = x_data

    for i in range(np_data.shape[0]):
        for j in range(np_data.shape[1]):
            plt.text(j, i, np_data[i, j], ha="center", va="center", color="w")

    if will_rotate:
        plt.xticks(np.arange(len(col_list)), col_list, rotation=-90)
    else:
        plt.xticks(np.arange(len(col_list)), col_list)

    plt.yticks(np.arange(len(col_list)), col_list)
    plt.imshow(np_data)
    plt.colorbar(True)
    plt.tight_layout()

    plt.title(paint_object.get_name())

    plt.xlabel(paint_object.get_x_cur_label())
    plt.ylabel(paint_object.get_y_cur_label())

    paint_object.set_color_cur_num(0)

    return plt, paint_object

