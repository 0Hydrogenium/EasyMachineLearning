import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from coding.llh.static.config import Config


# Draw heat map
def draw_heat_map(x_data, title, is_rotate, col_name):
    # col_name = np.delete(col_name, np.where(col_name == "swing"))

    plt.rcParams.update({'figure.autolayout': True})

    plt.figure(figsize=(16, 16))

    if isinstance(x_data, np.ndarray):
        np_data = np.around(x_data.astype("float64"), 2)
        pd_data = pd.DataFrame(x_data)
    elif isinstance(x_data, pd.DataFrame):
        np_data = np.around(x_data.to_numpy().astype("float64"), 2)
        pd_data = x_data

    for i in range(np_data.shape[0]):
        for j in range(np_data.shape[1]):
            plt.text(j, i, np_data[i, j], ha="center", va="center", color="w")

    if is_rotate:
        plt.xticks(np.arange(len(pd_data.columns.values)), col_name, rotation=-90)
    else:
        plt.xticks(np.arange(len(pd_data.columns.values)), col_name)

    plt.yticks(np.arange(len(pd_data.index.values)), col_name)
    plt.imshow(np_data)
    # plt.colorbar(False)
    plt.tight_layout()
    # plt.title(title)

    plt.savefig("./diagram/{}.png".format(title), dpi=300)

    plt.show()