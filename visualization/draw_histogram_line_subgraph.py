import numpy as np
from matplotlib import pyplot as plt

from coding.llh.static.config import Config


def draw_histogram_line_subgraph(total_data_for_plot):
    # Manually adjust based on the data
    layout = """
        ABC
        DDE
        FGH
        IJK
    """

    fig, ax = plt.subplot_mosaic(layout, figsize=(16, 16))

    for i, data in enumerate(total_data_for_plot):
        if data[0] == "line_graph":
            ax[str(chr(i+65))].grid()
            ax[str(chr(i+65))].plot(
                data[1],
                data[2],
                "-o",
                color=Config.COLORS[0],
                markersize=4
            )
            ax[str(chr(i+65))].set_title(data[3])
        elif data[0] == "histogram":
            ax[str(chr(i+65))].grid()
            ax[str(chr(i+65))].bar(
                np.arange(0, len(data[1])),
                data[1],
                align="center",
                alpha=1,
                color=Config.COLORS,
                tick_label=data[2]
            )

            if data[3]:
                ax[str(chr(i+65))].tick_params(axis='x', labelrotation=-90)

            ax[str(chr(i+65))].set_title(data[5])

    plt.tight_layout()
    plt.savefig("./diagram/{}.png".format("total"), dpi=300)

    plt.show()