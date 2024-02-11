import matplotlib.pyplot as plt

from coding.llh.static.config import Config


# draw boxplot
def draw_boxplot(x_data, title):
    plt.figure(figsize=(10, 14))
    plt.grid(True)

    plt.boxplot(
        x_data,
        meanline=True,
        showmeans=True,
        medianprops={"color": Config.COLORS[0], "linewidth": 1.5},
        meanprops={"color": Config.COLORS[1], "ls": "--", "linewidth": 1.5},
        flierprops={"marker": "o", "markerfacecolor": Config.COLORS[2]},
        labels=x_data.columns.values
    )

    plt.xticks(rotation=-45)
    plt.title(title)

    plt.savefig("./diagram/{}.png".format(title), dpi=300)

    plt.show()
