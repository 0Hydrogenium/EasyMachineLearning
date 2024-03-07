import matplotlib.pyplot as plt


def draw_momentum(df, p1_name, p2_name):
    plt.figure(figsize=(10, 6))

    plt.plot(
        df.loc[:, "elapsed_time"].values,
        df.loc[:, "p1_momentum_value"].values,
        "-",
        color=Config.COLORS_1[8],
        alpha=0.5,
        label=p1_name
    )
    plt.plot(
        df.loc[:, "elapsed_time"].values,
        df.loc[:, "p2_momentum_value"].values,
        "-",
        color=Config.COLORS_1[9],
        alpha=0.5,
        label=p2_name
    )
    plt.axhline(
        y=0,
        linestyle="--",
        color="black",
        alpha=0.5
    )
    plt.plot(
        df.loc[:, "elapsed_time"].values,
        df.loc[:, "p1_momentum_value_better"].values,
        "-",
        color=Config.COLORS_1[10],
        alpha=0.7,
        label="Degree of Superiority"
    )

    title = "Momentum"
    # plt.title(title)

    plt.xlabel("Elapsed time")
    plt.ylabel("Momentum value")
    plt.legend()

    plt.savefig("./diagram/{}.png".format(title), dpi=300)

    plt.show()