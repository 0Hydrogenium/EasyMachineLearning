import matplotlib.pyplot as plt


def draw_play_flow(df, p1_name, p2_name, p1_ace, p2_ace, p1_net_pt_won, p2_net_pt_won, p1_break_pt_won, p2_break_pt_won):
    plt.figure(figsize=(10, 6))

    plt.plot(
        df.loc[:, "elapsed_time"].values,
        df.loc[:, "p1_points_won"].values,
        "-",
        color=Config.COLORS_1[6],
        alpha=0.5,
        label=p1_name
    )
    plt.plot(
        df.loc[:, "elapsed_time"].values,
        df.loc[:, "p2_points_won"].values,
        "-",
        color=Config.COLORS_1[7],
        alpha=0.5,
        label=p2_name
    )

    plt.scatter(
        p1_ace.loc[:, "elapsed_time"].values,
        p1_ace.loc[:, "p1_points_won"].values,
        s=40,
        c=Config.COLORS_1[0],
        marker="v",
        label="p1_ace"
    )
    plt.scatter(
        p2_ace.loc[:, "elapsed_time"].values,
        p2_ace.loc[:, "p2_points_won"].values,
        s=40,
        c=Config.COLORS_1[1],
        marker="v",
        label="p2_ace"
    )
    plt.scatter(
        p1_net_pt_won.loc[:, "elapsed_time"].values,
        p1_net_pt_won.loc[:, "p1_points_won"].values,
        s=40,
        c=Config.COLORS_1[2],
        marker="*",
        label="p1_net_pt_won"
    )
    plt.scatter(
        p2_net_pt_won.loc[:, "elapsed_time"].values,
        p2_net_pt_won.loc[:, "p2_points_won"].values,
        s=40,
        c=Config.COLORS_1[3],
        marker="*",
        label="p2_net_pt_won"
    )
    plt.scatter(
        p1_break_pt_won.loc[:, "elapsed_time"].values,
        p1_break_pt_won.loc[:, "p1_points_won"].values,
        s=40,
        c=Config.COLORS_1[4],
        marker="+",
        label="p1_break_pt_won"
    )
    plt.scatter(
        p2_break_pt_won.loc[:, "elapsed_time"].values,
        p2_break_pt_won.loc[:, "p2_points_won"].values,
        s=40,
        c=Config.COLORS_1[5],
        marker="+",
        label="p1_break_pt_won"
    )

    title = "Flow of play"
    # plt.title(title)

    plt.xlabel("Elapsed time")
    plt.ylabel("Points")
    plt.legend()

    plt.savefig("./diagram/{}.png".format(title), dpi=300)

    plt.show()