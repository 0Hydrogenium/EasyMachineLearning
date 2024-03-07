import matplotlib.pyplot as plt
import pandas as pd


def draw_parallel_coordinates(df):
    df.drop("match_id", axis=1, inplace=True)
    df.drop("player1", axis=1, inplace=True)
    df.drop("player2", axis=1, inplace=True)
    df.drop("elapsed_time", axis=1, inplace=True)
    df.drop("set_no", axis=1, inplace=True)
    df.drop("game_no", axis=1, inplace=True)
    df.drop("point_no", axis=1, inplace=True)
    df.drop("p1_sets", axis=1, inplace=True)
    df.drop("p2_sets", axis=1, inplace=True)
    df.drop("p1_games", axis=1, inplace=True)
    df.drop("p2_games", axis=1, inplace=True)
    df.drop("p1_points_won", axis=1, inplace=True)
    df.drop("p2_points_won", axis=1, inplace=True)
    df.drop("p1_distance_run", axis=1, inplace=True)
    df.drop("p2_distance_run", axis=1, inplace=True)
    df.drop("speed_mph", axis=1, inplace=True)
    df.drop("p1_score_normal", axis=1, inplace=True)
    df.drop("p2_score_normal", axis=1, inplace=True)
    df.drop("p1_score_tiebreak", axis=1, inplace=True)
    df.drop("p2_score_tiebreak", axis=1, inplace=True)
    df.drop("p1_game_victor", axis=1, inplace=True)
    df.drop("p2_game_victor", axis=1, inplace=True)
    df.drop("p1_set_victor", axis=1, inplace=True)
    df.drop("p2_set_victor", axis=1, inplace=True)

    plt.figure(figsize=(10, 6))

    pd.plotting.parallel_coordinates(df, "point_victor", colormap="viridis")

    title = "Parallel Coordinates Plot"
    plt.title(title)

    plt.xlabel("Attributes")
    plt.ylabel("Values")
    plt.legend()

    plt.savefig("./diagram/{}.png".format(title), dpi=300)

    plt.show()
