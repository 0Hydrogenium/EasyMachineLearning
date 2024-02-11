
from datetime import datetime

import json
import sys
import numpy as np
import pandas as pd
import math
import time as sys_time

from coding.llh.visualization.draw_boxplot import draw_boxplot
from coding.llh.visualization.draw_heat_map import draw_heat_map
from coding.llh.visualization.draw_histogram import draw_histogram
from coding.llh.visualization.draw_histogram_line_subgraph import draw_histogram_line_subgraph
from coding.llh.visualization.draw_line_graph import draw_line_graph
from tqdm import tqdm


# 0202:
def data_transformation_extra(df: pd.DataFrame, str2int_mappings: dict) -> (pd.DataFrame):

    # Delete "match_id" column
    # df.drop("match_id", axis=1, inplace=True)
    df["match_id"] = df["match_id"].apply(lambda x: x[-4:])

    # Dissolve the two-mode data mapping into two part

    value_to_replace_dict = {
        "AD": "50"
    }

    value_to_replace = "AD"
    df["p1_score"].replace(value_to_replace, value_to_replace_dict[value_to_replace], inplace=True)
    df["p2_score"].replace(value_to_replace, value_to_replace_dict[value_to_replace], inplace=True)

    str2int_mappings_to_dissolve = {
        "p1_score": {"0": 0},
        "p2_score": {"0": 0}
    }

    df["p1_score_mark"] = 0
    df["p2_score_mark"] = 0

    for key in str2int_mappings_to_dissolve.keys():
        for i in range(1, len(df)):
            if df.loc[i, key] == "15" and df.loc[i-1, key] == "0":
                df.loc[i, key+"_mark"] = 1
            elif df.loc[i, key] == "1" and df.loc[i-1, key] == "0":
                df.loc[i, key + "_mark"] = 2

    df["p1_score_normal"] = 0
    df["p1_score_tiebreak"] = 0
    df["p2_score_normal"] = 0
    df["p2_score_tiebreak"] = 0

    normal_counter = 0
    tiebreak_counter = 0
    for key in str2int_mappings_to_dissolve.keys():
        for i in range(0, len(df)):
            if df.loc[i, key] == "0":
                normal_counter = 0
                tiebreak_counter = 0
                continue

            if df.loc[i, key+"_mark"] == 1 or normal_counter > 0:
                if int(df.loc[i, key]) > int(df.loc[i-1, key]):
                    normal_counter += 1
                    df.loc[i, key + "_normal"] = normal_counter
                    if df.loc[i, key] == value_to_replace_dict[value_to_replace]:
                        str2int_mappings_to_dissolve[key][value_to_replace] = normal_counter
                    else:
                        str2int_mappings_to_dissolve[key][df.loc[i, key]] = normal_counter

                elif int(df.loc[i, key]) < int(df.loc[i-1, key]):
                    normal_counter -= 1
                    df.loc[i, key + "_normal"] = normal_counter

                else:
                    df.loc[i, key + "_normal"] = normal_counter

            elif df.loc[i, key+"_mark"] == 2 or tiebreak_counter > 0:
                if int(df.loc[i, key]) > int(df.loc[i - 1, key]):
                    tiebreak_counter += 1
                    df.loc[i, key+"_tiebreak"] = tiebreak_counter
                    if df.loc[i, key] == value_to_replace_dict[value_to_replace]:
                        str2int_mappings_to_dissolve[key][value_to_replace] = tiebreak_counter
                    else:
                        str2int_mappings_to_dissolve[key][df.loc[i, key]] = tiebreak_counter

                elif int(df.loc[i, key]) < int(df.loc[i - 1, key]):
                    tiebreak_counter -= 1
                    df.loc[i, key+"_tiebreak"] = tiebreak_counter

                else:
                    df.loc[i, key + "_tiebreak"] = tiebreak_counter

    str2int_mappings.update(str2int_mappings_to_dissolve)

    df.drop("p1_score_mark", axis=1, inplace=True)
    df.drop("p2_score_mark", axis=1, inplace=True)
    df.drop("p1_score", axis=1, inplace=True)
    df.drop("p2_score", axis=1, inplace=True)

    # Transform "elapsed_time" time column

    def transform_time_col(time: str):
        h, m, s = time.strip().split(":")
        seconds = int(h) * 3600 + int(m) * 60 + int(s)
        return seconds

    df["elapsed_time"] = df["elapsed_time"].apply(transform_time_col)

    # Calculate "game_victor", "set_victor" column cumulative value

    df["p1_game_victor"] = df.apply(lambda x: 1 if x["game_victor"] == 1 else 0, axis=1)
    df["p2_game_victor"] = df.apply(lambda x: 1 if x["game_victor"] == 2 else 0, axis=1)
    df["p1_set_victor"] = df.apply(lambda x: 1 if x["set_victor"] == 1 else 0, axis=1)
    df["p2_set_victor"] = df.apply(lambda x: 1 if x["set_victor"] == 2 else 0, axis=1)

    df["p1_game_victor"] = df.groupby(["player1", "player2"])["p1_game_victor"].cumsum()
    df["p2_game_victor"] = df.groupby(["player1", "player2"])["p2_game_victor"].cumsum()
    df["p1_set_victor"] = df.groupby(["player1", "player2"])["p1_set_victor"].cumsum()
    df["p2_set_victor"] = df.groupby(["player1", "player2"])["p2_set_victor"].cumsum()

    # Forced conversion of data types
    for col in df.columns.values:
        df[col] = df[col].astype("float")

    # Save the mappings to a json format file
    with open("./data/mappings.json", "w", encoding="utf-8") as f:
        json.dump(str2int_mappings, f, indent=4, ensure_ascii=False)

    return df


def data_transformation(df: pd.DataFrame) -> (pd.DataFrame, dict):
    """
    0.
    1. Define mappings
    2. Create mappings
    3. Modify the original data according to the mappings
    4. Get type exception
    5. Forced conversion of data types
    """

    info = {}

    # Define mappings
    str2int_mappings = {
        "player1": {},
        "player2": {},
        "winner_shot_type": {},
        "serve_width": {},
        "serve_depth": {},
        "return_depth": {}
    }

    # Create mappings
    for col in str2int_mappings.copy():
        keys = np.array(df[col].drop_duplicates())
        values = [x for x in range(len(keys))]
        str2int_mappings[col] = dict(zip(keys, values))

    # Modify the original data according to the mappings
    for col, mapping in str2int_mappings.items():
        series = df[col]

        for k, v in mapping.items():
            series.replace(k, v, inplace=True)
        df[col] = series

    df.replace('Not A Number', 0, inplace=True)

    # Get type exception

    # abnormal_type_values = []
    #
    # for col in df.columns.values:
    #     if col not in str2int_mappings.keys():
    #         for row in df[col]:

    #             if not (0 <= row <= sys.maxsize):
    #                 abnormal_type_values.append(row)
    #
    # info["Number of abnormal type value"] = sorted(abnormal_type_values)


    # # Forced conversion of data types
    # for col in df.columns.values:
    #     df[col] = df[col].astype("float")
    #
    # # Save the mappings to a json format file
    # with open("./mappings.json", "w", encoding="utf-8") as f:
    #     json.dump(str2int_mappings, f, indent=4, ensure_ascii=False)


    # 0202:
    df = data_transformation_extra(df, str2int_mappings)

    return df, info


# Get descriptive indicators and filtered data based on boxplpot
def get_descriptive_indicators_related(df):
    info = {}

    descriptive_indicators_df = pd.DataFrame(
        index=list(df.columns.values),
        columns=[
            "Min",
            "Max",
            "Avg",
            "Standard Deviation",
            "Standard Error",
            "Upper Quartile",
            "Median",
            "Lower Quartile",
            "Interquartile Distance",
            "Kurtosis",
            "Skewness",
            "Coefficient of Variation"
        ]
    )

    for col in df.columns.values:
        descriptive_indicators_df["Min"][col] = df[col].min()
        descriptive_indicators_df["Max"][col] = df[col].max()
        descriptive_indicators_df["Avg"][col] = df[col].mean()
        descriptive_indicators_df["Standard Deviation"][col] = df[col].std()
        descriptive_indicators_df["Standard Error"][col] = descriptive_indicators_df["Standard Deviation"][col] / \
                                                           math.sqrt(len(df[col]))
        descriptive_indicators_df["Upper Quartile"][col] = df[col].quantile(0.75)
        descriptive_indicators_df["Median"][col] = df[col].quantile(0.5)
        descriptive_indicators_df["Lower Quartile"][col] = df[col].quantile(0.25)
        descriptive_indicators_df["Interquartile Distance"][col] = descriptive_indicators_df["Lower Quartile"][col] - \
                                                                   descriptive_indicators_df["Upper Quartile"][col]
        descriptive_indicators_df["Kurtosis"][col] = df[col].kurt()
        descriptive_indicators_df["Skewness"][col] = df[col].skew()
        descriptive_indicators_df["Coefficient of Variation"][col] = descriptive_indicators_df["Standard Deviation"][
                                                                         col] \
                                                                     / descriptive_indicators_df["Avg"][col]

    # draw_heat_map(descriptive_indicators_df.to_numpy(), "descriptive indicators", True)
    #
    # draw_boxplot(df, "descriptive indicators boxplot")

    len_0 = len(df)

    # tmp_df = \
    # df[(df >= (descriptive_indicators_df["Lower Quartile"] - 1.5 * (descriptive_indicators_df["Upper Quartile"] -
    #                                                                 descriptive_indicators_df["Lower Quartile"])))
    #    & (df <= (descriptive_indicators_df["Upper Quartile"] + 1.5 * (descriptive_indicators_df["Upper Quartile"] -
    #                                                                   descriptive_indicators_df["Lower Quartile"])))][[
    #     "ProductChoice", "MembershipPoints", "ModeOfPayment", "ResidentCity", "PurchaseTenure", "IncomeClass",
    #     "CustomerPropensity", "CustomerAge", "LastPurchaseDuration"
    # ]]

    # tmp_df.dropna(inplace=True)

    # df = pd.concat([tmp_df, df[["ProductChoice", "Channel", "MartialStatus"]]], axis=1, join="inner")

    # df = pd.concat([df.iloc[:, :9], df.iloc[:, 10:]], axis=1)

    # info["Number of offsetting value"] = len_0 - len(df)
    #
    # info["Total size of filtered data after descriptive analysis"] = len(df)

    return df, info


# Create images of the distribution of the number of each variable
def variable_distribution(df):
    counts_mappings = {}
    print("counts analysis")
    for col in tqdm(df.columns.values, desc='columns:'):
        counts_mapping = {}
        for x in tqdm(df[col], desc='cells'):
            if x in counts_mapping.keys():
                counts_mapping[x] += 1
            else:
                counts_mapping[x] = 1
        counts_mappings[col] = counts_mapping

    total_data_for_plot = []
    print("plotting")
    for col, mapping in tqdm(counts_mappings.items(), desc='columns'):
        if col in ["set_no", 'game_no']:
            sorting = sorted(mapping.items(), reverse=True, key=lambda m: m[0])
            data = [x[1] for x in sorting]
            labels = [x[0] for x in sorting]

            total_data_for_plot.append(["line_graph", labels, data, col])
            draw_line_graph(labels, data, col)
        else:
            sorting = sorted(mapping.items(), reverse=True, key=lambda m: m[1])
            data = [x[1] for x in sorting]
            labels = [x[0] for x in sorting]

            will_rotate = True if col in ["player1","player2", "match_id"] else False
            will_show_text = False if col in ["ResidentCity"] else True

            total_data_for_plot.append(["histogram", data, labels, will_rotate, will_show_text, col])
            draw_histogram(data, labels, will_rotate, will_show_text, col)
    # draw_histogram_line_subgraph(total_data_for_plot)
