import copy

import numpy as np
import pandas as pd
import skfuzzy.control.fuzzyvariable
from sklearn.mixture import GaussianMixture

from coding.llh.analysis.evaluation_model import *
from coding.llh.analysis.gaussian_model import gaussian_mix
from coding.llh.analysis.markov_model import train_and_predict_hidden_markov_model
from coding.llh.analysis.poly_model import poly_fit
from coding.llh.static.col import *
from coding.llh.visualization.draw_learning_curve_total import draw_learning_curve_total
from coding.llh.visualization.draw_momentum import draw_momentum
from coding.llh.visualization.draw_parallel_coordinates import draw_parallel_coordinates
from coding.llh.visualization.draw_play_flow import draw_play_flow
from coding.llh.visualization.draw_roc_auc_curve_total import draw_roc_auc_curve_total
from static.config import *
from static.process import *
from analysis.descriptive_analysis import *
from analysis.exploratory_analysis import *
from analysis.linear_model import *
from analysis.tree_model import *
from analysis.kernel_model import *
from analysis.bayes_model import *
from analysis.neural_model import *


pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)

total_info = {}

# 读取excel
df = pd.read_csv("../data/Wimbledon_featured_matches.csv")

df, new_info = preprocess_raw_data_filtering(copy.deepcopy(df))
total_info.update(new_info)

_, p1_name, p2_name = pick_matches_with_longest(df)

df = replace_na_to_label(df)

df, new_info = data_transformation(copy.deepcopy(df))
total_info.update(new_info)

df.to_csv("./data/data_after_mapping.csv", index=False)

# Get descriptive indicators and filtered data
# df, new_info = get_descriptive_indicators_related(copy.deepcopy(df))
# total_info.update(new_info)

# Get the standardized data
# array = get_standardized_data(copy.deepcopy(df))

# df = pd.DataFrame(data=get_standardized_data(df), columns=df.columns.values)

# Principal component analysis
projected_array, new_info = pca(copy.deepcopy(df))
# total_info.update(new_info)

# K-means
# new_info = k_means(preprocessing.scale(copy.deepcopy(projected_array)))
# total_info.update(new_info)

# df = remove_nan_from_data(df)

# 将"point_victor"列放到最前面
df = choose_y_col_in_dataframe(copy.deepcopy(df), "point_victor")

# df, _, _ = pick_matches_with_longest(df)

# 根据比赛切分总数据
match_dict = match_split(df)

data_with_momentum_list = []
for match_name, match_data in match_dict:

    df = match_data

    # 将"point_victor"的[1, 2]映射为[0, 1]
    df = point_victor_mapping(df)

    df = df.reset_index()

    p1_ace_df = need_to_mark_in_plot(df, "p1_ace")
    p2_ace_df = need_to_mark_in_plot(df, "p2_ace")
    p1_net_pt_won = need_to_mark_in_plot(df, "p1_net_pt_won")
    p2_net_pt_won = need_to_mark_in_plot(df, "p2_net_pt_won")
    p1_break_pt_won = need_to_mark_in_plot(df, "p1_break_pt_won")
    p2_break_pt_won = need_to_mark_in_plot(df, "p2_break_pt_won")

    # draw_play_flow(
    #     df,
    #     p1_name,
    #     p2_name,
    #     p1_ace_df,
    #     p2_ace_df,
    #     p1_net_pt_won,
    #     p2_net_pt_won,
    #     p1_break_pt_won,
    #     p1_break_pt_won
    # )

    # 删除全为0的列，因为无法计算相关性 ("p1_double_fault")
    df_1 = df.loc[:, (df != 0).any(axis=0)]

    all_df = df_1

    new_col = list(set(get_momentum_col("p1")) - (set(match_data.columns.values) - set(df_1.columns.values)))

    # df = df.loc[:, get_pca_col()]

    df = df_1.loc[:, new_col]

    df = pd.DataFrame(data=get_standardized_data(df), columns=new_col)

    df, new_info = pca(df)
    total_info.update(new_info)

    df = pd.DataFrame(df)

    p1_state_impacts, p1_neg_log_likelihood, pred, p1_start_prob, p1_transition_prob, p1_emission_prob = train_and_predict_hidden_markov_model(copy.deepcopy(df))

    all_df["p1_momentum"] = pd.Series(pred)
    print(all_df["p1_momentum"])
    match_data["p1_momentum"] = all_df["p1_momentum"]






    new_col = list(set(get_momentum_col("p2")) - (set(match_data.columns.values) - set(df_1.columns.values)))

    df = all_df.loc[:, new_col]

    df = pd.DataFrame(data=get_standardized_data(df), columns=new_col)

    df, new_info = pca(df)
    total_info.update(new_info)

    df = pd.DataFrame(df)

    p2_state_impacts, p2_neg_log_likelihood, pred, p2_start_prob, p2_transition_prob, p2_emission_prob = train_and_predict_hidden_markov_model(copy.deepcopy(df))

    all_df["p2_momentum"] = pd.Series(pred)
    print(all_df["p2_momentum"])
    match_data["p2_momentum"] = all_df["p2_momentum"]



    # for index, col_name in zip(del_col_index_list, del_col_list):
    #     all_df.insert(loc=index, column=col_name, value=match_data.loc[:, col_name])


    # state_impact 的值大，意味着对观测序列的期望影响程度就大
    p1_importance_dict = dict(zip(np.argsort(p1_state_impacts), [x for x in range(len(p1_state_impacts))]))
    p2_importance_dict = dict(zip(np.argsort(p2_state_impacts), [x for x in range(len(p2_state_impacts))]))

    # momentum越大，影响程度越大（因为p1和p2都是相同的计算方法，所以可以当做两人的影响都是同方向）
    all_df["p1_momentum"] = all_df["p1_momentum"].map(p1_importance_dict)
    all_df["p2_momentum"] = all_df["p2_momentum"].map(p2_importance_dict)

    # all_df.to_csv("./data/after_momentum_all.csv", index=False)

    # df_with_momentum = pd.read_csv("./data/after_momentum_all.csv")

    df_with_momentum = all_df

    df_with_momentum["p1_momentum_value"] = poly_fit([x for x in range(len(df_with_momentum))], df_with_momentum[["p1_momentum"]].values.reshape(-1).tolist())
    df_with_momentum["p2_momentum_value"] = poly_fit([x for x in range(len(df_with_momentum))], df_with_momentum[["p2_momentum"]].values.reshape(-1).tolist())
    match_data["p1_momentum_value"] = poly_fit([x for x in range(len(df_with_momentum))], df_with_momentum[["p1_momentum"]].values.reshape(-1).tolist())
    match_data["p2_momentum_value"] = poly_fit([x for x in range(len(df_with_momentum))], df_with_momentum[["p2_momentum"]].values.reshape(-1).tolist())

    match_data["p1_momentum_value_better"] = match_data["p1_momentum_value"] - match_data["p2_momentum_value"]


    data_with_momentum_list.append(match_data)

total_df = data_with_momentum_list[0]
for cur_df in data_with_momentum_list:
    total_df = pd.concat([total_df, cur_df])

total_df.to_csv("./data/after_momentum_value_all.csv", index=False)











# # 将"point_victor"的[1, 2]映射为[0, 1]
# df = point_victor_mapping(df)
#
# df = df.reset_index()
#
# p1_ace_df = need_to_mark_in_plot(df, "p1_ace")
# p2_ace_df = need_to_mark_in_plot(df, "p2_ace")
# p1_net_pt_won = need_to_mark_in_plot(df, "p1_net_pt_won")
# p2_net_pt_won = need_to_mark_in_plot(df, "p2_net_pt_won")
# p1_break_pt_won = need_to_mark_in_plot(df, "p1_break_pt_won")
# p2_break_pt_won = need_to_mark_in_plot(df, "p2_break_pt_won")
#
# # draw_play_flow(
# #     df,
# #     p1_name,
# #     p2_name,
# #     p1_ace_df,
# #     p2_ace_df,
# #     p1_net_pt_won,
# #     p2_net_pt_won,
# #     p1_break_pt_won,
# #     p1_break_pt_won
# # )
#
# all_df = df
#
# # 删除全为0的列，因为无法计算相关性 ("p1_double_fault")
# df_1 = df.loc[:, (df != 0).any(axis=0)]
#
# # df = df.loc[:, get_pca_col()]
# df = df_1.loc[:, get_momentum_col("p1")]
#
# df = pd.DataFrame(data=get_standardized_data(df), columns=get_momentum_col("p1"))
#
# df, new_info = pca(df)
# total_info.update(new_info)
#
# df = pd.DataFrame(df)
#
# p1_state_impacts, p1_neg_log_likelihood, pred, p1_start_prob, p1_transition_prob, p1_emission_prob = train_and_predict_hidden_markov_model(copy.deepcopy(df))
#
# all_df["p1_momentum"] = pd.DataFrame(pred)
#
#
#
# df = df_1.loc[:, get_momentum_col("p2")]
#
# df = pd.DataFrame(data=get_standardized_data(df), columns=get_momentum_col("p2"))
#
# df, new_info = pca(df)
# total_info.update(new_info)
#
# df = pd.DataFrame(df)
#
# p2_state_impacts, p2_neg_log_likelihood, pred, p2_start_prob, p2_transition_prob, p2_emission_prob = train_and_predict_hidden_markov_model(copy.deepcopy(df))
#
# all_df["p2_momentum"] = pd.DataFrame(pred)
#
# # state_impact 的值大，意味着对观测序列的期望影响程度就大
# p1_importance_dict = dict(zip(np.argsort(p1_state_impacts), [x for x in range(len(p1_state_impacts))]))
# p2_importance_dict = dict(zip(np.argsort(p2_state_impacts), [x for x in range(len(p2_state_impacts))]))
#
# # momentum越大，影响程度越大（因为p1和p2都是相同的计算方法，所以可以当做两人的影响都是同方向）
# all_df["p1_momentum"] = all_df["p1_momentum"].map(p1_importance_dict)
# all_df["p2_momentum"] = all_df["p2_momentum"].map(p2_importance_dict)
#
# all_df.to_csv("./data/after_momentum_all.csv", index=False)
#
# df_with_momentum = pd.read_csv("./data/after_momentum_all.csv")
#
# df_with_momentum["p1_momentum_value"] = poly_fit([x for x in range(len(df_with_momentum))], df_with_momentum[["p1_momentum"]].values.reshape(-1).tolist())
# df_with_momentum["p2_momentum_value"] = poly_fit([x for x in range(len(df_with_momentum))], df_with_momentum[["p2_momentum"]].values.reshape(-1).tolist())
#
# df_with_momentum["p1_momentum_value_better"] = \
#     df_with_momentum["p1_momentum_value"] - df_with_momentum["p2_momentum_value"]
#
# df_with_momentum.to_csv("./data/after_momentum_value_all.csv", index=False)




# draw_momentum(df_with_momentum, p1_name, p2_name)






# t1 = pd.read_csv("../llh/data/transition_prob_de.csv")
#
# t2 = get_state_distribution(t1)
# t2 = t2.tolist()
#
# t3 = pd.read_csv("../llh/data/emission_prob_de.csv")
# a = 0.7
# b = 0.3
# importance = {}
# for i in range(len(t3)):
#     importance[i+1] = a * t3.iloc[i, 0] + b * t3.iloc[i, 2] - a * t3.iloc[i, 1] - b * t3.iloc[i, 3]
#
# # [1,4,3,5,2] 正向->负向
#
# importance = {
#     1: 1,
#     4: 2,
#     3: 3,
#     5: 4,
#     2: 5
# }
#
# t4 = pd.read_csv("../llh/data/match_momentum.csv")
#
# t4 = t4.iloc[:, 1:]
#
# t4["p1_momentum"] = t4["p1_momentum"].map(importance)
# t4["p2_momentum"] = t4["p2_momentum"].map(importance)
#
# t4.fillna(0.0, inplace=True)

# t4["p1_momentum_value"] = pd.DataFrame(gaussian_mix(t4["p1_momentum"].to_numpy()))
# t4["p2_momentum_value"] = pd.DataFrame(gaussian_mix(t4["p2_momentum"].to_numpy()))




# t4 = t4.loc[:, ["point_victor", "p1_momentum", "p2_momentum"]]
#
# t4_1 = pd.concat([t4["p1_momentum"], t4.drop("p1_momentum", axis=1)], axis=1)
#
# # t4_1.drop("p2_momentum", axis=1, inplace=True)
#
# t4_1.fillna(0.0, inplace=True)
#
# t4_1 = get_standardized_data(t4_1)
#
# array = t4_1
#
# # segment the whole dataset into total train dataset and test dataset
# x_train_and_validate, x_test, y_train_and_validate, y_test = split_dataset(array)
#
# # segment train dataset into train datasets and validate datasets
# # train_and_validate_data_list = k_fold_cross_validation_data_segmentation(x_train_and_validate, y_train_and_validate)
#
# # 0202:
# # fuzzy_comprehensive_evaluation_model()
#
# learning_curve_values_dict = {}
# f1_score_dict = {}
# roc_auc_dict = {}
#
# # y_pred, new_info = momentum_polynomial_regression(array[:, 1:], array[:, :1], x_train_and_validate, y_train_and_validate, x_test, y_test, None, "grid_search")
#
# # Logistic regression
# name = "logistic regression"
# y_pred, new_info, train_sizes, train_scores_mean, train_scores_std, test_scores_mean, test_scores_std, f1_score, fpr, tpr, thresholds = \
#     momentum_logistic_regression(array[:, 1:], array[:, :1], x_train_and_validate, y_train_and_validate, x_test, y_test, None, "grid_search")
# total_info.update(new_info)
# learning_curve_values_dict[name] = [
#     train_sizes, train_scores_mean, train_scores_std, test_scores_mean, test_scores_std
# ]
# f1_score_dict[name] = f1_score
# roc_auc_dict[name] = [
#     fpr, tpr, thresholds
# ]
#
# t4 = pd.concat([t4, pd.DataFrame(y_pred)], axis=1)
#
#
#
#
# t4_2 = pd.concat([t4["p2_momentum"], t4.drop("p2_momentum", axis=1)], axis=1)
#
# # t4_2.drop("p1_momentum", axis=1, inplace=True)
#
# t4_2.fillna(0.0, inplace=True)
#
# t4_2 = get_standardized_data(t4_2)
#
# array = t4_2
#
# # segment the whole dataset into total train dataset and test dataset
# x_train_and_validate, x_test, y_train_and_validate, y_test = split_dataset(array)
#
# # segment train dataset into train datasets and validate datasets
# # train_and_validate_data_list = k_fold_cross_validation_data_segmentation(x_train_and_validate, y_train_and_validate)
#
# # 0202:
# # fuzzy_comprehensive_evaluation_model()
#
# learning_curve_values_dict = {}
# f1_score_dict = {}
# roc_auc_dict = {}
#
# # y_pred, new_info = momentum_polynomial_regression(array[:, 1:], array[:, :1], x_train_and_validate, y_train_and_validate, x_test, y_test, None, "grid_search")
#
# # Logistic regression
# name = "logistic regression"
# y_pred, new_info, train_sizes, train_scores_mean, train_scores_std, test_scores_mean, test_scores_std, f1_score, fpr, tpr, thresholds = \
#     momentum_logistic_regression(array[:, 1:], array[:, :1], x_train_and_validate, y_train_and_validate, x_test, y_test, None, "grid_search")
# total_info.update(new_info)
# learning_curve_values_dict[name] = [
#     train_sizes, train_scores_mean, train_scores_std, test_scores_mean, test_scores_std
# ]
# f1_score_dict[name] = f1_score
# roc_auc_dict[name] = [
#     fpr, tpr, thresholds
# ]
#
# t4 = pd.concat([t4, pd.DataFrame(y_pred)], axis=1)
#
#
# # t4.to_csv("./data/t4.csv", index=False)
#
# # 0204:
# array = df.values
#
# # segment the whole dataset into total train dataset and test dataset
x_train_and_validate, x_test, y_train_and_validate, y_test = split_dataset(array)
#
# # segment train dataset into train datasets and validate datasets
# # train_and_validate_data_list = k_fold_cross_validation_data_segmentation(x_train_and_validate, y_train_and_validate)
#
# # 0202:
# # fuzzy_comprehensive_evaluation_model()
#
# learning_curve_values_dict = {}
# f1_score_dict = {}
# roc_auc_dict = {}
#
# # Logistic regression
# name = "logistic regression"
# new_info, train_sizes, train_scores_mean, train_scores_std, test_scores_mean, test_scores_std, f1_score, fpr, tpr, thresholds = \
#     logistic_regression(array[:, 1:], array[:, :1], x_train_and_validate, y_train_and_validate, x_test, y_test, None, "grid_search")
# total_info.update(new_info)
# learning_curve_values_dict[name] = [
#     train_sizes, train_scores_mean, train_scores_std, test_scores_mean, test_scores_std
# ]
# f1_score_dict[name] = f1_score
# roc_auc_dict[name] = [
#     fpr, tpr, thresholds
# ]
#
#
# # Random Forest classifier
# name = "random forest"
# new_info, train_sizes, train_scores_mean, train_scores_std, test_scores_mean, test_scores_std, f1_score, fpr, tpr, thresholds = \
#     random_forest_classifier(array[:, 1:], array[:, :1], x_train_and_validate, y_train_and_validate, x_test, y_test, None, "grid_search")
# total_info.update(new_info)
# learning_curve_values_dict[name] = [
#     train_sizes, train_scores_mean, train_scores_std, test_scores_mean, test_scores_std
# ]
# f1_score_dict[name] = f1_score
# roc_auc_dict[name] = [
#     fpr, tpr, thresholds
# ]
#
# # xgboost classifier
# name = "xgboost"
# new_info, train_sizes, train_scores_mean, train_scores_std, test_scores_mean, test_scores_std, f1_score, fpr, tpr, thresholds = \
#     xgboost_classifier(array[:, 1:], array[:, :1], x_train_and_validate, y_train_and_validate, x_test, y_test, None, "grid_search")
# total_info.update(new_info)
# learning_curve_values_dict[name] = [
#     train_sizes, train_scores_mean, train_scores_std, test_scores_mean, test_scores_std
# ]
# f1_score_dict[name] = f1_score
# roc_auc_dict[name] = [
#     fpr, tpr, thresholds
# ]
#
# draw_learning_curve_total(learning_curve_values_dict, "train")
# draw_learning_curve_total(learning_curve_values_dict, "validation")
#
# draw_roc_auc_curve_total(roc_auc_dict, "train")
#
#
#
#
# print("f1_score_dict"+str(f1_score_dict))
#
# Save the info of predictive analysis to a json format file
with open("../data/total_info.json", "w", encoding="utf-8") as f:
    json.dump(total_info, f, indent=4, ensure_ascii=False)
















