import copy

import numpy as np
import pandas as pd
import skfuzzy.control.fuzzyvariable
from sklearn.mixture import GaussianMixture
from sklearn import preprocessing

from coding.llh.analysis.evaluation_model import *
from coding.llh.analysis.gaussian_model import gaussian_mix
from coding.llh.analysis.markov_model import train_and_predict_hidden_markov_model
from coding.llh.analysis.poly_model import poly_fit
from coding.llh.static.col import *
from coding.llh.visualization.draw_learning_curve_total import draw_learning_curve_total
from coding.llh.visualization.draw_momentum import draw_momentum
from coding.llh.visualization.draw_parallel_coordinates import draw_parallel_coordinates
from coding.llh.visualization.draw_play_flow import draw_play_flow
from coding.llh.visualization.draw_pred_total import draw_pred_total
from coding.llh.visualization.draw_roc_auc_curve_total import draw_roc_auc_curve_total
from coding.llh.visualization.draw_swings_and_positives import draw_swings_and_positives
from static.config import *
from static.process import *
from analysis.descriptive_analysis import *
from analysis.exploratory_analysis import *
from analysis.linear_model import *
from analysis.tree_model import *
from analysis.kernel_model import *
from analysis.bayes_model import *
from analysis.neural_model import *
from analysis.gradient_model import *


pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)

total_info = {}



df_with_momentum = pd.read_csv("../llh/data/after_momentum_value.csv")

p1_name = "Alejandro Davidovich Fokina"
p2_name = "Holger Rune"

# 计算转折点
df_with_momentum = calculate_swing_point(copy.deepcopy(df_with_momentum))

# 计算连胜点
# df_with_momentum = calculate_remain_positive_points(copy.deepcopy(df_with_momentum))

# draw_swings_and_positives(df_with_momentum, p1_name, p2_name)



# df_with_momentum = pd.concat([df_with_momentum["swing"], df_with_momentum.drop("swing", axis=1)], axis=1)
df_with_momentum = pd.concat([df_with_momentum["p1_momentum_value_better"], df_with_momentum.drop("p1_momentum_value_better", axis=1)], axis=1)

df_with_momentum.drop([
    "p1_game_victor",
    "p2_game_victor",
    "p1_set_victor",
    "p2_set_victor",
    "p1_momentum_value",
    "p2_momentum_value"
], axis=1, inplace=True)

# 删除全为0的列，因为无法计算相关性 ("p1_double_fault")
# df_1 = df_with_momentum.loc[:, (df_with_momentum != 0).any(axis=0)]

df_with_momentum = df_with_momentum.loc[:, get_pca_col()]


feature_names = df_with_momentum.columns.values
# df = df_1.loc[:, get_momentum_col("p1")]

df_with_momentum = pd.DataFrame(data=get_standardized_data(df_with_momentum), columns=get_pca_col())

# df_with_momentum, new_info = pca(df_with_momentum)
# total_info.update(new_info)

# df_with_momentum = pd.DataFrame(df_with_momentum)

# df_with_momentum = df_with_momentum.iloc[:600]

array = df_with_momentum.values

# segment the whole dataset into total train dataset and test dataset
x_train_and_validate, x_test, y_train_and_validate, y_test = split_dataset(array)

# segment train dataset into train datasets and validate datasets
# train_and_validate_data_list = k_fold_cross_validation_data_segmentation(x_train_and_validate, y_train_and_validate)

# 0202:
# fuzzy_comprehensive_evaluation_model()

learning_curve_values_dict = {}
pred_dict = {}

# name = "Support Vector Regression"
# y_pred, new_info, train_sizes, train_scores_mean, train_scores_std, test_scores_mean, test_scores_std = \
#     svm_regression(feature_names, preprocessing.scale(array[:, 1:]), array[:, :1], preprocessing.scale(x_train_and_validate), y_train_and_validate, preprocessing.scale(x_test), y_test, None)
# total_info.update(new_info)
# learning_curve_values_dict[name] = [
#     train_sizes, train_scores_mean, train_scores_std, test_scores_mean, test_scores_std
# ]
# pred_dict[name] = [y_pred, y_test]
# #
# name = "Random Forest Regression"
# y_pred, new_info, train_sizes, train_scores_mean, train_scores_std, test_scores_mean, test_scores_std = \
#     random_forest_regression(feature_names, preprocessing.scale(array[:, 1:]), array[:, :1], preprocessing.scale(x_train_and_validate), y_train_and_validate, preprocessing.scale(x_test), y_test, None)
# total_info.update(new_info)
# learning_curve_values_dict[name] = [
#     train_sizes, train_scores_mean, train_scores_std, test_scores_mean, test_scores_std
# ]
pred_dict[name] = [y_pred, y_test]

name = "Double Exponential Smoothing Plus"
y_pred, new_info, train_sizes, train_scores_mean, train_scores_std, test_scores_mean, test_scores_std = \
    gradient_boosting_regression(feature_names, preprocessing.scale(array[:, 1:]), array[:, :1], preprocessing.scale(x_train_and_validate), y_train_and_validate, preprocessing.scale(x_test), y_test, None)
total_info.update(new_info)
learning_curve_values_dict[name] = [
    train_sizes, train_scores_mean, train_scores_std, test_scores_mean, test_scores_std
]
pred_dict[name] = [y_pred, y_test]

# name = "mlp"
# new_info, train_sizes, train_scores_mean, train_scores_std, test_scores_mean, test_scores_std = \
#     mlp_regression(feature_names, preprocessing.scale(array[:, 1:]), array[:, :1], preprocessing.scale(x_train_and_validate), y_train_and_validate, preprocessing.scale(x_test), y_test, "grid_search")
# total_info.update(new_info)
# learning_curve_values_dict[name] = [
#     train_sizes, train_scores_mean, train_scores_std, test_scores_mean, test_scores_std
# ]

# draw_learning_curve_total(learning_curve_values_dict, "train")
# draw_learning_curve_total(learning_curve_values_dict, "validation")

# draw_pred_total(pred_dict)

# Save the info of predictive analysis to a json format file
with open("./data/total_info.json", "w", encoding="utf-8") as f:
    json.dump(total_info, f, indent=4, ensure_ascii=False)
