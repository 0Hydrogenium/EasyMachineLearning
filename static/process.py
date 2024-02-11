import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
import copy
import pandas as pd
from scipy.stats import spearmanr

from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from sklearn.datasets import load_breast_cancer
from scipy.linalg import eig

from coding.llh.static.config import Config


def match_split(df: pd.DataFrame):
    return df.groupby("match_id")


# 斯皮尔曼秩相关系数
def calculate_spearmanr(x, y):
    rho, p_value = spearmanr(x, y)

    return rho, p_value


def calculate_remain_positive_points(df: pd.DataFrame):
    # remain_positive距离无限远设置为len(df)

    df["p1_remain_positive"] = 0
    df["p2_remain_positive"] = 0
    p1_zero_distance_list = []
    p2_zero_distance_list = []

    for i in range(1, len(df)):
        if (df.loc[i, "p1_momentum_value_better"] > 0
            and i != 0):
            p1_zero_distance_list.append(i)
        elif (df.loc[i, "p1_momentum_value_better"] < 0
            and i != 0):
            p2_zero_distance_list.append(i)

    for j in range(len(df)):
        for x in p1_zero_distance_list:
            if j <= x:
                df.loc[j, "p1_remain_positive"] = x - j
                break
        else:
            continue

    for j in range(len(df)):
        for x in p2_zero_distance_list:
            if j <= x:
                df.loc[j, "p2_remain_positive"] = x - j
                break
        else:
            continue

    return df


def calculate_swing_point(df:pd.DataFrame):
    # swing距离无限远设置为len(df)

    df["swing"] = 0
    zero_distance_list = []

    for i in range(1, len(df)):
        if (df.loc[i, "p1_momentum_value_better"] > 0 and df.loc[i-1, "p1_momentum_value_better"] < 0
            and i != 0) or (df.loc[i, "p1_momentum_value_better"] < 0 and df.loc[i - 1, "p1_momentum_value_better"] > 0
             and i != 0):
            zero_distance_list.append(i)

    for j in range(len(df)):
        for x in zero_distance_list:
            if j <= x:
                df.loc[j, "swing"] = x - j
                break
        else:
            continue

    return df


def replace_na_to_label(df: pd.DataFrame):
    return df.fillna("Not A Number")


def get_state_distribution(data):
    # get the matrix of correlation coefficients
    covX = np.around(np.corrcoef(data.T), decimals=3)

    # draw_heat_map(covX, "related", False)

    # Solve the eigenvalues and eigenvectors of the coefficient correlation matrix
    eigenvalues, eigenvectors = np.linalg.eig(covX.T)

    eigenvalues = np.around(eigenvalues, decimals=3)

    eigenvalues_dict = dict(zip(eigenvalues.tolist(), list(range(0, len(eigenvalues)))))

    # Sort feature values in descending order
    eigenvalues = sorted(eigenvalues, reverse=True)

    for i, value in enumerate(eigenvalues):
        if i == 0:
            sorted_eigenvectors = eigenvectors[:, eigenvalues_dict[value]].reshape(-1, 1)
        else:
            sorted_eigenvectors = np.concatenate((sorted_eigenvectors, eigenvectors[:, eigenvalues_dict[value]].reshape(-1, 1)), axis=1)

    # draw_line_graph(range(1, len(eigenvalues) + 1), eigenvalues, "Eigenvalue")

    # get the contribution of the eigenvalues
    contribution = eigenvalues / np.sum(eigenvalues)

    return contribution


# 指数加权平均
def exponential_moving_average(df):
    alpha = 0.3

    ema = [df[0]]

    for i in range(1, len(df)):
        ema_value = alpha * df[i] + (1 - alpha) * ema[i-1]
        ema.append(ema_value)

    return ema


def need_to_mark_in_plot(df, col_name):
    return df.where(df[col_name] == 1).dropna()


def point_victor_mapping(df):
    mapping = {
        1: 0.0,
        2: 1.0
    }
    df["point_victor"] = df["point_victor"].map(mapping)

    return df


def pick_matches_with_name(df, name):
    df = df.where(df["match_id"] == name).dropna()

    p1_name = df["player1"].iloc[0]
    p2_name = df["player2"].iloc[0]

    return df, p1_name, p2_name


def pick_matches_with_longest(df):
    target_match_id = df.groupby("match_id").size().idxmax()

    df = df.where(df["match_id"] == target_match_id).dropna()

    p1_name = df["player1"].iloc[0]
    p2_name = df["player2"].iloc[0]

    return df, p1_name, p2_name


def choose_y_col_in_dataframe(df: pd.DataFrame, y_col: str):
    y_data = df[y_col]
    df.drop(y_col, axis=1, inplace=True)
    df.insert(0, y_col, y_data)

    return df


def load_data(sort):
    if sort == "Iris Dataset":
        sk_data = load_iris()
    elif sort == "Wine Dataset":
        sk_data = load_wine()
    elif sort == "Breast Cancer Dataset":
        sk_data = load_breast_cancer()
    else:
        # 默认数据源
        sk_data = load_iris()

    df = pd.DataFrame(data=sk_data.data, columns=sk_data.feature_names)

    return df


def load_custom_data(file):
    return pd.read_csv(file)


def preprocess_raw_data_filtering(df):
    info = {}

    len_0 = len(df)
    info["Total size of raw data"] = len_0

    # Delete the column "CUSTOMER_ID"
    # df.drop("CUSTOMER_ID", axis=1, inplace=True)

    # Remove duplicate data
    df.drop_duplicates()
    len_1 = len_0 - len(df)
    info["Number of duplicates in the raw data"] = len_1

    # Remove "nan" data
    # df = remove_nan_from_data(df)
    # len_2 = len_0 - len_1 - len(df)
    # info["Number of nan in the raw data"] = len_2

    info["Total size of filtered data after data preprocessing"] = len(df)

    # Save the cleaned data to a csv format file
    # df.to_csv("../data/filtered_data.csv", index=False)

    return df, info


def remove_nan_from_data(df):
    # Remove "nan" data
    df.dropna(inplace=True)

    return df


# Get standardized data
def get_standardized_data(df):
    array = np.concatenate(((df.iloc[:, :1]).values, preprocessing.scale(df.iloc[:, 1:])), axis=1)

    return array


def split_dataset(array):
    x_train_and_validate, x_test, y_train_and_validate, y_test = train_test_split(
        array[:, 1:],
        array[:, :1],
        random_state=Config.RANDOM_STATE,
        train_size=0.8
    )

    return x_train_and_validate, x_test, y_train_and_validate, y_test


def k_fold_cross_validation_data_segmentation(x_train, y_train):
    k = 5

    train_data_array = np.concatenate((y_train, x_train), axis=1)

    k_fold = KFold(n_splits=k, shuffle=True, random_state=Config.RANDOM_STATE)

    train_data_list = []
    validate_data_list = []
    for train_index, validate_index in k_fold.split(train_data_array):
        train_data_list.append(train_data_array[train_index])
        validate_data_list.append(train_data_array[validate_index])

    train_and_validate_data_list = []

    for i in range(k):
        train_and_validate_data_list.append((
            train_data_list[i][:, 1:],
            validate_data_list[i][:, 1:],
            train_data_list[i][:, 0],
            validate_data_list[i][:, 0]
        ))

    return train_and_validate_data_list


def grid_search(params, model, x_train, y_train, scoring=None):
    info = {}

    if scoring == "neg_mean_squared_error":
        grid_search_model = GridSearchCV(model, params, cv=5, scoring="neg_mean_squared_error")
    else:
        grid_search_model = GridSearchCV(model, params, cv=5)

    grid_search_model.fit(x_train, y_train)

    info["Optimal hyperparameters"] = grid_search_model.best_params_

    best_model = grid_search_model.best_estimator_

    return best_model


def bayes_search(params, model, x_train, y_train, scoring=None):
    info = {}

    if scoring == "neg_mean_squared_error":
        bayes_search_model = BayesSearchCV(model, params, cv=5, n_iter=50, scoring="neg_mean_squared_error")
    else:
        bayes_search_model = BayesSearchCV(model, params, cv=5, n_iter=50)

    bayes_search_model.fit(x_train, y_train)

    info["Optimal hyperparameters"] = bayes_search_model.best_params_

    best_model = bayes_search_model.best_estimator_

    return best_model


