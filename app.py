import warnings
import os
import random
import math
import matplotlib.pyplot as plt
import gradio as gr
import pandas as pd
import copy
import numpy as np
import sklearn.preprocessing as preprocessing
from sklearn.model_selection import train_test_split
from sklearn.datasets import *

from analysis.model_train.bayes_model import NaiveBayesClassifierParams, naive_bayes_classifier
from analysis.model_train.distance_model import KNNClassifierParams, KNNRegressionParams, knn_classifier, knn_regressor
from analysis.model_train.gradient_model import GradientBoostingParams, gradient_boosting_regressor
from analysis.model_train.kernel_model import SVMClassifierParams, SVMRegressionParams, svm_classifier, svm_regressor
from analysis.model_train.linear_model import LinearRegressionParams, LogisticRegressionParams, \
    PolynomialRegressionParams, linear_regressor, polynomial_regressor, logistic_classifier
from analysis.model_train.tree_model import DecisionTreeClassifierParams, RandomForestClassifierParams, \
    RandomForestRegressionParams, XgboostClassifierParams, LightGBMClassifierParams, decision_tree_classifier, \
    random_forest_classifier, random_forest_regressor, xgboost_classifier, lightGBM_classifier
from analysis.others.shap_model import draw_dependence, draw_force, draw_waterfall, draw_shap_beeswarm
from classes.static_custom_class import *
from metrics.calculate_classification_metrics import ClassificationMetrics
from metrics.calculate_regression_metrics import RegressionMetrics
from visualization.draw_boxplot import draw_boxplot
from visualization.draw_data_fit_total import draw_data_fit_total
from visualization.draw_heat_map import draw_heat_map
from visualization.draw_histogram import draw_histogram
from visualization.draw_learning_curve_total import draw_learning_curve_total

# from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings("ignore")


# thread_pool_executor = ThreadPoolExecutor(1024)


def get_container_dict():
    model_name_list = [
        # [模型]
        MN.linear_regressor,
        MN.polynomial_regressor,
        MN.logistic_classifier,
        MN.decision_tree_classifier,
        MN.random_forest_classifier,
        MN.random_forest_regressor,
        MN.xgboost_classifier,
        # MN.lightGBM_classifier,
        MN.gradient_boosting_regressor,
        MN.svm_classifier,
        MN.svm_regressor,
        MN.knn_classifier,
        MN.knn_regressor,
        MN.naive_bayes_classifier,
        # 模型Step 10:在这里添加新的模型名称映射 (MN.模型名称: Container())
    ]

    return dict(zip(model_name_list, [Container()] * len(model_name_list)))


class PaintObject:
    def __init__(self):
        self.color_cur_num = 0
        self.color_cur_list = []
        self.label_cur_num = 0
        self.label_cur_list = []
        self.x_cur_label = ""
        self.y_cur_label = ""
        self.name = ""

    def get_color_cur_num(self):
        return self.color_cur_num

    def set_color_cur_num(self, color_cur_num):
        self.color_cur_num = color_cur_num

    def get_color_cur_list(self):
        return self.color_cur_list

    def set_color_cur_list(self, color_cur_list):
        self.color_cur_list = color_cur_list

    def get_label_cur_num(self):
        return self.label_cur_num

    def set_label_cur_num(self, label_cur_num):
        self.label_cur_num = label_cur_num

    def get_label_cur_list(self):
        return self.label_cur_list

    def set_label_cur_list(self, label_cur_list):
        self.label_cur_list = label_cur_list

    def get_x_cur_label(self):
        return self.x_cur_label

    def set_x_cur_label(self, x_cur_label):
        self.x_cur_label = x_cur_label

    def get_y_cur_label(self):
        return self.y_cur_label

    def set_y_cur_label(self, y_cur_label):
        self.y_cur_label = y_cur_label

    def get_name(self):
        return self.name

    def set_name(self, name):
        self.name = name


class Container:
    def __init__(self, x_train=None, y_train=None, x_test=None, y_test=None, hyper_params_optimize=None):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.hyper_params_optimize = hyper_params_optimize
        self.info = {"参数": {}, "指标": {}}
        self.y_pred = None
        self.train_sizes = None
        self.train_scores_mean = None
        self.train_scores_std = None
        self.test_scores_mean = None
        self.test_scores_std = None
        self.status = None
        self.model = None

    def get_info(self):
        return self.info

    def set_info(self, info: dict):
        self.info = info

    def set_y_pred(self, y_pred):
        self.y_pred = y_pred

    def get_data_fit_values(self):
        return [
            self.y_pred,
            self.y_test
        ]

    def get_learning_curve_values(self):
        return [
            self.train_sizes,
            self.train_scores_mean,
            self.train_scores_std,
            self.test_scores_mean,
            self.test_scores_std
        ]

    def set_learning_curve_values(self, train_sizes, train_scores_mean, train_scores_std, test_scores_mean,
                                  test_scores_std):
        self.train_sizes = train_sizes
        self.train_scores_mean = train_scores_mean
        self.train_scores_std = train_scores_std
        self.test_scores_mean = test_scores_mean
        self.test_scores_std = test_scores_std

    def get_status(self):
        return self.status

    def set_status(self, status: str):
        self.status = status

    def get_model(self):
        return self.model

    def set_model(self, model):
        self.model = model


class SelectModel:
    def __init__(self):
        self.models = None
        self.waterfall_number = None
        self.force_number = None
        self.beeswarm_plot_type = None
        self.dependence_col = None
        self.data_distribution_col = None
        self.data_distribution_is_rotate = None
        self.descriptive_indicators_col = None
        self.descriptive_indicators_is_rotate = None
        self.heatmap_col = None
        self.heatmap_is_rotate = None

    def get_heatmap_col(self):
        return self.heatmap_col

    def set_heatmap_col(self, heatmap_col):
        self.heatmap_col = heatmap_col

    def get_heatmap_is_rotate(self):
        return self.heatmap_is_rotate

    def set_heatmap_is_rotate(self, heatmap_is_rotate):
        self.heatmap_is_rotate = heatmap_is_rotate

    def get_models(self):
        return self.models

    def set_models(self, models):
        self.models = models

    def get_waterfall_number(self):
        return self.waterfall_number

    def set_waterfall_number(self, waterfall_number):
        self.waterfall_number = waterfall_number

    def get_force_number(self):
        return self.force_number

    def set_force_number(self, force_number):
        self.force_number = force_number

    def get_beeswarm_plot_type(self):
        return self.beeswarm_plot_type

    def set_beeswarm_plot_type(self, beeswarm_plot_type):
        self.beeswarm_plot_type = beeswarm_plot_type

    def get_dependence_col(self):
        return self.dependence_col

    def set_dependence_col(self, dependence_col):
        self.dependence_col = dependence_col

    def get_data_distribution_col(self):
        return self.data_distribution_col

    def set_data_distribution_col(self, data_distribution_col):
        self.data_distribution_col = data_distribution_col

    def get_data_distribution_is_rotate(self):
        return self.data_distribution_is_rotate

    def set_data_distribution_is_rotate(self, data_distribution_is_rotate):
        self.data_distribution_is_rotate = data_distribution_is_rotate

    def get_descriptive_indicators_is_rotate(self):
        return self.descriptive_indicators_is_rotate

    def set_descriptive_indicators_is_rotate(self, descriptive_indicators_is_rotate):
        self.descriptive_indicators_is_rotate = descriptive_indicators_is_rotate

    def get_descriptive_indicators_col(self):
        return self.descriptive_indicators_col

    def set_descriptive_indicators_col(self, descriptive_indicators_col):
        self.descriptive_indicators_col = descriptive_indicators_col


# [模型]
class ChooseModelMetrics:
    @classmethod
    def choose(cls, cur_model):
        if cur_model == MN.linear_regressor:
            return RegressionMetrics.get_metrics()
        elif cur_model == MN.polynomial_regressor:
            return RegressionMetrics.get_metrics()
        elif cur_model == MN.logistic_classifier:
            return ClassificationMetrics.get_metrics()
        elif cur_model == MN.decision_tree_classifier:
            return ClassificationMetrics.get_metrics()
        elif cur_model == MN.random_forest_classifier:
            return ClassificationMetrics.get_metrics()
        elif cur_model == MN.random_forest_regressor:
            return RegressionMetrics.get_metrics()
        elif cur_model == MN.xgboost_classifier:
            return ClassificationMetrics.get_metrics()
        elif cur_model == MN.lightGBM_classifier:
            return ClassificationMetrics.get_metrics()
        elif cur_model == MN.gradient_boosting_regressor:
            return RegressionMetrics.get_metrics()
        elif cur_model == MN.svm_classifier:
            return ClassificationMetrics.get_metrics()
        elif cur_model == MN.svm_regressor:
            return RegressionMetrics.get_metrics()
        elif cur_model == MN.knn_classifier:
            return ClassificationMetrics.get_metrics()
        elif cur_model == MN.knn_regressor:
            return RegressionMetrics.get_metrics()
        elif cur_model == MN.naive_bayes_classifier:
            return ClassificationMetrics.get_metrics()
        # 模型Step 12:在这里添加新的模型指标类 (分类指标 / 回归指标)


# [模型]
class ChooseModelParams:
    @classmethod
    def choose(cls, cur_model):
        if cur_model == MN.linear_regressor:
            return LinearRegressionParams.get_params(Dataset.linear_regression_model_type)
        elif cur_model == MN.polynomial_regressor:
            return PolynomialRegressionParams.get_params()
        elif cur_model == MN.logistic_classifier:
            return LogisticRegressionParams.get_params()
        elif cur_model == MN.decision_tree_classifier:
            return DecisionTreeClassifierParams.get_params()
        elif cur_model == MN.random_forest_classifier:
            return RandomForestClassifierParams.get_params()
        elif cur_model == MN.random_forest_regressor:
            return RandomForestRegressionParams.get_params()
        elif cur_model == MN.xgboost_classifier:
            return XgboostClassifierParams.get_params()
        elif cur_model == MN.lightGBM_classifier:
            return LightGBMClassifierParams.get_params()
        elif cur_model == MN.gradient_boosting_regressor:
            return GradientBoostingParams.get_params()
        elif cur_model == MN.svm_classifier:
            return SVMClassifierParams.get_params()
        elif cur_model == MN.svm_regressor:
            return SVMRegressionParams.get_params()
        elif cur_model == MN.knn_classifier:
            return KNNClassifierParams.get_params()
        elif cur_model == MN.knn_regressor:
            return KNNRegressionParams.get_params()
        elif cur_model == MN.naive_bayes_classifier:
            return NaiveBayesClassifierParams.get_params(Dataset.naive_bayes_classifier_model_type)
        # 模型Step 13:在这里添加新的模型超参数类 (如果该模型含有额外组件，需传入函数)

        return {}


class Dataset:
    file = ""
    data = pd.DataFrame()

    na_list = []
    non_numeric_list = []
    str2int_mappings = {}
    max_num = 0
    data_copy = pd.DataFrame()
    assign = ""
    cur_model = ""
    select_y_mark = False

    descriptive_indicators_df = pd.DataFrame()

    # [模型]
    linear_regression_model_type = ""
    naive_bayes_classifier_model_type = ""
    # 模型Step 14:在这里添加新的模型额外组件

    container_dict = get_container_dict()

    visualize = ""
    choose_optimize = ""

    @classmethod
    def check_model_optimize_radio(cls):
        if cls.choose_optimize != "None" and cls.choose_optimize:
            return True
        return False

    @classmethod
    def get_dependence_col(cls):
        return [x for x in cls.data.columns.values][1:]

    @classmethod
    def reset_containers(cls):
        cls.file = ""
        cls.data = pd.DataFrame()

        cls.na_list = []
        cls.non_numeric_list = []
        cls.str2int_mappings = {}
        cls.max_num = 0
        cls.data_copy = pd.DataFrame()
        cls.assign = ""
        cls.cur_model = ""
        cls.select_y_mark = False

        cls.descriptive_indicators_df = pd.DataFrame()

        # [模型]
        cls.linear_regression_model_type = ""
        cls.naive_bayes_classifier_model_type = ""
        # 模型Step 15:在这里添加新的模型额外组件

        cls.container_dict = get_container_dict()

    @classmethod
    def check_descriptive_indicators_df(cls):
        return True if not cls.descriptive_indicators_df.empty else False

    @classmethod
    def get_descriptive_indicators_df(cls):
        return cls.descriptive_indicators_df

    @classmethod
    def get_notes(cls):
        notes = ""
        with open("./data/notes.md", "r", encoding="utf-8") as f:
            notes = str(f.read())
        return notes

    @classmethod
    def get_dataset_list(cls):
        return ["自定义", "Iris Dataset", "Wine Dataset", "Breast Cancer Dataset", "Diabetes Dataset",
                "California Housing Dataset"]

    @classmethod
    def get_col_list(cls):
        return [x for x in cls.data.columns.values]

    @classmethod
    def get_na_list_str(cls) -> str:
        na_series = cls.data.isna().any(axis=0)
        na_list = []
        na_list_str = ""
        for i in range(len(na_series)):
            cur_value = na_series[i]
            cur_index = na_series.index[i]
            if cur_value:
                na_list_str += cur_index + ", "
                na_list.append(cur_index)

        na_list_str = na_list_str.rstrip(", ")

        cls.na_list = na_list

        if not na_list:
            return "无"

        return na_list_str

    @classmethod
    def get_total_col_num(cls) -> int:
        return len(cls.data.columns)

    @classmethod
    def get_total_row_num(cls) -> int:
        return len(cls.data)

    @classmethod
    def update(cls, file: str, data: pd.DataFrame):
        cls.file = file
        cls.data = data
        cls.max_num = len(data)
        cls.data_copy = data

    @classmethod
    def clear(cls):
        cls.file = ""
        cls.data = pd.DataFrame()

    @classmethod
    def get_display_dataset_file(cls):
        file_path = FilePath.excel_base.format(FilePath.display_dataset)

        return file_path

    @classmethod
    def check_display_dataset_file(cls):
        return os.path.exists(cls.get_display_dataset_file())

    @classmethod
    def after_get_display_dataset_file(cls):
        if not cls.data.empty:
            cls.data.to_excel(cls.get_display_dataset_file(), index=False)

        return cls.get_display_dataset_file() if cls.check_display_dataset_file() else None

    @classmethod
    def del_col(cls, col_list: list):
        for col in col_list:
            if col in cls.data.columns.values:
                cls.data.drop(col, axis=1, inplace=True)

    @classmethod
    def get_max_num(cls):
        return cls.max_num

    @classmethod
    def remain_row(cls, num):
        cls.data = cls.data_copy.iloc[:num, :]

    @classmethod
    def del_all_na_col(cls):
        for col in cls.na_list:
            if col in cls.data.columns.values:
                cls.data.drop(col, axis=1, inplace=True)

    @classmethod
    def get_duplicate_num(cls):
        data_copy = copy.deepcopy(cls.data)
        return len(cls.data) - len(data_copy.drop_duplicates())

    @classmethod
    def del_duplicate(cls):
        cls.data = cls.data.drop_duplicates().reset_index().drop("index", axis=1)

    @classmethod
    def encode_label(cls, col_list: list, extra_mark=False):
        data_copy = copy.deepcopy(cls.data)

        str2int_mappings = dict(zip(col_list, [{} for _ in range(len(col_list))]))

        for col in str2int_mappings.keys():
            keys = np.array(data_copy[col].drop_duplicates())
            values = [x for x in range(len(keys))]
            str2int_mappings[col] = dict(zip(keys, values))

        for col, mapping in str2int_mappings.items():
            series = data_copy[col]

            for k, v in mapping.items():
                series.replace(k, v, inplace=True)
            data_copy[col] = series

        for k, v in str2int_mappings.items():
            if np.nan in v.keys():
                v.update({"nan": v.pop(np.nan)})
                str2int_mappings[k] = v

        if extra_mark:
            return data_copy
        else:
            cls.data = data_copy
            cls.str2int_mappings = str2int_mappings

    @classmethod
    def get_str2int_mappings_df(cls):
        columns_list = ["列名", "字符型", "数值型"]
        str2int_mappings_df = pd.DataFrame(columns=columns_list)

        for k, v in cls.str2int_mappings.items():
            cur_df = pd.DataFrame(columns=columns_list)
            cur_df["列名"] = pd.DataFrame([k] * len(v.keys()))
            cur_df["字符型"] = pd.DataFrame([x for x in v.keys()])
            cur_df["数值型"] = pd.DataFrame([x for x in v.values()])

            str2int_mappings_df = pd.concat([str2int_mappings_df, cur_df], axis=0)

            blank_df = pd.DataFrame(columns=columns_list)
            blank_df.loc[0] = ["", "", ""]
            str2int_mappings_df = pd.concat([str2int_mappings_df, blank_df], axis=0)

        return str2int_mappings_df.iloc[:-1, :]

    @classmethod
    def get_non_numeric_list(cls):
        data_copy = copy.deepcopy(cls.data)
        data_copy = data_copy.astype(str)

        non_numeric_list = []
        for col in data_copy.columns.values:
            if pd.to_numeric(data_copy[col], errors="coerce").isnull().values.any():
                non_numeric_list.append(col)

        cls.non_numeric_list = non_numeric_list

        return non_numeric_list

    @classmethod
    def get_data_type(cls):
        columns_list = ["列名", "数据类型"]

        data_type_dict = {}

        for col in cls.data.columns.values:
            data_type_dict[col] = cls.data[col].dtype.name

        data_type_df = pd.DataFrame(columns=columns_list)
        data_type_df["列名"] = [x for x in data_type_dict.keys()]
        data_type_df["数据类型"] = [x for x in data_type_dict.values()]

        return data_type_df

    @classmethod
    def change_data_type_to_float(cls):
        data_copy = cls.data

        for i, col in enumerate(data_copy.columns.values):
            if i != 0:
                data_copy[col] = data_copy[col].astype(float)

        cls.data = data_copy

    @classmethod
    def get_non_standardized_data(cls):
        not_standardized_data_list = []

        for col in cls.data.columns.values:
            if cls.data[col].dtype.name in ["int64", "float64"]:
                if not np.array_equal(np.round(preprocessing.scale(cls.data[col]), decimals=2),
                                      np.round(cls.data[col].values.round(2), decimals=2)):
                    not_standardized_data_list.append(col)

        return not_standardized_data_list

    @classmethod
    def check_before_train(cls):
        if cls.assign == "" or not cls.select_y_mark:
            return False

        for i, col in enumerate(cls.data.columns.values):
            if i == 0:
                if not (all(isinstance(x, str) for x in cls.data.iloc[:, 0]) or all(
                        isinstance(x, float) for x in cls.data.iloc[:, 0])):
                    return False
            else:
                if cls.data[col].dtype.name != "float64":
                    return False

        return True

    @classmethod
    def standardize_data(cls, col_list: list):
        for col in col_list:
            cls.data[col] = preprocessing.scale(cls.data[col])

    @classmethod
    def select_as_y(cls, col: str):
        cls.data = pd.concat([cls.data[col], cls.data.drop(col, axis=1)], axis=1)
        cls.select_y_mark = True

    @classmethod
    def get_optimize_list(cls):
        return ["无", "网格搜索", "贝叶斯优化"]

    @classmethod
    def get_optimize_name_mapping(cls):
        return dict(zip(cls.get_optimize_list(), ["None", "grid_search", "bayes_search"]))

    @classmethod
    def get_linear_regression_model_list(cls):
        return ["线性回归", "Lasso回归", "Ridge回归", "弹性网络回归"]

    @classmethod
    def get_naive_bayes_classifier_model_list(cls):
        return ["多项式朴素贝叶斯分类", "高斯朴素贝叶斯分类", "补充朴素贝叶斯分类"]

    @classmethod
    def get_linear_regression_model_name_mapping(cls):
        return dict(zip(cls.get_linear_regression_model_list(), ["LinearRegression", "Lasso", "Ridge", "ElasticNet"]))

    @classmethod
    def get_naive_bayes_classifier_model_name_mapping(cls):
        return dict(zip(cls.get_naive_bayes_classifier_model_list(), ["MultinomialNB", "GaussianNB", "ComplementNB"]))

    @classmethod
    def train_model(cls, optimize, params_list, train_size, extra_components_list):
        # 清除超参数的空值文本框
        params_list_copy = []
        for param in params_list:
            if param:
                params_list_copy.append(param)
        params_list = params_list_copy

        # 获取超参数优化方法英文名
        optimize = cls.get_optimize_name_mapping()[optimize]

        data_copy = cls.data
        # 若为分类任务，再次对数据表第一列进行 数值型转字符型 操作，确保训练的成功进行
        if cls.assign == MN.classification:
            data_copy = cls.encode_label([cls.data.columns.values[0]], True)
        # 分割数据集为训练集和测试集
        x_train, x_test, y_train, y_test = train_test_split(
            data_copy.values[:, 1:],
            data_copy.values[:, :1],
            random_state=StaticValue.RANDOM_STATE,
            train_size=train_size
        )
        container = Container(x_train, y_train, x_test, y_test, optimize)

        # 各个模型的训练方法
        # [模型]
        if cls.cur_model == MN.linear_regressor:
            linear_regression_model_type = extra_components_list[0]
            cls.linear_regression_model_type = cls.get_linear_regression_model_name_mapping()[
                linear_regression_model_type]
            container = linear_regressor(container, params_list, cls.linear_regression_model_type)
        elif cls.cur_model == MN.polynomial_regressor:
            container = polynomial_regressor(container, params_list)
        elif cls.cur_model == MN.logistic_classifier:
            container = logistic_classifier(container, params_list)
        elif cls.cur_model == MN.decision_tree_classifier:
            container = decision_tree_classifier(container, params_list)
        elif cls.cur_model == MN.random_forest_classifier:
            container = random_forest_classifier(container, params_list)
        elif cls.cur_model == MN.random_forest_regressor:
            container = random_forest_regressor(container, params_list)
        elif cls.cur_model == MN.xgboost_classifier:
            container = xgboost_classifier(container, params_list)
        elif cls.cur_model == MN.lightGBM_classifier:
            container = lightGBM_classifier(container, params_list)
        elif cls.cur_model == MN.gradient_boosting_regressor:
            container = gradient_boosting_regressor(container, params_list)
        elif cls.cur_model == MN.svm_classifier:
            container = svm_classifier(container, params_list)
        elif cls.cur_model == MN.svm_regressor:
            container = svm_regressor(container, params_list)
        elif cls.cur_model == MN.knn_classifier:
            container = knn_classifier(container, params_list)
        elif cls.cur_model == MN.knn_regressor:
            container = knn_regressor(container, params_list)
        elif cls.cur_model == MN.naive_bayes_classifier:
            naive_bayes_classifier_model_type = extra_components_list[1]
            cls.naive_bayes_classifier_model_type = cls.get_naive_bayes_classifier_model_name_mapping()[
                naive_bayes_classifier_model_type]
            container = naive_bayes_classifier(container, params_list, cls.naive_bayes_classifier_model_type)
        # 模型Step 6:在这里添加新的模型训练方法
        # (若有额外组件，需要根据获取顺序写 extra_components_list 的下标)
        # (输入为Container(), 输出也为Container())
        # (在对应类型的文件目录下的模型.py内写模型训练函数+模型超参数存储类)

        cls.container_dict[cls.cur_model] = container

    @classmethod
    def get_model_container_status(cls):
        return True if cls.cur_model != "" and cls.container_dict[cls.cur_model].get_status() == "trained" else False

    @classmethod
    def get_model_label(cls):
        return str(cls.get_model_name_mapping()[cls.cur_model]) + "模型是否完成训练" if cls.cur_model != "" else ""

    @classmethod
    def check_select_model(cls):
        return True if cls.cur_model != "" and cls.check_before_train() else False

    @classmethod
    def get_model_name(cls):
        return [x for x in cls.container_dict.keys()]

    @classmethod
    # [模型]
    def get_model_chinese_name(cls):
        # 模型Step 11:在这里添加新的模型名称到列表
        return ["线性回归", "多项式回归", "逻辑斯谛分类", "决策树分类", "随机森林分类", "随机森林回归", "XGBoost分类",
                # "LightGBM分类",
                "梯度提升回归", "支持向量机分类", "支持向量机回归", "K-最近邻分类", "K-最近邻回归", "朴素贝叶斯分类"]

    @classmethod
    def get_model_name_mapping(cls):
        return dict(zip(cls.get_model_name(), cls.get_model_chinese_name()))

    @classmethod
    def get_model_name_mapping_reverse(cls):
        return dict(zip(cls.get_model_chinese_name(), cls.get_model_name()))

    @classmethod
    def get_trained_model_list(cls):
        trained_model_list = []

        for model_name, container in cls.container_dict.items():
            if container.get_status() == "trained":
                trained_model_list.append(cls.get_model_name_mapping()[model_name])

        return trained_model_list

    @classmethod
    def draw_plot(cls, select_model, color_list: list, label_list: list, name: str, x_label: str, y_label: str,
                  is_default: bool):
        # [绘图]
        if cls.visualize == MN.learning_curve:
            return cls.draw_learning_curve_plot(select_model, color_list, label_list, name, x_label, y_label,
                                                is_default)
        elif cls.visualize == MN.shap_beeswarm:
            return cls.draw_shap_beeswarm_plot(select_model, color_list, label_list, name, x_label, y_label, is_default)
        elif cls.visualize == MN.data_fit:
            return cls.draw_data_fit_plot(select_model, color_list, label_list, name, x_label, y_label, is_default)
        elif cls.visualize == MN.waterfall:
            return cls.draw_waterfall_plot(select_model, color_list, label_list, name, x_label, y_label, is_default)
        elif cls.visualize == MN.force:
            return cls.draw_force_plot(select_model, color_list, label_list, name, x_label, y_label, is_default)
        elif cls.visualize == MN.dependence:
            return cls.draw_dependence_plot(select_model, color_list, label_list, name, x_label, y_label, is_default)
        elif cls.visualize == MN.data_distribution:
            return cls.draw_data_distribution_plot(select_model, color_list, label_list, name, x_label, y_label,
                                                   is_default)
        elif cls.visualize == MN.descriptive_indicators:
            return cls.draw_descriptive_indicators_plot(select_model, color_list, label_list, name, x_label, y_label,
                                                        is_default)
        elif cls.visualize == MN.heatmap:
            return cls.draw_heatmap_plot(select_model, color_list, label_list, name, x_label, y_label, is_default)
        # 绘图Step 9:在这里添加新的绘图函数

    @classmethod
    def draw_heatmap_plot(cls, select_model, color_list: list, label_list: list, name: str, x_label: str, y_label: str,
                          is_default: bool):
        color_cur_list = [] if is_default else color_list
        x_cur_label = "Indicators" if is_default else x_label
        y_cur_label = "Value" if is_default else y_label
        cur_name = "" if is_default else name

        paint_object = PaintObject()
        paint_object.set_color_cur_list(color_cur_list)
        paint_object.set_x_cur_label(x_cur_label)
        paint_object.set_y_cur_label(y_cur_label)
        paint_object.set_name(cur_name)

        if cls.check_col_list(select_model.get_heatmap_col()):
            return cls.error_return_draw(paint_object)

        df = Dataset.data
        heatmap_col = select_model.get_heatmap_col()

        covX = np.around(np.corrcoef(df[heatmap_col].T), decimals=3)
        std_dev = np.sqrt(np.diag(covX))
        pearson_matrix = covX / np.outer(std_dev, std_dev)

        return draw_heat_map(pearson_matrix, heatmap_col, paint_object, select_model.get_heatmap_is_rotate())

    @classmethod
    def draw_descriptive_indicators_plot(cls, select_model, color_list: list, label_list: list, name: str, x_label: str,
                                         y_label: str, is_default: bool):
        color_cur_list = [StaticValue.COLORS[random.randint(0, 11)]] * 3 if is_default else color_list
        x_cur_label = "Indicators" if is_default else x_label
        y_cur_label = "Value" if is_default else y_label
        cur_name = "" if is_default else name

        paint_object = PaintObject()
        paint_object.set_color_cur_list(color_cur_list)
        paint_object.set_x_cur_label(x_cur_label)
        paint_object.set_y_cur_label(y_cur_label)
        paint_object.set_name(cur_name)

        if cls.check_col_list(select_model.get_descriptive_indicators_col()):
            return cls.error_return_draw(paint_object)

        df = Dataset.data
        descriptive_indicators_col = select_model.get_descriptive_indicators_col()

        descriptive_indicators_df = pd.DataFrame(
            index=list(descriptive_indicators_col),
            columns=[
                "Name",
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

        for col in descriptive_indicators_col:
            descriptive_indicators_df["Name"][col] = col
            descriptive_indicators_df["Min"][col] = df[col].min()
            descriptive_indicators_df["Max"][col] = df[col].max()
            descriptive_indicators_df["Avg"][col] = df[col].mean()
            descriptive_indicators_df["Standard Deviation"][col] = df[col].std()
            descriptive_indicators_df["Standard Error"][col] = descriptive_indicators_df["Standard Deviation"][
                                                                   col] / math.sqrt(len(df[col]))
            descriptive_indicators_df["Upper Quartile"][col] = df[col].quantile(0.75)
            descriptive_indicators_df["Median"][col] = df[col].quantile(0.5)
            descriptive_indicators_df["Lower Quartile"][col] = df[col].quantile(0.25)
            descriptive_indicators_df["Interquartile Distance"][col] = descriptive_indicators_df["Lower Quartile"][
                                                                           col] - \
                                                                       descriptive_indicators_df["Upper Quartile"][col]
            descriptive_indicators_df["Kurtosis"][col] = df[col].kurt()
            descriptive_indicators_df["Skewness"][col] = df[col].skew()
            descriptive_indicators_df["Coefficient of Variation"][col] = \
                descriptive_indicators_df["Standard Deviation"][col] / descriptive_indicators_df["Avg"][col]

        cls.descriptive_indicators_df = descriptive_indicators_df

        cur_df = df[descriptive_indicators_col].astype(float)

        return draw_boxplot(cur_df, paint_object, select_model.get_descriptive_indicators_is_rotate())

    @classmethod
    def error_return_draw(cls, paint_object):
        cur_plt = plt.Figure(figsize=(10, 8))
        return cur_plt, paint_object

    @classmethod
    def draw_data_distribution_plot(cls, select_model, color_list: list, label_list: list, name: str, x_label: str,
                                    y_label: str, is_default: bool):
        cur_col = select_model.get_data_distribution_col()

        color_cur_list = [StaticValue.COLORS[random.randint(0, 11)]] if is_default else color_list
        x_cur_label = cur_col if is_default else x_label
        y_cur_label = "Num" if is_default else y_label
        cur_name = "" if is_default else name

        paint_object = PaintObject()
        paint_object.set_color_cur_list(color_cur_list)
        paint_object.set_x_cur_label(x_cur_label)
        paint_object.set_y_cur_label(y_cur_label)
        paint_object.set_name(cur_name)

        if cls.check_col_list(select_model.get_data_distribution_col()):
            return cls.error_return_draw(paint_object)

        counts_mapping = {}
        for x in Dataset.data.loc[:, cur_col].values:
            if x in counts_mapping.keys():
                counts_mapping[x] += 1
            else:
                counts_mapping[x] = 1

        sorting = sorted(counts_mapping.items(), reverse=True, key=lambda m: m[1])
        nums = [x[1] for x in sorting]
        labels = [x[0] for x in sorting]

        if Dataset.check_data_distribution_type(cur_col) == "histogram":
            return draw_histogram(nums, labels, paint_object, select_model.get_data_distribution_is_rotate())
        else:
            return cls.error_return_draw(paint_object)

    @classmethod
    def draw_dependence_plot(cls, select_model, color_list: list, label_list: list, name: str, x_label: str,
                             y_label: str, is_default: bool):
        model_name = select_model.get_models()

        paint_object = PaintObject()

        if cls.check_string(model_name):
            return cls.error_return_draw(paint_object)

        model_name = cls.get_model_name_mapping_reverse()[model_name]
        container = cls.container_dict[model_name]

        # color_cur_list = Config.COLORS if is_default else color_list
        # label_cur_list = [x for x in learning_curve_dict.keys()] if is_default else label_list
        # x_cur_label = "Train Sizes" if is_default else x_label
        # y_cur_label = "Accuracy" if is_default else y_label
        cur_name = "" if is_default else name

        # paint_object.set_color_cur_list(color_cur_list)
        # paint_object.set_label_cur_list(label_cur_list)
        # paint_object.set_x_cur_label(x_cur_label)
        # paint_object.set_y_cur_label(y_cur_label)
        paint_object.set_name(cur_name)

        if cls.check_string(select_model.get_dependence_col()):
            gr.Warning("请选择特征依赖图的相应列")
            return cls.error_return_draw(paint_object)

        return draw_dependence(container.get_model(), container.x_train, cls.data.columns.values.tolist()[1:],
                               select_model.get_dependence_col(), paint_object)

    @classmethod
    def draw_force_plot(cls, select_model, color_list: list, label_list: list, name: str, x_label: str, y_label: str,
                        is_default: bool):
        model_name = select_model.get_models()

        paint_object = PaintObject()

        if cls.check_string(model_name):
            return cls.error_return_draw(paint_object)

        model_name = cls.get_model_name_mapping_reverse()[model_name]
        container = cls.container_dict[model_name]

        # color_cur_list = Config.COLORS if is_default else color_list
        # label_cur_list = [x for x in learning_curve_dict.keys()] if is_default else label_list
        # x_cur_label = "Train Sizes" if is_default else x_label
        # y_cur_label = "Accuracy" if is_default else y_label
        cur_name = "" if is_default else name

        # paint_object.set_color_cur_list(color_cur_list)
        # paint_object.set_label_cur_list(label_cur_list)
        # paint_object.set_x_cur_label(x_cur_label)
        # paint_object.set_y_cur_label(y_cur_label)
        paint_object.set_name(cur_name)

        return draw_force(container.get_model(), container.x_train, cls.data.columns.values.tolist()[1:],
                          select_model.get_force_number(), paint_object)

    @classmethod
    def draw_waterfall_plot(cls, select_model, color_list: list, label_list: list, name: str, x_label: str,
                            y_label: str, is_default: bool):
        model_name = select_model.get_models()

        paint_object = PaintObject()

        if cls.check_string(model_name):
            return cls.error_return_draw(paint_object)

        model_name = cls.get_model_name_mapping_reverse()[model_name]
        container = cls.container_dict[model_name]

        # color_cur_list = Config.COLORS if is_default else color_list
        # label_cur_list = [x for x in learning_curve_dict.keys()] if is_default else label_list
        # x_cur_label = "Train Sizes" if is_default else x_label
        # y_cur_label = "Accuracy" if is_default else y_label
        cur_name = "" if is_default else name

        # paint_object.set_color_cur_list(color_cur_list)
        # paint_object.set_label_cur_list(label_cur_list)
        # paint_object.set_x_cur_label(x_cur_label)
        # paint_object.set_y_cur_label(y_cur_label)
        paint_object.set_name(cur_name)

        return draw_waterfall(container.get_model(), container.x_train, cls.data.columns.values.tolist()[1:],
                              select_model.get_waterfall_number(), paint_object)

    @classmethod
    def draw_learning_curve_plot(cls, select_model, color_list: list, label_list: list, name: str, x_label: str,
                                 y_label: str, is_default: bool):
        cur_dict = {}

        model_list = select_model.get_models()

        for model_name in model_list:
            model_name = cls.get_model_name_mapping_reverse()[model_name]
            cur_dict[model_name] = cls.container_dict[model_name].get_learning_curve_values()

        color_cur_list = StaticValue.COLORS if is_default else color_list
        if is_default:
            label_cur_list = []
            for x in cur_dict.keys():
                label_cur_list.append("train " + str(x))
                label_cur_list.append("validation " + str(x))
        else:
            label_cur_list = label_list

        x_cur_label = "Train Sizes" if is_default else x_label
        y_cur_label = "Accuracy" if is_default else y_label
        cur_name = "" if is_default else name

        paint_object = PaintObject()
        paint_object.set_color_cur_list(color_cur_list)
        paint_object.set_label_cur_list(label_cur_list)
        paint_object.set_x_cur_label(x_cur_label)
        paint_object.set_y_cur_label(y_cur_label)
        paint_object.set_name(cur_name)

        if cls.check_cur_dict(cur_dict):
            return cls.error_return_draw(paint_object)

        return draw_learning_curve_total(cur_dict, paint_object)

    @classmethod
    def draw_shap_beeswarm_plot(cls, select_model, color_list: list, label_list: list, name: str, x_label: str,
                                y_label: str, is_default: bool):
        model_name = select_model.get_models()

        paint_object = PaintObject()

        if cls.check_string(model_name):
            return cls.error_return_draw(paint_object)

        model_name = cls.get_model_name_mapping_reverse()[model_name]
        container = cls.container_dict[model_name]

        # color_cur_list = Config.COLORS if is_default else color_list
        # label_cur_list = [x for x in learning_curve_dict.keys()] if is_default else label_list
        # x_cur_label = "Train Sizes" if is_default else x_label
        # y_cur_label = "Accuracy" if is_default else y_label
        cur_name = "" if is_default else name

        # paint_object.set_color_cur_list(color_cur_list)
        # paint_object.set_label_cur_list(label_cur_list)
        # paint_object.set_x_cur_label(x_cur_label)
        # paint_object.set_y_cur_label(y_cur_label)
        paint_object.set_name(cur_name)

        if cls.check_string(select_model.get_beeswarm_plot_type()):
            gr.Warning("请选择特征蜂群图的图像类型")
            return cls.error_return_draw(paint_object)

        return draw_shap_beeswarm(container.get_model(), container.x_train, cls.data.columns.values.tolist()[1:],
                                  select_model.get_beeswarm_plot_type(), paint_object)

    @classmethod
    def draw_data_fit_plot(cls, select_model, color_list: list, label_list: list, name: str, x_label: str, y_label: str,
                           is_default: bool):
        cur_dict = {}

        model_list = select_model.get_models()

        for model_name in model_list:
            model_name = cls.get_model_name_mapping_reverse()[model_name]
            cur_dict[model_name] = cls.container_dict[model_name].get_data_fit_values()

        color_cur_list = StaticValue.COLORS if is_default else color_list
        if is_default:
            label_cur_list = []
            for x in cur_dict.keys():
                label_cur_list.append("pred " + str(x))
            label_cur_list.append("real data")
        else:
            label_cur_list = label_list

        x_cur_label = "n value" if is_default else x_label
        y_cur_label = "y value" if is_default else y_label
        cur_name = "" if is_default else name

        paint_object = PaintObject()
        paint_object.set_color_cur_list(color_cur_list)
        paint_object.set_label_cur_list(label_cur_list)
        paint_object.set_x_cur_label(x_cur_label)
        paint_object.set_y_cur_label(y_cur_label)
        paint_object.set_name(cur_name)

        if cls.check_cur_dict(cur_dict):
            return cls.error_return_draw(paint_object)

        return draw_data_fit_total(cur_dict, paint_object)

    @classmethod
    def get_shap_beeswarm_plot_type(cls):
        return ["bar", "violin"]

    @classmethod
    def get_file(cls):
        # [绘图]
        if cls.visualize == MN.learning_curve:
            return FilePath.png_base.format(FilePath.learning_curve_plot)
        elif cls.visualize == MN.shap_beeswarm:
            return FilePath.png_base.format(FilePath.shap_beeswarm_plot)
        elif cls.visualize == MN.data_fit:
            return FilePath.png_base.format(FilePath.data_fit_plot)
        elif cls.visualize == MN.waterfall:
            return FilePath.png_base.format(FilePath.waterfall_plot)
        elif cls.visualize == MN.force:
            return FilePath.png_base.format(FilePath.force_plot)
        elif cls.visualize == MN.dependence:
            return FilePath.png_base.format(FilePath.dependence_plot)
        elif cls.visualize == MN.data_distribution:
            return FilePath.png_base.format(FilePath.data_distribution_plot)
        elif cls.visualize == MN.descriptive_indicators:
            return FilePath.png_base.format(FilePath.descriptive_indicators_plot)
        elif cls.visualize == MN.heatmap:
            return FilePath.png_base.format(FilePath.heatmap_plot)
        # 绘图Step 16:在这里添加新的绘图文件路径

    @classmethod
    def check_file(cls):
        return os.path.exists(cls.get_file())

    @classmethod
    def after_get_file(cls):
        return cls.get_file() if cls.check_file() else None

    @classmethod
    def get_model_list(cls):
        model_list = []
        for model_name in cls.container_dict.keys():
            model_list.append(cls.get_model_name_mapping()[model_name])

        return model_list

    @classmethod
    def select_as_model(cls, model_name: str):
        cls.cur_model = cls.get_model_name_mapping_reverse()[model_name]

    @classmethod
    def get_model_mark(cls):
        return True if cls.cur_model != "" else False

    @classmethod
    def get_linear_regression_mark(cls):
        return True if cls.cur_model == MN.linear_regressor else False

    @classmethod
    def get_naive_bayes_classifier_mark(cls):
        return True if cls.cur_model == MN.naive_bayes_classifier else False

    @classmethod
    def get_assign_list(cls):
        return ["分类", "回归"]

    @classmethod
    def get_assign_mapping_reverse(cls):
        return dict(zip(cls.get_assign_list(), [MN.classification, MN.regression]))

    @classmethod
    def choose_assign(cls, assign: str):
        cls.assign = cls.get_assign_mapping_reverse()[assign]

        data_copy = cls.data

        if cls.assign == MN.classification:
            # gr.Info("分类任务请确保目标变量列(第一列)数值为字符型[有限个标签]")
            data_copy.iloc[:, 0] = data_copy.iloc[:, 0].astype(str)
        else:
            # gr.Info("回归任务请确保目标变量列(第一列)数值为数值型")
            data_copy.iloc[:, 0] = data_copy.iloc[:, 0].astype(float)

        cls.data = data_copy
        cls.change_data_type_to_float()

    @classmethod
    def colorpickers_change(cls, paint_object):
        cur_num = paint_object.get_color_cur_num()

        true_list = [gr.ColorPicker(paint_object.get_color_cur_list()[i], visible=True, label=LN.colors[i]) for i in
                     range(cur_num)]

        return true_list + [gr.ColorPicker(visible=False)] * (StaticValue.MAX_NUM - cur_num)

    @classmethod
    def get_model_train_input_params(cls):
        EACH_ROW_NUM = 6 - 1
        output_list = []

        if cls.cur_model and cls.choose_optimize:
            output_dict = ChooseModelParams.choose(cls.cur_model)
            row_unit_num_list = []
            row_len = len(output_dict.keys())
            dict_keys_list = [x for x in output_dict.keys()]

            for k, v in output_dict.items():
                row_unit_num_list.append(len(v))
                for x in v:
                    output_list.append(x)

            return_list = []
            cumulative_sum = 0
            for j in range(row_len):
                return_list.append(gr.Textbox(dict_keys_list[j], visible=cls.check_model_optimize_radio(), show_label=False, elem_classes="params_name"))
                return_list.extend(
                    [gr.Textbox(output_list[k], visible=cls.check_model_optimize_radio(), show_label=False)
                     for k in range(cumulative_sum, cumulative_sum + row_unit_num_list[j])]
                )
                return_list.extend(
                    [gr.Textbox(visible=False)] * (EACH_ROW_NUM - row_unit_num_list[j])
                )

                cumulative_sum += row_unit_num_list[j]

            return_list.extend([gr.Textbox(visible=False)] * (StaticValue.MAX_PARAMS_NUM - row_len - cumulative_sum))

            return return_list

        else:
            return [gr.Textbox(visible=False)] * StaticValue.MAX_PARAMS_NUM

    @classmethod
    def color_textboxs_change(cls, paint_object):
        cur_num = paint_object.get_color_cur_num()

        true_list = [gr.Textbox(paint_object.get_color_cur_list()[i], visible=True, show_label=False) for i in range(cur_num)]

        return true_list + [gr.Textbox(visible=False)] * (StaticValue.MAX_NUM - cur_num)

    @classmethod
    def labels_change(cls, paint_object):
        cur_num = paint_object.get_label_cur_num()

        true_list = [gr.Textbox(paint_object.get_label_cur_list()[i], visible=True, label=LN.labels[i]) for i in
                     range(cur_num)]

        return true_list + [gr.Textbox(visible=False)] * (StaticValue.MAX_NUM - cur_num)

    @classmethod
    def get_model_train_metrics_dataframe(cls):
        if cls.cur_model != "" and cls.get_model_container_status():
            columns_list = ["指标", "数值"]

            output_dict = cls.container_dict[cls.cur_model].get_info()["指标"]

            output_df = pd.DataFrame(columns=columns_list)
            output_df["指标"] = [x for x in output_dict.keys() if x in ChooseModelMetrics.choose(cls.cur_model)]
            output_df["数值"] = [output_dict[x] for x in output_df["指标"]]

            return output_df

    @classmethod
    def get_model_train_params_dataframe(cls):
        if cls.cur_model != "" and cls.get_model_container_status():
            columns_list = ["参数", "数值"]

            output_dict = cls.container_dict[cls.cur_model].get_info()["参数"]

            output_df = pd.DataFrame(columns=columns_list)
            output_df["参数"] = [x for x in output_dict.keys() if x in ChooseModelParams.choose(cls.cur_model).keys()]
            output_df["数值"] = [output_dict[x] for x in output_df["参数"]]

            return output_df

    @classmethod
    def get_str_col_list(cls):
        str_col_list = []
        for col in cls.get_col_list():
            if all(isinstance(x, str) for x in cls.data.loc[:, col]):
                str_col_list.append(col)

        return str_col_list

    @classmethod
    def get_float_col_list(cls):
        float_col_list = []
        for col in cls.get_col_list():
            if all(isinstance(x, float) for x in cls.data.loc[:, col]):
                float_col_list.append(col)

        return float_col_list

    @classmethod
    def check_data_distribution_type(cls, col):
        if all(isinstance(x, str) for x in cls.data.loc[:, col]):
            return "histogram"
        # elif all(isinstance(x, float) for x in cls.data.loc[:, col]):
        #     return "line_graph"
        else:
            gr.Warning("所选列的所有数据必须为字符型或浮点型")

    @classmethod
    def check_col_list(cls, col):
        if not col:
            gr.Warning("请选择所需列")
            return True
        return False

    @classmethod
    def check_train_model(cls, optimize, train_size):
        if cls.cur_model == "":
            gr.Warning("请选择所需训练的模型")
            return True
        if not optimize:
            gr.Warning("请选择超参数优化方法")
            return True
        if not train_size:
            gr.Warning("请输入训练集所占比例")
            return True
        if not (0 < train_size < 1):
            gr.Warning("训练集所占比例必须是0到1之间的一个小数")
            return True
        return False

    @classmethod
    def check_train_model_other_related(cls, extra_components_list):
        # [模型]
        if cls.cur_model == MN.linear_regressor:
            if not extra_components_list[0]:
                gr.Warning("请选择线性回归对应的模型")
                return True
        elif cls.cur_model == MN.naive_bayes_classifier:
            if not extra_components_list[1]:
                gr.Warning("请选择朴素贝叶斯对应的模型")
                return True
        # 模型Step 3:在这里添加新的模型的额外组件的空白判断 (给出错误提示) (extra_components_list[]下标按传入的顺序即可)
        return False

    @classmethod
    def check_cur_dict(cls, cur_dict):
        if not cur_dict:
            gr.Warning("请选择绘图所需的模型")
            return True
        return False

    @classmethod
    def check_string(cls, string):
        if not string:
            gr.Warning("请选择绘图所需的模型")
            return True
        return False

    @classmethod
    def add_index_into_df(cls, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        index_df = pd.DataFrame([x for x in range(len(df))], columns=["[*index]"])

        return pd.concat([index_df, df], axis=1)

    @classmethod
    def load_data(cls, sort):
        type = ""
        if sort == "Iris Dataset":
            sk_data = load_iris()
            type = "classification"
        elif sort == "Wine Dataset":
            sk_data = load_wine()
            type = "classification"
        elif sort == "Breast Cancer Dataset":
            sk_data = load_breast_cancer()
            type = "classification"
        elif sort == "Diabetes Dataset":
            sk_data = load_diabetes()
            type = "regression"
        elif sort == "California Housing Dataset":
            df = pd.read_csv("./data/fetch_california_housing.csv")
            return df
        else:
            sk_data = load_iris()
            type = "classification"

        if type == "classification":
            target_data = sk_data.target.astype(str)
            for i in range(len(sk_data.target_names)):
                target_data = np.where(target_data == str(i), sk_data.target_names[i], target_data)
        else:
            target_data = sk_data.target

        feature_names = sk_data.feature_names
        sk_feature_names = ["target"] + feature_names.tolist() if isinstance(feature_names, np.ndarray) else [
                                                                                                                 "target"] + feature_names
        sk_data = np.concatenate((target_data.reshape(-1, 1), sk_data.data), axis=1)

        df = pd.DataFrame(data=sk_data, columns=sk_feature_names)

        return df

    @classmethod
    def load_custom_data(cls, file):
        if "xlsx" in file or "xls" in file:
            return pd.read_excel(file)
        elif "csv" in file:
            return pd.read_csv(file)


def get_return_extra(is_visible, extra_gr_dict: dict = None):
    if is_visible:
        gr_dict = {
            draw_file: gr.File(Dataset.after_get_file(), visible=Dataset.check_file()),
        }

        if extra_gr_dict:
            gr_dict.update(extra_gr_dict)

        return gr_dict

    gr_dict = {
        draw_plot: gr.Plot(visible=False),
        draw_file: gr.File(visible=False),
    }

    gr_dict.update(dict(zip(colorpickers, [gr.ColorPicker(visible=False)] * StaticValue.MAX_NUM)))
    gr_dict.update(dict(zip(color_textboxs, [gr.Textbox(visible=False)] * StaticValue.MAX_NUM)))
    gr_dict.update(dict(zip(legend_labels_textboxs, [gr.Textbox(visible=False)] * StaticValue.MAX_NUM)))
    gr_dict.update({title_name_textbox: gr.Textbox(visible=False)})
    gr_dict.update({x_label_textbox: gr.Textbox(visible=False)})
    gr_dict.update({y_label_textbox: gr.Textbox(visible=False)})

    return gr_dict


def get_return(is_visible, extra_gr_dict: dict = None):
    if is_visible:
        gr_dict = {
            display_dataset_dataframe: gr.Dataframe(Dataset.add_index_into_df(Dataset.data), type="pandas",
                                                    visible=True),
            display_dataset: gr.File(Dataset.after_get_display_dataset_file(),
                                     visible=Dataset.check_display_dataset_file()),
            display_total_col_num_text: gr.Textbox(str(Dataset.get_total_col_num()), visible=True,
                                                   label=LN.display_total_col_num_text),
            display_total_row_num_text: gr.Textbox(str(Dataset.get_total_row_num()), visible=True,
                                                   label=LN.display_total_row_num_text),
            display_na_list_text: gr.Textbox(Dataset.get_na_list_str(), visible=True, label=LN.display_na_list_text),
            del_all_na_col_button: gr.Button(LN.del_all_na_col_button, visible=True),
            display_duplicate_num_text: gr.Textbox(str(Dataset.get_duplicate_num()), visible=True,
                                                   label=LN.display_duplicate_num_text),
            del_duplicate_button: gr.Button(LN.del_duplicate_button, visible=True),
            del_col_checkboxgroup: gr.Checkboxgroup(Dataset.get_col_list(), visible=True,
                                                    label=LN.del_col_checkboxgroup),
            del_col_button: gr.Button(LN.del_col_button, visible=True),
            remain_row_slider: gr.Slider(0, Dataset.get_max_num(), value=Dataset.get_total_row_num(), step=1,
                                         visible=True, label=LN.remain_row_slider),
            remain_row_button: gr.Button(LN.remain_row_button, visible=True),
            encode_label_button: gr.Button(LN.encode_label_button, visible=True),
            encode_label_checkboxgroup: gr.Checkboxgroup(Dataset.get_non_numeric_list(), visible=True,
                                                         label=LN.encode_label_checkboxgroup),
            display_encode_label_dataframe: gr.Dataframe(visible=False),
            data_type_dataframe: gr.Dataframe(Dataset.get_data_type(), visible=True),
            change_data_type_to_float_button: gr.Button(LN.change_data_type_to_float_button, visible=True),
            select_as_y_radio: gr.Radio(Dataset.get_col_list(), visible=True, label=LN.select_as_y_radio),
            standardize_data_checkboxgroup: gr.Checkboxgroup(Dataset.get_non_standardized_data(), visible=True,
                                                             label=LN.standardize_data_checkboxgroup),
            standardize_data_button: gr.Button(LN.standardize_data_button, visible=True),
            choose_assign_radio: gr.Radio(Dataset.get_assign_list(), visible=True, label=LN.choose_assign_radio),

            select_as_model_radio: gr.Radio(Dataset.get_model_list(), visible=Dataset.check_before_train(),
                                            label=LN.select_as_model_radio),
            model_optimize_radio: gr.Radio(Dataset.get_optimize_list(), visible=Dataset.check_before_train(),
                                           label=LN.model_optimize_radio),
            train_size_textbox: gr.Textbox(str(0.8), visible=Dataset.check_before_train(), label=LN.train_size_textbox),
            model_train_button: gr.Button(LN.model_train_button, visible=Dataset.check_before_train()),
            model_train_checkbox: gr.Checkbox(Dataset.get_model_container_status(),
                                              visible=Dataset.check_select_model(), label=Dataset.get_model_label()),
            model_train_params_dataframe: gr.Dataframe(Dataset.get_model_train_params_dataframe(), type="pandas",
                                                       visible=Dataset.get_model_container_status()),
            model_train_metrics_dataframe: gr.Dataframe(Dataset.get_model_train_metrics_dataframe(), type="pandas",
                                                        visible=Dataset.get_model_container_status()),

            draw_plot: gr.Plot(visible=False),
            draw_file: gr.File(visible=False),
            title_name_textbox: gr.Textbox(visible=False),
            x_label_textbox: gr.Textbox(visible=False),
            y_label_textbox: gr.Textbox(visible=False),

            # [模型]
            linear_regression_model_radio: gr.Radio(Dataset.get_linear_regression_model_list(),
                                                    visible=Dataset.get_linear_regression_mark(),
                                                    label=LN.linear_regression_model_radio),
            naive_bayes_classification_model_radio: gr.Radio(Dataset.get_naive_bayes_classifier_model_list(),
                                                             visible=Dataset.get_naive_bayes_classifier_mark(),
                                                             label=LN.naive_bayes_classification_model_radio),
            # 模型Step 8:在这里添加新的模型额外的组件+写入新的Dataset类方法的函数

            # [绘图]
            heatmap_checkboxgroup: gr.Checkboxgroup(Dataset.get_float_col_list(), visible=True,
                                                    label=LN.heatmap_checkboxgroup),
            heatmap_is_rotate: gr.Checkbox(visible=True, label=LN.heatmap_is_rotate),
            heatmap_button: gr.Button(LN.heatmap_button, visible=True),
            descriptive_indicators_checkboxgroup: gr.Checkboxgroup(Dataset.get_float_col_list(), visible=True,
                                                                   label=LN.descriptive_indicators_checkboxgroup),
            data_distribution_radio: gr.Radio(Dataset.get_str_col_list(), visible=True,
                                              label=LN.data_distribution_radio),
            data_distribution_is_rotate: gr.Checkbox(visible=True, label=LN.data_distribution_is_rotate),
            data_distribution_button: gr.Button(LN.data_distribution_button, visible=True),
            descriptive_indicators_is_rotate: gr.Checkbox(visible=True, label=LN.descriptive_indicators_is_rotate),
            descriptive_indicators_dataframe: gr.Dataframe(Dataset.get_descriptive_indicators_df(), type="pandas",
                                                           visible=Dataset.check_descriptive_indicators_df()),
            descriptive_indicators_button: gr.Button(LN.descriptive_indicators_button, visible=True),
            learning_curve_checkboxgroup: gr.Checkboxgroup(Dataset.get_trained_model_list(),
                                                           visible=Dataset.check_before_train(),
                                                           label=LN.learning_curve_checkboxgroup),
            learning_curve_button: gr.Button(LN.learning_curve_button, visible=Dataset.check_before_train()),
            shap_beeswarm_radio: gr.Radio(Dataset.get_trained_model_list(), visible=Dataset.check_before_train(),
                                          label=LN.shap_beeswarm_radio),
            shap_beeswarm_type: gr.Radio(Dataset.get_shap_beeswarm_plot_type(), visible=Dataset.check_before_train(),
                                         label=LN.shap_beeswarm_type),
            shap_beeswarm_button: gr.Button(LN.shap_beeswarm_button, visible=Dataset.check_before_train()),
            data_fit_checkboxgroup: gr.Checkboxgroup(Dataset.get_trained_model_list(),
                                                     visible=Dataset.check_before_train(),
                                                     label=LN.data_fit_checkboxgroup),
            data_fit_button: gr.Button(LN.data_fit_button, visible=Dataset.check_before_train()),
            waterfall_radio: gr.Radio(Dataset.get_trained_model_list(), visible=Dataset.check_before_train(),
                                      label=LN.waterfall_radio),
            waterfall_number: gr.Slider(0, Dataset.get_total_row_num(), value=0, step=1,
                                        visible=Dataset.check_before_train(), label=LN.waterfall_number),
            waterfall_button: gr.Button(LN.waterfall_button, visible=Dataset.check_before_train()),
            force_radio: gr.Radio(Dataset.get_trained_model_list(), visible=Dataset.check_before_train(),
                                  label=LN.force_radio),
            force_number: gr.Slider(0, Dataset.get_total_row_num(), value=0, step=1,
                                    visible=Dataset.check_before_train(), label=LN.force_number),
            force_button: gr.Button(LN.force_button, visible=Dataset.check_before_train()),
            dependence_radio: gr.Radio(Dataset.get_trained_model_list(), visible=Dataset.check_before_train(),
                                       label=LN.dependence_radio),
            dependence_col: gr.Radio(Dataset.get_dependence_col(), visible=Dataset.check_before_train(),
                                     label=LN.dependence_col),
            dependence_button: gr.Button(LN.dependence_button, visible=Dataset.check_before_train()),
            # 绘图Step 13:在这里添加新的绘图组件
        }

        gr_dict.update(dict(zip(colorpickers, [gr.ColorPicker(visible=False)] * StaticValue.MAX_NUM)))
        gr_dict.update(dict(zip(color_textboxs, [gr.Textbox(visible=False)] * StaticValue.MAX_NUM)))
        gr_dict.update(dict(zip(legend_labels_textboxs, [gr.Textbox(visible=False)] * StaticValue.MAX_NUM)))

        gr_dict.update(dict(zip(model_train_params_textboxs, Dataset.get_model_train_input_params())))

        if extra_gr_dict:
            gr_dict.update(extra_gr_dict)

        return gr_dict

    gr_dict = {
        choose_custom_dataset_file: gr.File(None, visible=True),
        display_dataset_dataframe: gr.Dataframe(visible=False),
        display_dataset: gr.File(visible=False),
        display_total_col_num_text: gr.Textbox(visible=False),
        display_total_row_num_text: gr.Textbox(visible=False),
        display_na_list_text: gr.Textbox(visible=False),
        del_all_na_col_button: gr.Button(visible=False),
        display_duplicate_num_text: gr.Textbox(visible=False),
        del_duplicate_button: gr.Button(visible=False),
        del_col_checkboxgroup: gr.Checkboxgroup(visible=False),
        del_col_button: gr.Button(visible=False),
        remain_row_slider: gr.Slider(visible=False),
        encode_label_button: gr.Button(visible=False),
        display_encode_label_dataframe: gr.Dataframe(visible=False),
        encode_label_checkboxgroup: gr.Checkboxgroup(visible=False),
        data_type_dataframe: gr.Dataframe(visible=False),
        change_data_type_to_float_button: gr.Button(visible=False),
        standardize_data_checkboxgroup: gr.Checkboxgroup(visible=False),
        standardize_data_button: gr.Button(visible=False),
        select_as_y_radio: gr.Radio(visible=False),
        train_size_textbox: gr.Textbox(visible=False),
        model_optimize_radio: gr.Radio(visible=False),
        model_train_button: gr.Button(visible=False),
        model_train_checkbox: gr.Checkbox(visible=False),
        model_train_metrics_dataframe: gr.Dataframe(visible=False),
        model_train_params_dataframe: gr.Dataframe(visible=False),
        select_as_model_radio: gr.Radio(visible=False),
        choose_assign_radio: gr.Radio(visible=False),

        draw_plot: gr.Plot(visible=False),
        draw_file: gr.File(visible=False),
        title_name_textbox: gr.Textbox(visible=False),
        x_label_textbox: gr.Textbox(visible=False),
        y_label_textbox: gr.Textbox(visible=False),

        # [模型]
        linear_regression_model_radio: gr.Radio(visible=False),
        naive_bayes_classification_model_radio: gr.Radio(visible=False),
        # 模型Step 9:在这里添加新的模型额外的组件 (不可见)

        # [绘图]
        heatmap_checkboxgroup: gr.Checkboxgroup(visible=False),
        heatmap_is_rotate: gr.Checkbox(visible=False),
        heatmap_button: gr.Button(visible=False),
        data_distribution_radio: gr.Radio(visible=False),
        data_distribution_is_rotate: gr.Checkbox(visible=False),
        data_distribution_button: gr.Button(visible=False),
        descriptive_indicators_checkboxgroup: gr.Checkboxgroup(visible=False),
        descriptive_indicators_is_rotate: gr.Checkbox(visible=False),
        descriptive_indicators_dataframe: gr.Dataframe(visible=False),
        descriptive_indicators_button: gr.Button(visible=False),
        learning_curve_checkboxgroup: gr.Checkboxgroup(visible=False),
        learning_curve_button: gr.Button(visible=False),
        shap_beeswarm_radio: gr.Radio(visible=False),
        shap_beeswarm_type: gr.Radio(visible=False),
        shap_beeswarm_button: gr.Button(visible=False),
        data_fit_checkboxgroup: gr.Checkboxgroup(visible=False),
        data_fit_button: gr.Button(visible=False),
        waterfall_radio: gr.Radio(visible=False),
        waterfall_number: gr.Slider(visible=False),
        waterfall_button: gr.Button(visible=False),
        force_radio: gr.Radio(visible=False),
        force_number: gr.Slider(visible=False),
        force_button: gr.Button(visible=False),
        dependence_radio: gr.Radio(visible=False),
        dependence_col: gr.Radio(visible=False),
        dependence_button: gr.Button(visible=False),
        # 绘图Step 14:在这里添加新的组件 (不可见)
    }

    gr_dict.update(dict(zip(colorpickers, [gr.ColorPicker(visible=False)] * StaticValue.MAX_NUM)))
    gr_dict.update(dict(zip(color_textboxs, [gr.Textbox(visible=False)] * StaticValue.MAX_NUM)))
    gr_dict.update(dict(zip(legend_labels_textboxs, [gr.Textbox(visible=False)] * StaticValue.MAX_NUM)))

    gr_dict.update(dict(zip(model_train_params_textboxs, [gr.Textbox(visible=False)] * StaticValue.MAX_PARAMS_NUM)))

    return gr_dict


# 选择任务类型(强制转换第1列)
def choose_assign(assign: str):
    try:
        Dataset.choose_assign(assign)
    except ValueError:
        gr.Warning("回归任务中目标变量列(第一列)数值不能为字符型数据")
        return get_return(True)

    return get_return(True)


# 数据模型
def select_as_model(model_name: str):
    Dataset.select_as_model(model_name)

    return get_return(True)


# [绘图]

# 绘图Step 7:在这里添加新的绘图监听事件函数

# 关系热力图监听事件主体函数
def heatmap_first_draw_plot(*inputs):
    Dataset.visualize = MN.heatmap
    return before_train_first_draw_plot(inputs)


# 描述指标图监听事件主体函数
def descriptive_indicators_first_draw_plot(*inputs):
    Dataset.visualize = MN.descriptive_indicators
    return before_train_first_draw_plot(inputs)


# 数据分布图监听事件主体函数
def data_distribution_first_draw_plot(*inputs):
    Dataset.visualize = MN.data_distribution
    return before_train_first_draw_plot(inputs)


# 依赖图监听事件主体函数
def dependence_first_draw_plot(*inputs):
    Dataset.visualize = MN.dependence
    return first_draw_plot(inputs)


# 力图监听事件主体函数
def force_first_draw_plot(*inputs):
    Dataset.visualize = MN.force
    return first_draw_plot(inputs)


# 瀑布图监听事件主体函数
def waterfall_first_draw_plot(*inputs):
    Dataset.visualize = MN.waterfall
    return first_draw_plot(inputs)


# 数据拟合图监听事件主体函数
def data_fit_first_draw_plot(*inputs):
    Dataset.visualize = MN.data_fit
    return first_draw_plot(inputs)


# 蜂群图监听事件主体函数
def shap_beeswarm_first_draw_plot(*inputs):
    Dataset.visualize = MN.shap_beeswarm
    return first_draw_plot(inputs)


# 学习曲线图监听事件主体函数
def learning_curve_first_draw_plot(*inputs):
    Dataset.visualize = MN.learning_curve
    return first_draw_plot(inputs)


def before_train_first_draw_plot(inputs):
    select_model = SelectModel()
    x_label = ""
    y_label = ""
    name = ""
    color_list = []
    label_list = []

    # [绘图][无训练模型]
    if Dataset.visualize == MN.data_distribution:
        select_model.set_data_distribution_col(inputs[0])
        select_model.set_data_distribution_is_rotate(inputs[1])
    elif Dataset.visualize == MN.descriptive_indicators:
        select_model.set_descriptive_indicators_is_rotate(inputs[0])
        select_model.set_descriptive_indicators_col(inputs[1])
    elif Dataset.visualize == MN.heatmap:
        select_model.set_heatmap_col(inputs[0])
        select_model.set_heatmap_is_rotate(inputs[1])
    # 绘图Step 8:在这里添加新的绘图值设定 (通过类方法) (无训练模型的绘图方法)

    cur_plt, paint_object = Dataset.draw_plot(select_model, color_list, label_list, name, x_label, y_label, True)

    return first_draw_plot_with_non_first_draw_plot(cur_plt, paint_object)


def first_draw_plot(inputs):
    select_model = SelectModel()
    select_model.set_models(inputs[0])
    x_label = ""
    y_label = ""
    name = ""
    color_list = []
    label_list = []

    # [绘图][有训练模型]
    if Dataset.visualize == MN.shap_beeswarm:
        select_model.set_beeswarm_plot_type(inputs[1])
    elif Dataset.visualize == MN.waterfall:
        select_model.set_waterfall_number(inputs[1])
    elif Dataset.visualize == MN.force:
        select_model.set_force_number(inputs[1])
    elif Dataset.visualize == MN.dependence:
        select_model.set_dependence_col(inputs[1])
    # 绘图Step 8:在这里添加新的绘图值设定 (通过类方法) (有训练模型的绘图方法)

    cur_plt, paint_object = Dataset.draw_plot(select_model, color_list, label_list, name, x_label, y_label, True)

    return first_draw_plot_with_non_first_draw_plot(cur_plt, paint_object)


# 可视化通用组件监听事件主体函数
def is_color_text_out_non_first_draw_plot(*inputs):
    return non_first_draw_plot(inputs, True)


# 可视化通用组件监听事件主体函数
def not_color_text_out_non_first_draw_plot(*inputs):
    return non_first_draw_plot(inputs, False)


def non_first_draw_plot(inputs, is_color_text):
    name = inputs[0]
    x_label = inputs[1]
    y_label = inputs[2]
    color_list = list(inputs[StaticValue.MAX_NUM + 3: 2 * StaticValue.MAX_NUM + 3]) \
        if is_color_text else list(inputs[3: StaticValue.MAX_NUM + 3])
    label_list = list(inputs[2 * StaticValue.MAX_NUM + 3: 3 * StaticValue.MAX_NUM + 3])
    start_index = 3 * StaticValue.MAX_NUM + 3

    select_model = SelectModel()

    # 若输入的是颜色的十六进制，则判断输入是否合法
    for color in color_list:
        if len(color) != 7 or color[0] != "#":
            gr.Warning("颜色的十六进制输入有误")
            return get_return(True)

    # [绘图]
    if Dataset.visualize == MN.learning_curve:
        select_model.set_models(inputs[start_index + 0])
        select_model.set_beeswarm_plot_type(inputs[start_index + 1])
    elif Dataset.visualize == MN.shap_beeswarm:
        select_model.set_models(inputs[start_index + 2])
    elif Dataset.visualize == MN.data_fit:
        select_model.set_models(inputs[start_index + 3])
    elif Dataset.visualize == MN.waterfall:
        select_model.set_models(inputs[start_index + 4])
        select_model.set_waterfall_number(inputs[start_index + 5])
    elif Dataset.visualize == MN.force:
        select_model.set_models(inputs[start_index + 6])
        select_model.set_force_number(inputs[start_index + 7])
    elif Dataset.visualize == MN.dependence:
        select_model.set_models(inputs[start_index + 8])
        select_model.set_dependence_col(inputs[start_index + 9])
    elif Dataset.visualize == MN.data_distribution:
        select_model.set_data_distribution_col(inputs[start_index + 10])
        select_model.set_data_distribution_is_rotate(inputs[start_index + 11])
    elif Dataset.visualize == MN.descriptive_indicators:
        select_model.set_descriptive_indicators_is_rotate(inputs[start_index + 12])
        select_model.set_descriptive_indicators_col(inputs[start_index + 13])
    elif Dataset.visualize == MN.descriptive_indicators:
        select_model.set_heatmap_col(inputs[start_index + 14])
        select_model.set_heatmap_is_rotate(inputs[start_index + 15])
    # 绘图Step 11:在这里添加新的出入参数赋值 (根据传入的组件，下标为 start_index+顺序)

    else:
        select_model.set_models(inputs[start_index])

    cur_plt, paint_object = Dataset.draw_plot(select_model, color_list, label_list, name, x_label, y_label, False)

    return first_draw_plot_with_non_first_draw_plot(cur_plt, paint_object)


def first_draw_plot_with_non_first_draw_plot(cur_plt, paint_object):
    extra_gr_dict = {}

    # [绘图]
    if Dataset.visualize == MN.learning_curve:
        cur_plt.savefig(FilePath.png_base.format(FilePath.learning_curve_plot), dpi=300)
        extra_gr_dict.update({draw_plot: gr.Plot(cur_plt, visible=True, label=LN.learning_curve_plot)})
    elif Dataset.visualize == MN.shap_beeswarm:
        cur_plt.savefig(FilePath.png_base.format(FilePath.shap_beeswarm_plot), dpi=300)
        extra_gr_dict.update({draw_plot: gr.Plot(cur_plt, visible=True, label=LN.shap_beeswarm_plot)})
    elif Dataset.visualize == MN.data_fit:
        cur_plt.savefig(FilePath.png_base.format(FilePath.data_fit_plot), dpi=300)
        extra_gr_dict.update({draw_plot: gr.Plot(cur_plt, visible=True, label=LN.data_fit_plot)})
    elif Dataset.visualize == MN.waterfall:
        cur_plt.savefig(FilePath.png_base.format(FilePath.waterfall_plot), dpi=300)
        extra_gr_dict.update({draw_plot: gr.Plot(cur_plt, visible=True, label=LN.waterfall_plot)})
    elif Dataset.visualize == MN.force:
        cur_plt.savefig(FilePath.png_base.format(FilePath.force_plot), dpi=300)
        extra_gr_dict.update({draw_plot: gr.Plot(cur_plt, visible=True, label=LN.force_plot)})
    elif Dataset.visualize == MN.dependence:
        cur_plt.savefig(FilePath.png_base.format(FilePath.dependence_plot), dpi=300)
        extra_gr_dict.update({draw_plot: gr.Plot(cur_plt, visible=True, label=LN.dependence_plot)})
    elif Dataset.visualize == MN.data_distribution:
        cur_plt.savefig(FilePath.png_base.format(FilePath.data_distribution_plot), dpi=300)
        extra_gr_dict.update({draw_plot: gr.Plot(cur_plt, visible=True, label=LN.data_distribution_plot)})
    elif Dataset.visualize == MN.descriptive_indicators:
        cur_plt.savefig(FilePath.png_base.format(FilePath.descriptive_indicators_plot), dpi=300)
        extra_gr_dict.update({draw_plot: gr.Plot(cur_plt, visible=True, label=LN.descriptive_indicators_plot)})
        extra_gr_dict.update({descriptive_indicators_dataframe: gr.Dataframe(Dataset.get_descriptive_indicators_df(),
                                                                             type="pandas",
                                                                             visible=Dataset.check_descriptive_indicators_df())})
    elif Dataset.visualize == MN.heatmap:
        cur_plt.savefig(FilePath.png_base.format(FilePath.heatmap_plot), dpi=300)
        extra_gr_dict.update({draw_plot: gr.Plot(cur_plt, visible=True, label=LN.heatmap_plot)})
    # 绘图Step 10:在这里添加新的绘图方法

    extra_gr_dict.update(dict(zip(colorpickers, Dataset.colorpickers_change(paint_object))))
    extra_gr_dict.update(dict(zip(color_textboxs, Dataset.color_textboxs_change(paint_object))))
    extra_gr_dict.update(dict(zip(legend_labels_textboxs, Dataset.labels_change(paint_object))))
    extra_gr_dict.update(
        {title_name_textbox: gr.Textbox(paint_object.get_name(), visible=True, label=LN.title_name_textbox)})
    extra_gr_dict.update(
        {x_label_textbox: gr.Textbox(paint_object.get_x_cur_label(), visible=True, label=LN.x_label_textbox)})
    extra_gr_dict.update(
        {y_label_textbox: gr.Textbox(paint_object.get_y_cur_label(), visible=True, label=LN.y_label_textbox)})

    return get_return_extra(True, extra_gr_dict)


# 模型训练
def train_model(*input):
    params_list = input[0: StaticValue.MAX_PARAMS_NUM]
    optimize = input[StaticValue.MAX_PARAMS_NUM]
    train_size = input[StaticValue.MAX_PARAMS_NUM+1]
    extra_components_list = input[StaticValue.MAX_PARAMS_NUM+2:]

    # 训练集分割比例有效判断
    try:
        train_size = float(train_size)
    except Exception:
        gr.Warning("训练集所占比例必须是小数")
        return get_return(True)

    # 模型选择和超参数优化组件的空白判断
    if Dataset.check_train_model(optimize, train_size):
        return get_return(True)
    # 模型额外组件的空白判断
    if Dataset.check_train_model_other_related(extra_components_list):
        return get_return(True)
    # 模型训练
    Dataset.train_model(optimize, params_list, train_size, extra_components_list)

    return get_return(True)


# 选择因变量
def select_as_y(col: str):
    Dataset.select_as_y(col)

    return get_return(True)


# 标准化数据
def standardize_data(col_list: list):
    Dataset.standardize_data(col_list)

    return get_return(True)


# 将所有数据强制转换为浮点型(除第1列之外)
def change_data_type_to_float():
    try:
        Dataset.change_data_type_to_float()
    except ValueError:
        gr.Warning("请先将数据源中的字符型数据转换为数值型 ([分类任务]除第一列以外)")
        return get_return(True)

    return get_return(True)


# 字符型列转数值型列
def encode_label(col_list: list):
    Dataset.encode_label(col_list)

    return get_return(True, {
        display_encode_label_dataframe: gr.Dataframe(Dataset.get_str2int_mappings_df(), type="pandas", visible=True,
                                                     label=LN.display_encode_label_dataframe)})


# 删除所有重复的行
def del_duplicate():
    Dataset.del_duplicate()

    return get_return(True)


# 删除所有存在缺失值的列
def del_all_na_col():
    Dataset.del_all_na_col()

    return get_return(True)


# 保留行
def remain_row(num):
    Dataset.remain_row(num)

    return get_return(True)


# 删除所选列
def del_col(col_list: list):
    Dataset.del_col(col_list)

    return get_return(True)


# 选择数据源
def choose_dataset(file: str):
    # 更改数据源会自动清空所有数据
    Dataset.reset_containers()

    if file == "自定义":
        Dataset.clear()

        return get_return(False)

    df = Dataset.load_data(file)
    Dataset.update(file, df)

    return get_return(True, {choose_custom_dataset_file: gr.File(visible=False)})


# 选择用户上传的数据源
def choose_custom_dataset(file: str):
    df = Dataset.load_custom_data(file)
    Dataset.update(file, df)

    return get_return(True, {choose_custom_dataset_file: gr.File(Dataset.file, visible=True)})


def select_model_optimize_radio(optimize):
    optimize = Dataset.get_optimize_name_mapping()[optimize]

    if optimize == "grid_search":
        Dataset.choose_optimize = "grid_search"
    elif optimize == "bayes_search":
        Dataset.choose_optimize = "bayes_search"
    elif optimize == "None":
        Dataset.choose_optimize = "None"

    return get_return(True)


def linear_regression_model_radio_change(model):
    if model:
        Dataset.linear_regression_model_type = Dataset.get_linear_regression_model_name_mapping()[model]
        return get_return(True)


def naive_bayes_classification_model_radio_change(model):
    if model:
        Dataset.naive_bayes_classifier_model_type = Dataset.get_naive_bayes_classifier_model_name_mapping()[model]
        return get_return(True)


# 主程序
# js: 使用的js代码
with gr.Blocks(js="./design/welcome.js", css="./design/custom.css") as demo:
    '''
        组件的数量、种类和排版
    '''

    with gr.Tab("机器学习"):

        '''
            数据预处理
        '''

        # 选择数据源
        with gr.Accordion("数据源"):
            with gr.Group():
                # 刚进入程序时，仅"选择数据库"radio可见 (visible=True)，其他组件默认为不可见 (visible=False)
                choose_dataset_radio = gr.Radio(Dataset.get_dataset_list(), label=LN.choose_dataset_radio)
                choose_custom_dataset_file = gr.File(visible=False)

        # 显示数据表信息
        with gr.Accordion("当前数据信息"):
            display_dataset_dataframe = gr.Dataframe(visible=False)
            display_dataset = gr.File(visible=False)
            with gr.Row():
                display_total_col_num_text = gr.Textbox(visible=False)
                display_total_row_num_text = gr.Textbox(visible=False)
                with gr.Column():
                    remain_row_slider = gr.Slider(visible=False)
                    remain_row_button = gr.Button(visible=False)
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        display_na_list_text = gr.Textbox(visible=False)
                        display_duplicate_num_text = gr.Textbox(visible=False)
                    with gr.Row():
                        del_all_na_col_button = gr.Button(visible=False)
                        del_duplicate_button = gr.Button(visible=False)

        # 操作数据表
        with gr.Accordion("数据处理"):
            select_as_y_radio = gr.Radio(visible=False)
            with gr.Row():
                with gr.Column():
                    data_type_dataframe = gr.Dataframe(visible=False)
                    change_data_type_to_float_button = gr.Button(visible=False)
                    choose_assign_radio = gr.Radio(visible=False)
                with gr.Column():
                    del_col_checkboxgroup = gr.Checkboxgroup(visible=False)
                    del_col_button = gr.Button(visible=False)
                    encode_label_checkboxgroup = gr.Checkboxgroup(visible=False)
                    encode_label_button = gr.Button(visible=False)
                    display_encode_label_dataframe = gr.Dataframe(visible=False)
                    standardize_data_checkboxgroup = gr.Checkboxgroup(visible=False)
                    standardize_data_button = gr.Button(visible=False)

        # 数据模型
        with gr.Accordion("数据模型"):
            # [模型]
            select_as_model_radio = gr.Radio(visible=False)
            linear_regression_model_radio = gr.Radio(visible=False)
            naive_bayes_classification_model_radio = gr.Radio(visible=False)
            train_size_textbox = gr.Textbox(visible=False)
            model_optimize_radio = gr.Radio(visible=False)

            model_train_params_textboxs = []
            with gr.Accordion("超参数列表"):
                ONE_PIECE_RANGE = int(StaticValue.MAX_PARAMS_NUM / 10)
                with gr.Row():
                    for i in range(0, ONE_PIECE_RANGE):
                        with gr.Row():
                            textbox = gr.Textbox(visible=False)
                            model_train_params_textboxs.append(textbox)
                with gr.Row():
                    for i in range(ONE_PIECE_RANGE, 2 * ONE_PIECE_RANGE):
                        with gr.Row():
                            textbox = gr.Textbox(visible=False)
                            model_train_params_textboxs.append(textbox)
                with gr.Row():
                    for i in range(2 * ONE_PIECE_RANGE, 3 * ONE_PIECE_RANGE):
                        with gr.Row():
                            textbox = gr.Textbox(visible=False)
                            model_train_params_textboxs.append(textbox)
                with gr.Row():
                    for i in range(3 * ONE_PIECE_RANGE, 4 * ONE_PIECE_RANGE):
                        with gr.Row():
                            textbox = gr.Textbox(visible=False)
                            model_train_params_textboxs.append(textbox)
                with gr.Row():
                    for i in range(4 * ONE_PIECE_RANGE, 5 * ONE_PIECE_RANGE):
                        with gr.Row():
                            textbox = gr.Textbox(visible=False)
                            model_train_params_textboxs.append(textbox)
                with gr.Row():
                    for i in range(5 * ONE_PIECE_RANGE, 6 * ONE_PIECE_RANGE):
                        with gr.Row():
                            textbox = gr.Textbox(visible=False)
                            model_train_params_textboxs.append(textbox)
                with gr.Row():
                    for i in range(6 * ONE_PIECE_RANGE, 7 * ONE_PIECE_RANGE):
                        with gr.Row():
                            textbox = gr.Textbox(visible=False)
                            model_train_params_textboxs.append(textbox)
                with gr.Row():
                    for i in range(7 * ONE_PIECE_RANGE, 8 * ONE_PIECE_RANGE):
                        with gr.Row():
                            textbox = gr.Textbox(visible=False)
                            model_train_params_textboxs.append(textbox)
                with gr.Row():
                    for i in range(8 * ONE_PIECE_RANGE, 9 * ONE_PIECE_RANGE):
                        with gr.Row():
                            textbox = gr.Textbox(visible=False)
                            model_train_params_textboxs.append(textbox)
                with gr.Row():
                    for i in range(9 * ONE_PIECE_RANGE, 10 * ONE_PIECE_RANGE):
                        with gr.Row():
                            textbox = gr.Textbox(visible=False)
                            model_train_params_textboxs.append(textbox)


            model_train_button = gr.Button(visible=False)
            model_train_checkbox = gr.Checkbox(visible=False)
            model_train_params_dataframe = gr.Dataframe(visible=False)
            model_train_metrics_dataframe = gr.Dataframe(visible=False)
            # 模型Step 1:在此处添加新的模型相关组件 (初始化)

        # 可视化
        with gr.Accordion("数据可视化"):
            # [绘图]
            with gr.Tab("数据分布图"):
                data_distribution_radio = gr.Radio(visible=False)
                data_distribution_is_rotate = gr.Checkbox(visible=False)
                data_distribution_button = gr.Button(visible=False)

            with gr.Tab("箱线统计图"):
                descriptive_indicators_checkboxgroup = gr.Checkboxgroup(visible=False)
                descriptive_indicators_is_rotate = gr.Checkbox(visible=False)
                descriptive_indicators_button = gr.Button(visible=False)
                descriptive_indicators_dataframe = gr.Dataframe(visible=False)

            with gr.Tab("系数热力图"):
                heatmap_checkboxgroup = gr.Checkboxgroup(visible=False)
                heatmap_is_rotate = gr.Checkbox(visible=False)
                heatmap_button = gr.Button(visible=False)

            # with gr.Tab("主成分分析"):
            #     pca_button = gr.Button(visible=False)
            #     pca_replace_data_button = gr.Button(visible=False)

            with gr.Tab("学习曲线图"):
                learning_curve_checkboxgroup = gr.Checkboxgroup(visible=False)
                learning_curve_button = gr.Button(visible=False)

            with gr.Tab("数据拟合图"):
                data_fit_checkboxgroup = gr.Checkboxgroup(visible=False)
                data_fit_button = gr.Button(visible=False)

            with gr.Tab("特征蜂群图"):
                shap_beeswarm_radio = gr.Radio(visible=False)
                shap_beeswarm_type = gr.Radio(visible=False)
                shap_beeswarm_button = gr.Button(visible=False)

            with gr.Tab("特征瀑布图"):
                waterfall_radio = gr.Radio(visible=False)
                waterfall_number = gr.Slider(visible=False)
                waterfall_button = gr.Button(visible=False)

            with gr.Tab("特征力图"):
                force_radio = gr.Radio(visible=False)
                force_number = gr.Slider(visible=False)
                force_button = gr.Button(visible=False)

            with gr.Tab("特征依赖图"):
                dependence_radio = gr.Radio(visible=False)
                dependence_col = gr.Radio(visible=False)
                dependence_button = gr.Button(visible=False)

            # 绘图Step 1:在这里添加新的绘图相关组件 (初始化)

            # 通用可视化组件

            legend_labels_textboxs = []
            with gr.Accordion("图例"):
                with gr.Row():
                    for i in range(StaticValue.MAX_NUM):
                        with gr.Row():
                            label = gr.Textbox(visible=False)
                            legend_labels_textboxs.append(label)

            with gr.Accordion("坐标轴"):
                with gr.Row():
                    title_name_textbox = gr.Textbox(visible=False)
                    x_label_textbox = gr.Textbox(visible=False)
                    y_label_textbox = gr.Textbox(visible=False)

            colorpickers = []
            color_textboxs = []
            with gr.Accordion("颜色"):
                with gr.Row():
                    for i in range(StaticValue.MAX_NUM):
                        with gr.Row():
                            colorpicker = gr.ColorPicker(visible=False)
                            colorpickers.append(colorpicker)
                            color_textbox = gr.Textbox(visible=False)
                            color_textboxs.append(color_textbox)

            draw_plot = gr.Plot(visible=False)
            draw_file = gr.File(visible=False)

    with gr.Tab("文字说明"):
        notes = gr.Markdown(Dataset.get_notes(), visible=True)

    '''
        监听事件函数
    '''

    # 模型训练传入的组件 (训练模型)
    def get_train_model_input():
        return model_train_params_textboxs + [
            # [模型]
            model_optimize_radio,
            train_size_textbox,
            linear_regression_model_radio,
            naive_bayes_classification_model_radio
            # 模型Step 2:在这里添加新的模型额外组件
        ]

    # 模型基础绘制传入的组件 (图例+标题+x标签+y标签+颜色)
    def get_draw_general_input():
        return [
            # [绘图]
            title_name_textbox,
            x_label_textbox,
            y_label_textbox
        ] + colorpickers + \
            color_textboxs + \
            legend_labels_textboxs + [
                learning_curve_checkboxgroup,
                shap_beeswarm_radio,
                shap_beeswarm_type,
                data_fit_checkboxgroup,
                waterfall_radio,
                waterfall_number,
                force_radio,
                force_number,
                dependence_radio,
                dependence_col,
                data_distribution_radio,
                data_distribution_is_rotate,
                descriptive_indicators_is_rotate,
                descriptive_indicators_checkboxgroup,
                heatmap_checkboxgroup, heatmap_is_rotate
                # 绘图Step 2:在这里添加新的绘图组件
            ]


    # 获取组件监听事件函数的输出
    def get_outputs():
        gr_set = {
            choose_custom_dataset_file,
            display_dataset_dataframe,
            display_total_col_num_text,
            display_total_row_num_text,
            display_na_list_text,
            del_all_na_col_button,
            display_duplicate_num_text,
            del_duplicate_button,
            del_col_checkboxgroup,
            del_col_button,
            remain_row_slider,
            remain_row_button,
            encode_label_button,
            display_encode_label_dataframe,
            encode_label_checkboxgroup,
            data_type_dataframe,
            change_data_type_to_float_button,
            standardize_data_checkboxgroup,
            standardize_data_button,
            select_as_y_radio,
            train_size_textbox,
            model_optimize_radio,
            model_train_button,
            model_train_checkbox,
            model_train_params_dataframe,
            model_train_metrics_dataframe,
            select_as_model_radio,
            choose_assign_radio,
            display_dataset,
            draw_plot,
            draw_file,
            title_name_textbox,
            x_label_textbox,
            y_label_textbox,

            # [模型]
            linear_regression_model_radio,
            naive_bayes_classification_model_radio,
            # 模型Step 7:在这里添加额外的模型组件

            # [绘图]
            heatmap_is_rotate,
            heatmap_checkboxgroup,
            heatmap_button,
            data_distribution_radio,
            data_distribution_is_rotate,
            data_distribution_button,
            descriptive_indicators_checkboxgroup,
            descriptive_indicators_is_rotate,
            descriptive_indicators_dataframe,
            descriptive_indicators_button,
            learning_curve_checkboxgroup,
            learning_curve_button,
            shap_beeswarm_radio,
            shap_beeswarm_type,
            shap_beeswarm_button,
            data_fit_checkboxgroup,
            data_fit_button,
            waterfall_radio,
            waterfall_number,
            waterfall_button,
            force_radio,
            force_number,
            force_button,
            dependence_radio,
            dependence_col,
            dependence_button,
            # 绘图Step 12:在这里添加新的绘图组件
        }

        gr_set.update(set(colorpickers))
        gr_set.update(set(color_textboxs))
        gr_set.update(set(legend_labels_textboxs))

        gr_set.update(set(model_train_params_textboxs))

        return gr_set


    # 选择数据源
    choose_dataset_radio.change(
        fn=choose_dataset,
        inputs=[choose_dataset_radio],
        outputs=get_outputs()
    )
    choose_custom_dataset_file.upload(
        fn=choose_custom_dataset,
        inputs=[choose_custom_dataset_file],
        outputs=get_outputs()
    )

    # 操作数据表

    # 删除所选列
    del_col_button.click(
        fn=del_col,
        inputs=[del_col_checkboxgroup],
        outputs=get_outputs()
    )
    # 保留行
    remain_row_button.click(
        fn=remain_row,
        inputs=[remain_row_slider],
        outputs=get_outputs()
    )
    # 删除所有存在缺失值的列
    del_all_na_col_button.click(
        fn=del_all_na_col,
        outputs=get_outputs()
    )
    # 删除所有重复的行
    del_duplicate_button.click(
        fn=del_duplicate,
        outputs=get_outputs()
    )
    # 字符型列转数值型列
    encode_label_button.click(
        fn=encode_label,
        inputs=[encode_label_checkboxgroup],
        outputs=get_outputs()
    )
    # 将所有数据强制转换为浮点型(除第1列之外)
    change_data_type_to_float_button.click(
        fn=change_data_type_to_float,
        outputs=get_outputs()
    )
    # 标准化数据
    standardize_data_button.click(
        fn=standardize_data,
        inputs=[standardize_data_checkboxgroup],
        outputs=get_outputs()
    )
    # 选择因变量
    select_as_y_radio.change(
        fn=select_as_y,
        inputs=[select_as_y_radio],
        outputs=get_outputs()
    )
    # 选择任务类型(强制转换第1列)
    choose_assign_radio.change(
        fn=choose_assign,
        inputs=[choose_assign_radio],
        outputs=get_outputs()
    )

    # 数据模型
    select_as_model_radio.change(
        fn=select_as_model,
        inputs=[select_as_model_radio],
        outputs=get_outputs()
    )

    # [模型]
    linear_regression_model_radio.change(
        fn=linear_regression_model_radio_change,
        inputs=[linear_regression_model_radio],
        outputs=get_outputs()
    )
    naive_bayes_classification_model_radio.change(
        fn=naive_bayes_classification_model_radio_change,
        inputs=[naive_bayes_classification_model_radio],
        outputs=get_outputs()
    )
    # 模型Step 16:在这里添加新的模型额外组件监听事件

    # 选择超参数优化方法
    model_optimize_radio.select(
        fn=select_model_optimize_radio,
        inputs=[model_optimize_radio],
        outputs=get_outputs()
    )
    # 模型训练
    model_train_button.click(
        fn=train_model,
        inputs=get_train_model_input(),
        outputs=get_outputs()
    )

    # 可视化

    # [绘图]

    data_distribution_button.click(
        fn=data_distribution_first_draw_plot,
        inputs=[
            data_distribution_radio,
            data_distribution_is_rotate
        ],
        outputs=get_outputs()
    )

    descriptive_indicators_button.click(
        fn=descriptive_indicators_first_draw_plot,
        inputs=[
            descriptive_indicators_is_rotate,
            descriptive_indicators_checkboxgroup
        ],
        outputs=get_outputs()
    )

    heatmap_button.click(
        fn=heatmap_first_draw_plot,
        inputs=[
            heatmap_checkboxgroup,
            heatmap_is_rotate
        ],
        outputs=get_outputs()
    )

    learning_curve_button.click(
        fn=learning_curve_first_draw_plot,
        inputs=[
            learning_curve_checkboxgroup
        ],
        outputs=get_outputs()
    )

    shap_beeswarm_button.click(
        fn=shap_beeswarm_first_draw_plot,
        inputs=[
            shap_beeswarm_radio,
            shap_beeswarm_type
        ],
        outputs=get_outputs()
    )

    data_fit_button.click(
        fn=data_fit_first_draw_plot,
        inputs=[
            data_fit_checkboxgroup
        ],
        outputs=get_outputs()
    )

    waterfall_button.click(
        fn=waterfall_first_draw_plot,
        inputs=[
            waterfall_radio,
            waterfall_number
        ],
        outputs=get_outputs()
    )

    force_button.click(
        fn=force_first_draw_plot,
        inputs=[
            force_radio,
            force_number
        ],
        outputs=get_outputs()
    )

    dependence_button.click(
        fn=dependence_first_draw_plot,
        inputs=[
            dependence_radio,
            dependence_col
        ],
        outputs=get_outputs()
    )
    # 绘图Step 3:在这里添加新的绘图监听事件

    # 可视化通用

    title_name_textbox.blur(
        fn=not_color_text_out_non_first_draw_plot,
        inputs=get_draw_general_input(),
        outputs=get_outputs()
    )

    x_label_textbox.blur(
        fn=not_color_text_out_non_first_draw_plot,
        inputs=get_draw_general_input(),
        outputs=get_outputs()
    )

    y_label_textbox.blur(
        fn=not_color_text_out_non_first_draw_plot,
        inputs=get_draw_general_input(),
        outputs=get_outputs()
    )

    for i in range(StaticValue.MAX_NUM):
        colorpickers[i].blur(
            fn=not_color_text_out_non_first_draw_plot,
            inputs=get_draw_general_input(),
            outputs=get_outputs()
        )

        color_textboxs[i].blur(
            fn=is_color_text_out_non_first_draw_plot,
            inputs=get_draw_general_input(),
            outputs=get_outputs()
        )

        legend_labels_textboxs[i].blur(
            fn=not_color_text_out_non_first_draw_plot,
            inputs=get_draw_general_input(),
            outputs=get_outputs()
        )

# 运行入口
if __name__ == "__main__":
    # 启动程序
    demo.launch()
