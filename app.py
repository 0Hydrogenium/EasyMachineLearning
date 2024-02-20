import copy
import math
import os.path
import random

import gradio as gr
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd

from analysis.bayes_model import *
from analysis.distance_model import *
from analysis.gradient_model import *
from analysis.kernel_model import *
from analysis.shap_model import *
from analysis.tree_model import *
from metrics.calculate_classification_metrics import ClassificationMetrics
from metrics.calculate_regression_metrics import RegressionMetrics
from static.process import *
from analysis.linear_model import *
from visualization.draw_boxplot import draw_boxplot
from visualization.draw_data_fit_total import draw_data_fit_total
from visualization.draw_heat_map import draw_heat_map
from visualization.draw_histogram import draw_histogram
from visualization.draw_learning_curve_total import draw_learning_curve_total
from static.new_class import *

import warnings

warnings.filterwarnings("ignore")


# [模型]
class ChooseModelMetrics:
    @classmethod
    def choose(cls, cur_model):
        if cur_model == MN.linear_regression:
            return RegressionMetrics.get_metrics()
        elif cur_model == MN.polynomial_regression:
            return RegressionMetrics.get_metrics()
        elif cur_model == MN.logistic_regression:
            return ClassificationMetrics.get_metrics()
        elif cur_model == MN.decision_tree_classifier:
            return ClassificationMetrics.get_metrics()
        elif cur_model == MN.random_forest_classifier:
            return ClassificationMetrics.get_metrics()
        elif cur_model == MN.random_forest_regression:
            return RegressionMetrics.get_metrics()
        elif cur_model == MN.xgboost_classifier:
            return ClassificationMetrics.get_metrics()
        elif cur_model == MN.lightGBM_classifier:
            return ClassificationMetrics.get_metrics()
        elif cur_model == MN.gradient_boosting_regression:
            return RegressionMetrics.get_metrics()
        elif cur_model == MN.svm_classifier:
            return ClassificationMetrics.get_metrics()
        elif cur_model == MN.svm_regression:
            return RegressionMetrics.get_metrics()
        elif cur_model == MN.knn_classifier:
            return ClassificationMetrics.get_metrics()
        elif cur_model == MN.knn_regression:
            return RegressionMetrics.get_metrics()
        elif cur_model == MN.naive_bayes_classification:
            return ClassificationMetrics.get_metrics()


# [模型]
class ChooseModelParams:
    @classmethod
    def choose(cls, cur_model):
        if cur_model == MN.linear_regression:
            return LinearRegressionParams.get_params(Dataset.linear_regression_model_type)
        elif cur_model == MN.polynomial_regression:
            return PolynomialRegressionParams.get_params()
        elif cur_model == MN.logistic_regression:
            return LogisticRegressionParams.get_params()
        elif cur_model == MN.decision_tree_classifier:
            return DecisionTreeClassifierParams.get_params()
        elif cur_model == MN.random_forest_classifier:
            return RandomForestClassifierParams.get_params()
        elif cur_model == MN.random_forest_regression:
            return RandomForestRegressionParams.get_params()
        elif cur_model == MN.xgboost_classifier:
            return XgboostClassifierParams.get_params()
        elif cur_model == MN.lightGBM_classifier:
            return LightGBMClassifierParams.get_params()
        elif cur_model == MN.gradient_boosting_regression:
            return GradientBoostingParams.get_params()
        elif cur_model == MN.svm_classifier:
            return SVMClassifierParams.get_params()
        elif cur_model == MN.svm_regression:
            return SVMRegressionParams.get_params()
        elif cur_model == MN.knn_classifier:
            return KNNClassifierParams.get_params()
        elif cur_model == MN.knn_regression:
            return KNNRegressionParams.get_params()
        elif cur_model == MN.naive_bayes_classification:
            return NaiveBayesClassifierParams.get_params(Dataset.naive_bayes_classifier_model_type)


class StaticValue:
    max_num = 20


class FilePath:
    png_base = "./buffer/{}.png"
    excel_base = "./buffer/{}.xlsx"

    # [绘图]
    display_dataset = "current_excel_data"

    data_distribution_plot = "data_distribution_plot"
    descriptive_indicators_plot = "descriptive_indicators_plot"
    heatmap_plot = "heatmap_plot"
    learning_curve_plot = "learning_curve_plot"
    shap_beeswarm_plot = "shap_beeswarm_plot"
    data_fit_plot = "data_fit_plot"
    waterfall_plot = "waterfall_plot"
    force_plot = "force_plot"
    dependence_plot = "dependence_plot"


class MN:  # ModelName
    classification = "classification"
    regression = "regression"

    # [模型]
    linear_regression = "linear regressor"
    polynomial_regression = "polynomial regressor"
    logistic_regression = "logistic regressor"
    decision_tree_classifier = "decision tree classifier"
    random_forest_classifier = "random forest classifier"
    random_forest_regression = "random forest regressor"
    xgboost_classifier = "xgboost classifier"
    lightGBM_classifier = "lightGBM classifier"
    gradient_boosting_regression = "gradient boosting regressor"
    svm_classifier = "svm classifier"
    svm_regression = "svm regressor"
    knn_classifier = "knn classifier"
    knn_regression = "knn regressor"
    naive_bayes_classification = "naive bayes classification"

    # [绘图]
    data_distribution = "data_distribution"
    descriptive_indicators = "descriptive_indicators"
    heatmap = "heatmap"
    learning_curve = "learning_curve"
    shap_beeswarm = "shap_beeswarm"
    data_fit = "data_fit"
    waterfall = "waterfall"
    force = "force"
    dependence = "dependence"


class LN:  # LabelName
    choose_dataset_radio = "选择所需数据源 [必选]"
    display_total_col_num_text = "总列数"
    display_total_row_num_text = "总行数"
    display_na_list_text = "存在缺失值的列"
    del_all_na_col_button = "删除所有存在缺失值的列 [可选]"
    display_duplicate_num_text = "重复的行数"
    del_col_checkboxgroup = "选择所需删除的列"
    del_col_button = "删除 [可选]"
    remain_row_slider = "保留的行数"
    remain_row_button = "保留 [可选]"
    del_duplicate_button = "删除所有重复行 [可选]"
    encode_label_checkboxgroup = "选择所需标签编码的字符型数值列"
    display_encode_label_dataframe = "标签编码信息"
    encode_label_button = "字符型转数值型 [可选]"
    change_data_type_to_float_button = "将所有数据强制转换为浮点型（除第1列以外）[必选]"
    standardize_data_checkboxgroup = "选择所需标准化的列"
    standardize_data_button = "标准化 [可选]"
    select_as_y_radio = "选择因变量 [必选]"
    choose_assign_radio = "选择任务类型（同时会根据任务类型将第1列数据强制转换）[必选]"
    model_optimize_radio = "选择超参数优化方法"
    model_train_button = "训练"
    model_train_params_dataframe = "训练后的模型参数"
    model_train_metrics_dataframe = "训练后的模型指标"
    select_as_model_radio = "选择所需训练的模型"

    # [模型]
    linear_regression_model_radio = "选择线性回归的模型"
    naive_bayes_classification_model_radio = "选择朴素贝叶斯分类的模型"

    title_name_textbox = "标题"
    x_label_textbox = "x 轴名称"
    y_label_textbox = "y 轴名称"
    colors = ["颜色 {}".format(i) for i in range(StaticValue.max_num)]
    labels = ["图例 {}".format(i) for i in range(StaticValue.max_num)]

    # [绘图]
    heatmap_is_rotate = "x轴标签是否旋转"
    heatmap_checkboxgroup = "选择所需绘制系数热力图的列"
    heatmap_button = "绘制系数热力图"
    data_distribution_radio = "选择所需绘制数据分布图的列"
    data_distribution_is_rotate = "x轴标签是否旋转"
    data_distribution_button = "绘制数据分布图"
    descriptive_indicators_checkboxgroup = "选择所需绘制箱线统计图的列"
    descriptive_indicators_is_rotate = "x轴标签是否旋转"
    descriptive_indicators_button = "绘制箱线统计图"
    learning_curve_checkboxgroup = "选择所需绘制学习曲线图的模型"
    learning_curve_button = "绘制学习曲线图"
    shap_beeswarm_radio = "选择所需绘制特征蜂群图的模型"
    shap_beeswarm_type = "选择图像类型"
    shap_beeswarm_button = "绘制特征蜂群图"
    data_fit_checkboxgroup = "选择所需绘制数据拟合图的模型"
    data_fit_button = "绘制数据拟合图"
    waterfall_radio = "选择所需绘制特征瀑布图的模型"
    waterfall_number = "输入相关特征的变量索引"
    waterfall_button = "绘制特征瀑布图"
    force_radio = "选择所需绘制特征力图的模型"
    force_number = "输入相关特征的变量索引"
    force_button = "绘制特征力图"
    dependence_radio = "选择所需绘制特征依赖图的模型"
    dependence_col = "选择相应的列"
    dependence_button = "绘制特征依赖图"

    data_distribution_plot = "数据分布图"
    descriptive_indicators_plot = "箱线统计图"
    heatmap_plot = "系数热力图"
    learning_curve_plot = "学习曲线图"
    shap_beeswarm_plot = "特征蜂群图"
    data_fit_plot = "数据拟合图"
    waterfall_plot = "特征瀑布图"
    force_plot = "特征力图"
    dependence_plot = "特征依赖图"


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

    gr_dict.update(dict(zip(colorpickers, [gr.ColorPicker(visible=False)] * StaticValue.max_num)))
    gr_dict.update(dict(zip(color_textboxs, [gr.Textbox(visible=False)] * StaticValue.max_num)))
    gr_dict.update(dict(zip(legend_labels_textboxs, [gr.Textbox(visible=False)] * StaticValue.max_num)))
    gr_dict.update({title_name_textbox: gr.Textbox(visible=False)})
    gr_dict.update({x_label_textbox: gr.Textbox(visible=False)})
    gr_dict.update({y_label_textbox: gr.Textbox(visible=False)})

    return gr_dict


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
    }

    gr_set.update(set(colorpickers))
    gr_set.update(set(color_textboxs))
    gr_set.update(set(legend_labels_textboxs))

    return gr_set


def get_return(is_visible, extra_gr_dict: dict = None):
    if is_visible:
        gr_dict = {
            display_dataset_dataframe: gr.Dataframe(add_index_into_df(Dataset.data), type="pandas", visible=True),
            display_dataset: gr.File(Dataset.after_get_display_dataset_file(), visible=Dataset.check_display_dataset_file()),
            display_total_col_num_text: gr.Textbox(str(Dataset.get_total_col_num()), visible=True, label=LN.display_total_col_num_text),
            display_total_row_num_text: gr.Textbox(str(Dataset.get_total_row_num()), visible=True, label=LN.display_total_row_num_text),
            display_na_list_text: gr.Textbox(Dataset.get_na_list_str(), visible=True, label=LN.display_na_list_text),
            del_all_na_col_button: gr.Button(LN.del_all_na_col_button, visible=True),
            display_duplicate_num_text: gr.Textbox(str(Dataset.get_duplicate_num()), visible=True, label=LN.display_duplicate_num_text),
            del_duplicate_button: gr.Button(LN.del_duplicate_button, visible=True),
            del_col_checkboxgroup: gr.Checkboxgroup(Dataset.get_col_list(), visible=True, label=LN.del_col_checkboxgroup),
            del_col_button: gr.Button(LN.del_col_button, visible=True),
            remain_row_slider: gr.Slider(0, Dataset.get_max_num(), value=Dataset.get_total_row_num(), step=1, visible=True, label=LN.remain_row_slider),
            remain_row_button: gr.Button(LN.remain_row_button, visible=True),
            encode_label_button: gr.Button(LN.encode_label_button, visible=True),
            encode_label_checkboxgroup: gr.Checkboxgroup(Dataset.get_non_numeric_list(), visible=True, label=LN.encode_label_checkboxgroup),
            display_encode_label_dataframe: gr.Dataframe(visible=False),
            data_type_dataframe: gr.Dataframe(Dataset.get_data_type(), visible=True),
            change_data_type_to_float_button: gr.Button(LN.change_data_type_to_float_button, visible=True),
            select_as_y_radio: gr.Radio(Dataset.get_col_list(), visible=True, label=LN.select_as_y_radio),
            standardize_data_checkboxgroup: gr.Checkboxgroup(Dataset.get_non_standardized_data(), visible=True, label=LN.standardize_data_checkboxgroup),
            standardize_data_button: gr.Button(LN.standardize_data_button, visible=True),
            choose_assign_radio: gr.Radio(Dataset.get_assign_list(), visible=True, label=LN.choose_assign_radio),

            select_as_model_radio: gr.Radio(Dataset.get_model_list(), visible=Dataset.check_before_train(), label=LN.select_as_model_radio),
            model_optimize_radio: gr.Radio(Dataset.get_optimize_list(), visible=Dataset.check_before_train(), label=LN.model_optimize_radio),
            model_train_button: gr.Button(LN.model_train_button, visible=Dataset.check_before_train()),
            model_train_checkbox: gr.Checkbox(Dataset.get_model_container_status(), visible=Dataset.check_select_model(), label=Dataset.get_model_label()),
            model_train_params_dataframe: gr.Dataframe(Dataset.get_model_train_params_dataframe(), type="pandas", visible=Dataset.get_model_container_status()),
            model_train_metrics_dataframe: gr.Dataframe(Dataset.get_model_train_metrics_dataframe(), type="pandas", visible=Dataset.get_model_container_status()),

            draw_plot: gr.Plot(visible=False),
            draw_file: gr.File(visible=False),
            title_name_textbox: gr.Textbox(visible=False),
            x_label_textbox: gr.Textbox(visible=False),
            y_label_textbox: gr.Textbox(visible=False),

            # [模型]
            linear_regression_model_radio: gr.Radio(Dataset.get_linear_regression_model_list(), visible=Dataset.get_linear_regression_mark(), label=LN.linear_regression_model_radio),
            naive_bayes_classification_model_radio: gr.Radio(Dataset.get_naive_bayes_classifier_model_list(), visible=Dataset.get_naive_bayes_classifier_mark(), label=LN.naive_bayes_classification_model_radio),

            # [绘图]
            heatmap_checkboxgroup: gr.Checkboxgroup(Dataset.get_float_col_list(), visible=True, label=LN.heatmap_checkboxgroup),
            heatmap_is_rotate: gr.Checkbox(visible=True, label=LN.heatmap_is_rotate),
            heatmap_button: gr.Button(LN.heatmap_button, visible=True),
            descriptive_indicators_checkboxgroup: gr.Checkboxgroup(Dataset.get_float_col_list(), visible=True, label=LN.descriptive_indicators_checkboxgroup),
            data_distribution_radio: gr.Radio(Dataset.get_str_col_list(), visible=True, label=LN.data_distribution_radio),
            data_distribution_is_rotate: gr.Checkbox(visible=True, label=LN.data_distribution_is_rotate),
            data_distribution_button: gr.Button(LN.data_distribution_button, visible=True),
            descriptive_indicators_is_rotate: gr.Checkbox(visible=True, label=LN.descriptive_indicators_is_rotate),
            descriptive_indicators_dataframe: gr.Dataframe(Dataset.get_descriptive_indicators_df(), type="pandas", visible=Dataset.check_descriptive_indicators_df()),
            descriptive_indicators_button: gr.Button(LN.descriptive_indicators_button, visible=True),
            learning_curve_checkboxgroup: gr.Checkboxgroup(Dataset.get_trained_model_list(), visible=Dataset.check_before_train(), label=LN.learning_curve_checkboxgroup),
            learning_curve_button: gr.Button(LN.learning_curve_button, visible=Dataset.check_before_train()),
            shap_beeswarm_radio: gr.Radio(Dataset.get_trained_model_list(), visible=Dataset.check_before_train(), label=LN.shap_beeswarm_radio),
            shap_beeswarm_type: gr.Radio(Dataset.get_shap_beeswarm_plot_type(), visible=Dataset.check_before_train(), label=LN.shap_beeswarm_type),
            shap_beeswarm_button: gr.Button(LN.shap_beeswarm_button, visible=Dataset.check_before_train()),
            data_fit_checkboxgroup: gr.Checkboxgroup(Dataset.get_trained_model_list(), visible=Dataset.check_before_train(), label=LN.data_fit_checkboxgroup),
            data_fit_button: gr.Button(LN.data_fit_button, visible=Dataset.check_before_train()),
            waterfall_radio: gr.Radio(Dataset.get_trained_model_list(), visible=Dataset.check_before_train(), label=LN.waterfall_radio),
            waterfall_number: gr.Slider(0, Dataset.get_total_row_num(), value=0, step=1, visible=Dataset.check_before_train(), label=LN.waterfall_number),
            waterfall_button: gr.Button(LN.waterfall_button, visible=Dataset.check_before_train()),
            force_radio: gr.Radio(Dataset.get_trained_model_list(), visible=Dataset.check_before_train(), label=LN.force_radio),
            force_number: gr.Slider(0, Dataset.get_total_row_num(), value=0, step=1, visible=Dataset.check_before_train(), label=LN.force_number),
            force_button: gr.Button(LN.force_button, visible=Dataset.check_before_train()),
            dependence_radio: gr.Radio(Dataset.get_trained_model_list(), visible=Dataset.check_before_train(), label=LN.dependence_radio),
            dependence_col: gr.Radio(Dataset.get_col_list(), visible=Dataset.check_before_train(), label=LN.dependence_col),
            dependence_button: gr.Button(LN.dependence_button, visible=Dataset.check_before_train()),

        }

        gr_dict.update(dict(zip(colorpickers, [gr.ColorPicker(visible=False)] * StaticValue.max_num)))
        gr_dict.update(dict(zip(color_textboxs, [gr.Textbox(visible=False)] * StaticValue.max_num)))
        gr_dict.update(dict(zip(legend_labels_textboxs, [gr.Textbox(visible=False)] * StaticValue.max_num)))

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
    }

    gr_dict.update(dict(zip(colorpickers, [gr.ColorPicker(visible=False)] * StaticValue.max_num)))
    gr_dict.update(dict(zip(color_textboxs, [gr.Textbox(visible=False)] * StaticValue.max_num)))
    gr_dict.update(dict(zip(legend_labels_textboxs, [gr.Textbox(visible=False)] * StaticValue.max_num)))

    return gr_dict


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

    linear_regression_model_type = ""
    naive_bayes_classifier_model_type = ""

    container_dict = {
        # [模型]
        MN.linear_regression: Container(),
        MN.polynomial_regression: Container(),
        MN.logistic_regression: Container(),
        MN.decision_tree_classifier: Container(),
        MN.random_forest_classifier: Container(),
        MN.random_forest_regression: Container(),
        MN.xgboost_classifier: Container(),
        MN.lightGBM_classifier: Container(),
        MN.gradient_boosting_regression: Container(),
        MN.svm_classifier: Container(),
        MN.svm_regression: Container(),
        MN.knn_classifier: Container(),
        MN.knn_regression: Container(),
        MN.naive_bayes_classification: Container(),
    }

    visualize = ""

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
        return ["自定义", "Iris Dataset", "Wine Dataset", "Breast Cancer Dataset", "Diabetes Dataset", "California Housing Dataset"]

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
                if not (all(isinstance(x, str) for x in cls.data.iloc[:, 0]) or all(isinstance(x, float) for x in cls.data.iloc[:, 0])):
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
    def train_model(cls, optimize, linear_regression_model_type=None, naive_bayes_classifier_model_type=None):
        optimize = cls.get_optimize_name_mapping()[optimize]

        data_copy = cls.data
        if cls.assign == MN.classification:
            data_copy = cls.encode_label([cls.data.columns.values[0]], True)

        x_train, x_test, y_train, y_test = train_test_split(
            data_copy.values[:, 1:],
            data_copy.values[:, :1],
            random_state=Config.RANDOM_STATE,
            train_size=0.8
        )
        container = Container(x_train, y_train, x_test, y_test, optimize)

        # [模型]
        if cls.cur_model == MN.linear_regression:
            cls.linear_regression_model_type = cls.get_linear_regression_model_name_mapping()[linear_regression_model_type]
            container = linear_regression(container, cls.linear_regression_model_type)
        elif cls.cur_model == MN.polynomial_regression:
            container = polynomial_regression(container)
        elif cls.cur_model == MN.logistic_regression:
            container = logistic_regression(container)
        elif cls.cur_model == MN.decision_tree_classifier:
            container = decision_tree_classifier(container)
        elif cls.cur_model == MN.random_forest_classifier:
            container = random_forest_classifier(container)
        elif cls.cur_model == MN.random_forest_regression:
            container = random_forest_regression(container)
        elif cls.cur_model == MN.xgboost_classifier:
            container = xgboost_classifier(container)
        elif cls.cur_model == MN.lightGBM_classifier:
            container = lightGBM_classifier(container)
        elif cls.cur_model == MN.gradient_boosting_regression:
            container = gradient_boosting_regression(container)
        elif cls.cur_model == MN.svm_classifier:
            container = svm_classifier(container)
        elif cls.cur_model == MN.svm_regression:
            container = svm_regression(container)
        elif cls.cur_model == MN.knn_classifier:
            container = knn_classifier(container)
        elif cls.cur_model == MN.knn_regression:
            container = knn_regression(container)
        elif cls.cur_model == MN.naive_bayes_classification:
            cls.naive_bayes_classifier_model_type = cls.get_naive_bayes_classifier_model_name_mapping()[naive_bayes_classifier_model_type]
            container = naive_bayes_classification(container, cls.naive_bayes_classifier_model_type)

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

    # [模型]
    @classmethod
    def get_model_chinese_name(cls):
        return ["线性回归", "多项式回归", "逻辑斯谛分类", "决策树分类", "随机森林分类", "随机森林回归", "XGBoost分类", "LightGBM分类",
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
    def draw_plot(cls, select_model, color_list: list, label_list: list, name: str, x_label: str, y_label: str, is_default: bool):
        # [绘图]
        if cls.visualize == MN.learning_curve:
            return cls.draw_learning_curve_plot(select_model, color_list, label_list, name, x_label, y_label, is_default)
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
            return cls.draw_data_distribution_plot(select_model, color_list, label_list, name, x_label, y_label, is_default)
        elif cls.visualize == MN.descriptive_indicators:
            return cls.draw_descriptive_indicators_plot(select_model, color_list, label_list, name, x_label, y_label, is_default)
        elif cls.visualize == MN.heatmap:
            return cls.draw_heatmap_plot(select_model, color_list, label_list, name, x_label, y_label, is_default)

    @classmethod
    def draw_heatmap_plot(cls, select_model, color_list: list, label_list: list, name: str, x_label: str, y_label: str, is_default: bool):
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
    def draw_descriptive_indicators_plot(cls, select_model, color_list: list, label_list: list, name: str, x_label: str, y_label: str, is_default: bool):
        color_cur_list = [Config.COLORS[random.randint(0, 11)]]*3 if is_default else color_list
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
    def draw_data_distribution_plot(cls, select_model, color_list: list, label_list: list, name: str, x_label: str, y_label: str, is_default: bool):
        cur_col = select_model.get_data_distribution_col()

        color_cur_list = [Config.COLORS[random.randint(0, 11)]] if is_default else color_list
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
    def draw_dependence_plot(cls, select_model, color_list: list, label_list: list, name: str, x_label: str, y_label: str, is_default: bool):
        model_name = select_model.get_models()

        model_name = cls.get_model_name_mapping_reverse()[model_name]
        container = cls.container_dict[model_name]

        # color_cur_list = Config.COLORS if is_default else color_list
        # label_cur_list = [x for x in learning_curve_dict.keys()] if is_default else label_list
        # x_cur_label = "Train Sizes" if is_default else x_label
        # y_cur_label = "Accuracy" if is_default else y_label
        cur_name = "" if is_default else name

        paint_object = PaintObject()
        # paint_object.set_color_cur_list(color_cur_list)
        # paint_object.set_label_cur_list(label_cur_list)
        # paint_object.set_x_cur_label(x_cur_label)
        # paint_object.set_y_cur_label(y_cur_label)
        paint_object.set_name(cur_name)

        return draw_dependence(container.get_model(), container.x_train, cls.data.columns.values.tolist()[1:], select_model.get_dependence_col(), paint_object)

    @classmethod
    def draw_force_plot(cls, select_model, color_list: list, label_list: list, name: str, x_label: str, y_label: str, is_default: bool):
        model_name = select_model.get_models()

        model_name = cls.get_model_name_mapping_reverse()[model_name]
        container = cls.container_dict[model_name]

        # color_cur_list = Config.COLORS if is_default else color_list
        # label_cur_list = [x for x in learning_curve_dict.keys()] if is_default else label_list
        # x_cur_label = "Train Sizes" if is_default else x_label
        # y_cur_label = "Accuracy" if is_default else y_label
        cur_name = "" if is_default else name

        paint_object = PaintObject()
        # paint_object.set_color_cur_list(color_cur_list)
        # paint_object.set_label_cur_list(label_cur_list)
        # paint_object.set_x_cur_label(x_cur_label)
        # paint_object.set_y_cur_label(y_cur_label)
        paint_object.set_name(cur_name)

        return draw_force(container.get_model(), container.x_train, cls.data.columns.values.tolist()[1:], select_model.get_force_number(), paint_object)

    @classmethod
    def draw_waterfall_plot(cls, select_model, color_list: list, label_list: list, name: str, x_label: str, y_label: str, is_default: bool):
        model_name = select_model.get_models()

        model_name = cls.get_model_name_mapping_reverse()[model_name]
        container = cls.container_dict[model_name]

        # color_cur_list = Config.COLORS if is_default else color_list
        # label_cur_list = [x for x in learning_curve_dict.keys()] if is_default else label_list
        # x_cur_label = "Train Sizes" if is_default else x_label
        # y_cur_label = "Accuracy" if is_default else y_label
        cur_name = "" if is_default else name

        paint_object = PaintObject()
        # paint_object.set_color_cur_list(color_cur_list)
        # paint_object.set_label_cur_list(label_cur_list)
        # paint_object.set_x_cur_label(x_cur_label)
        # paint_object.set_y_cur_label(y_cur_label)
        paint_object.set_name(cur_name)

        return draw_waterfall(container.get_model(), container.x_train, cls.data.columns.values.tolist()[1:], select_model.get_waterfall_number(), paint_object)

    @classmethod
    def draw_learning_curve_plot(cls, select_model, color_list: list, label_list: list, name: str, x_label: str, y_label: str, is_default: bool):
        cur_dict = {}

        model_list = select_model.get_models()

        for model_name in model_list:
            model_name = cls.get_model_name_mapping_reverse()[model_name]
            cur_dict[model_name] = cls.container_dict[model_name].get_learning_curve_values()

        color_cur_list = Config.COLORS if is_default else color_list
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
    def draw_shap_beeswarm_plot(cls, select_model, color_list: list, label_list: list, name: str, x_label: str, y_label: str, is_default: bool):
        model_name = select_model.get_models()

        model_name = cls.get_model_name_mapping_reverse()[model_name]
        container = cls.container_dict[model_name]

        # color_cur_list = Config.COLORS if is_default else color_list
        # label_cur_list = [x for x in learning_curve_dict.keys()] if is_default else label_list
        # x_cur_label = "Train Sizes" if is_default else x_label
        # y_cur_label = "Accuracy" if is_default else y_label
        cur_name = "" if is_default else name

        paint_object = PaintObject()
        # paint_object.set_color_cur_list(color_cur_list)
        # paint_object.set_label_cur_list(label_cur_list)
        # paint_object.set_x_cur_label(x_cur_label)
        # paint_object.set_y_cur_label(y_cur_label)
        paint_object.set_name(cur_name)

        return draw_shap_beeswarm(container.get_model(), container.x_train, cls.data.columns.values.tolist()[1:], select_model.get_beeswarm_plot_type(), paint_object)

    @classmethod
    def draw_data_fit_plot(cls, select_model, color_list: list, label_list: list, name: str, x_label: str, y_label: str, is_default: bool):
        cur_dict = {}

        model_list = select_model.get_models()

        for model_name in model_list:
            model_name = cls.get_model_name_mapping_reverse()[model_name]
            cur_dict[model_name] = cls.container_dict[model_name].get_data_fit_values()

        color_cur_list = Config.COLORS if is_default else color_list
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
        return True if cls.cur_model == MN.linear_regression else False

    @classmethod
    def get_naive_bayes_classifier_mark(cls):
        return True if cls.cur_model == MN.naive_bayes_classification else False

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
            data_copy.iloc[:, 0] = data_copy.iloc[:, 0].astype(str)
        else:
            data_copy.iloc[:, 0] = data_copy.iloc[:, 0].astype(float)

        cls.data = data_copy
        cls.change_data_type_to_float()

    @classmethod
    def colorpickers_change(cls, paint_object):
        cur_num = paint_object.get_color_cur_num()

        true_list = [gr.ColorPicker(paint_object.get_color_cur_list()[i], visible=True, label=LN.colors[i]) for i in range(cur_num)]

        return true_list + [gr.ColorPicker(visible=False)] * (StaticValue.max_num - cur_num)

    @classmethod
    def color_textboxs_change(cls, paint_object):
        cur_num = paint_object.get_color_cur_num()

        true_list = [gr.Textbox(paint_object.get_color_cur_list()[i], visible=True, show_label=False) for i in range(cur_num)]

        return true_list + [gr.Textbox(visible=False)] * (StaticValue.max_num - cur_num)

    @classmethod
    def labels_change(cls, paint_object):
        cur_num = paint_object.get_label_cur_num()

        true_list = [gr.Textbox(paint_object.get_label_cur_list()[i], visible=True, label=LN.labels[i]) for i in range(cur_num)]

        return true_list + [gr.Textbox(visible=False)] * (StaticValue.max_num - cur_num)

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
    def check_train_model(cls, optimize):
        if cls.cur_model == "":
            gr.Warning("请选择所需训练的模型")
            return True
        if not optimize:
            gr.Warning("请选择超参数优化方法")
            return True
        return False

    @classmethod
    def error_return_train(cls):
        return get_return(True)

    @classmethod
    def check_train_model_other_related(cls, linear_regression_model_type, naive_bayes_classifier_model_type):
        if cls.cur_model == MN.linear_regression:
            if not linear_regression_model_type:
                gr.Warning("请选择线性回归对应的模型")
                return True
        elif cls.cur_model == MN.naive_bayes_classification:
            if not naive_bayes_classifier_model_type:
                gr.Warning("请选择朴素贝叶斯对应的模型")
                return True
        return False

    @classmethod
    def check_cur_dict(cls, cur_dict):
        if not cur_dict:
            gr.Warning("请选择绘图所需的模型")
            return True
        return False


def choose_assign(assign: str):
    Dataset.choose_assign(assign)

    return get_return(True)


def select_as_model(model_name: str):
    Dataset.select_as_model(model_name)

    return get_return(True)


# [绘图]
def heatmap_first_draw_plot(*inputs):
    Dataset.visualize = MN.heatmap
    return before_train_first_draw_plot(inputs)


def descriptive_indicators_first_draw_plot(*inputs):
    Dataset.visualize = MN.descriptive_indicators
    return before_train_first_draw_plot(inputs)


def data_distribution_first_draw_plot(*inputs):
    Dataset.visualize = MN.data_distribution
    return before_train_first_draw_plot(inputs)


def dependence_first_draw_plot(*inputs):
    Dataset.visualize = MN.dependence
    return first_draw_plot(inputs)


def force_first_draw_plot(*inputs):
    Dataset.visualize = MN.force
    return first_draw_plot(inputs)


def waterfall_first_draw_plot(*inputs):
    Dataset.visualize = MN.waterfall
    return first_draw_plot(inputs)


def data_fit_first_draw_plot(*inputs):
    Dataset.visualize = MN.data_fit
    return first_draw_plot(inputs)


def shap_beeswarm_first_draw_plot(*inputs):
    Dataset.visualize = MN.shap_beeswarm
    return first_draw_plot(inputs)


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

    cur_plt, paint_object = Dataset.draw_plot(select_model, color_list, label_list, name, x_label, y_label, True)

    return first_draw_plot_with_non_first_draw_plot(cur_plt, paint_object)


def out_non_first_draw_plot(*inputs):
    return non_first_draw_plot(inputs)


def non_first_draw_plot(inputs):
    name = inputs[0]
    x_label = inputs[1]
    y_label = inputs[2]
    color_list = list(inputs[3: StaticValue.max_num+3])
    label_list = list(inputs[StaticValue.max_num+3: 2*StaticValue.max_num+3])
    start_index = 2*StaticValue.max_num+3

    select_model = SelectModel()

    # 绘图
    if Dataset.visualize == MN.learning_curve:
        select_model.set_models(inputs[start_index+0])
        select_model.set_beeswarm_plot_type(inputs[start_index+1])
    elif Dataset.visualize == MN.shap_beeswarm:
        select_model.set_models(inputs[start_index+2])
    elif Dataset.visualize == MN.data_fit:
        select_model.set_models(inputs[start_index+3])
    elif Dataset.visualize == MN.waterfall:
        select_model.set_models(inputs[start_index+4])
        select_model.set_waterfall_number(inputs[start_index+5])
    elif Dataset.visualize == MN.force:
        select_model.set_models(inputs[start_index+6])
        select_model.set_force_number(inputs[start_index+7])
    elif Dataset.visualize == MN.dependence:
        select_model.set_models(inputs[start_index+8])
        select_model.set_dependence_col(inputs[start_index+9])
    elif Dataset.visualize == MN.data_distribution:
        select_model.set_data_distribution_col(inputs[start_index+10])
        select_model.set_data_distribution_is_rotate(inputs[start_index+11])
    elif Dataset.visualize == MN.descriptive_indicators:
        select_model.set_descriptive_indicators_is_rotate(inputs[start_index+12])
        select_model.set_descriptive_indicators_col(inputs[start_index+13])
    elif Dataset.visualize == MN.descriptive_indicators:
        select_model.set_heatmap_col(inputs[start_index+14])
        select_model.set_heatmap_is_rotate(inputs[start_index+15])

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
        extra_gr_dict.update({descriptive_indicators_dataframe: gr.Dataframe(Dataset.get_descriptive_indicators_df(), type="pandas", visible=Dataset.check_descriptive_indicators_df())})
    elif Dataset.visualize == MN.heatmap:
        cur_plt.savefig(FilePath.png_base.format(FilePath.heatmap_plot), dpi=300)
        extra_gr_dict.update({draw_plot: gr.Plot(cur_plt, visible=True, label=LN.heatmap_plot)})

    extra_gr_dict.update(dict(zip(colorpickers, Dataset.colorpickers_change(paint_object))))
    extra_gr_dict.update(dict(zip(color_textboxs, Dataset.color_textboxs_change(paint_object))))
    extra_gr_dict.update(dict(zip(legend_labels_textboxs, Dataset.labels_change(paint_object))))
    extra_gr_dict.update({title_name_textbox: gr.Textbox(paint_object.get_name(), visible=True, label=LN.title_name_textbox)})
    extra_gr_dict.update({x_label_textbox: gr.Textbox(paint_object.get_x_cur_label(), visible=True, label=LN.x_label_textbox)})
    extra_gr_dict.update({y_label_textbox: gr.Textbox(paint_object.get_y_cur_label(), visible=True, label=LN.y_label_textbox)})

    return get_return_extra(True, extra_gr_dict)


# [模型]
def train_model(optimize, linear_regression_model_type, naive_bayes_classifier_model_type):
    if Dataset.check_train_model(optimize):
        return Dataset.error_return_train()

    if Dataset.check_train_model_other_related(linear_regression_model_type, naive_bayes_classifier_model_type):
        return Dataset.error_return_train()

    Dataset.train_model(optimize, linear_regression_model_type, naive_bayes_classifier_model_type)

    return get_return(True)


def select_as_y(col: str):
    Dataset.select_as_y(col)

    return get_return(True)


def standardize_data(col_list: list):
    Dataset.standardize_data(col_list)

    return get_return(True)


def change_data_type_to_float():
    Dataset.change_data_type_to_float()

    return get_return(True)


def encode_label(col_list: list):
    Dataset.encode_label(col_list)

    return get_return(True, {display_encode_label_dataframe: gr.Dataframe(Dataset.get_str2int_mappings_df(), type="pandas", visible=True, label=LN.display_encode_label_dataframe)})


def del_duplicate():
    Dataset.del_duplicate()

    return get_return(True)


def del_all_na_col():
    Dataset.del_all_na_col()

    return get_return(True)


def remain_row(num):
    Dataset.remain_row(num)

    return get_return(True)


def del_col(col_list: list):
    Dataset.del_col(col_list)

    return get_return(True)


def add_index_into_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    index_df = pd.DataFrame([x for x in range(len(df))], columns=["[*index]"])

    return pd.concat([index_df, df], axis=1)


def choose_dataset(file: str):
    if file == "自定义":
        Dataset.clear()

        return get_return(False)

    df = load_data(file)
    Dataset.update(file, df)

    return get_return(True, {choose_custom_dataset_file: gr.File(visible=False)})


def choose_custom_dataset(file: str):
    df = load_custom_data(file)
    Dataset.update(file, df)

    return get_return(True, {choose_custom_dataset_file: gr.File(Dataset.file, visible=True)})


with gr.Blocks(js=Config.JS_0) as demo:
    '''
        组件
    '''

    with gr.Tab("机器学习"):
        # 选择数据源
        with gr.Accordion("数据源"):
            with gr.Group():
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
            model_optimize_radio = gr.Radio(visible=False)
            model_train_button = gr.Button(visible=False)
            model_train_checkbox = gr.Checkbox(visible=False)
            model_train_params_dataframe = gr.Dataframe(visible=False)
            model_train_metrics_dataframe = gr.Dataframe(visible=False)

        # 可视化
        with gr.Accordion("数据可视化"):
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

            legend_labels_textboxs = []
            with gr.Accordion("图例"):
                with gr.Row():
                    for i in range(StaticValue.max_num):
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
                    for i in range(StaticValue.max_num):
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
        监听事件
    '''

    # 选择数据源
    choose_dataset_radio.change(fn=choose_dataset, inputs=[choose_dataset_radio], outputs=get_outputs())
    choose_custom_dataset_file.upload(fn=choose_custom_dataset, inputs=[choose_custom_dataset_file], outputs=get_outputs())

    # 操作数据表

    # 删除所选列
    del_col_button.click(fn=del_col, inputs=[del_col_checkboxgroup], outputs=get_outputs())
    # 保留行
    remain_row_button.click(fn=remain_row, inputs=[remain_row_slider], outputs=get_outputs())
    # 删除所有存在缺失值的列
    del_all_na_col_button.click(fn=del_all_na_col, outputs=get_outputs())
    # 删除所有重复的行
    del_duplicate_button.click(fn=del_duplicate, outputs=get_outputs())
    # 字符型列转数值型列
    encode_label_button.click(fn=encode_label, inputs=[encode_label_checkboxgroup], outputs=get_outputs())
    # 将所有数据强制转换为浮点型(除第1列之外)
    change_data_type_to_float_button.click(fn=change_data_type_to_float, outputs=get_outputs())
    # 标准化数据
    standardize_data_button.click(fn=standardize_data, inputs=[standardize_data_checkboxgroup], outputs=get_outputs())
    # 选择因变量
    select_as_y_radio.change(fn=select_as_y, inputs=[select_as_y_radio], outputs=get_outputs())
    # 选择任务类型(强制转换第1列)
    choose_assign_radio.change(fn=choose_assign, inputs=[choose_assign_radio], outputs=get_outputs())

    # 数据模型
    select_as_model_radio.change(fn=select_as_model, inputs=[select_as_model_radio], outputs=get_outputs())

    # [模型]
    model_train_button.click(fn=train_model, inputs=[model_optimize_radio, linear_regression_model_radio, naive_bayes_classification_model_radio], outputs=get_outputs())

    # [绘图]

    # 可视化
    data_distribution_button.click(fn=data_distribution_first_draw_plot, inputs=[data_distribution_radio] + [data_distribution_is_rotate], outputs=get_outputs())
    descriptive_indicators_button.click(fn=descriptive_indicators_first_draw_plot, inputs=[descriptive_indicators_is_rotate] + [descriptive_indicators_checkboxgroup], outputs=get_outputs())
    heatmap_button.click(fn=heatmap_first_draw_plot, inputs=[heatmap_checkboxgroup] + [heatmap_is_rotate], outputs=get_outputs())
    learning_curve_button.click(fn=learning_curve_first_draw_plot, inputs=[learning_curve_checkboxgroup], outputs=get_outputs())
    shap_beeswarm_button.click(fn=shap_beeswarm_first_draw_plot, inputs=[shap_beeswarm_radio] + [shap_beeswarm_type], outputs=get_outputs())
    data_fit_button.click(fn=data_fit_first_draw_plot, inputs=[data_fit_checkboxgroup], outputs=get_outputs())
    waterfall_button.click(fn=waterfall_first_draw_plot, inputs=[waterfall_radio] + [waterfall_number], outputs=get_outputs())
    force_button.click(fn=force_first_draw_plot, inputs=[force_radio] + [force_number], outputs=get_outputs())
    dependence_button.click(fn=dependence_first_draw_plot, inputs=[dependence_radio] + [dependence_col], outputs=get_outputs())

    title_name_textbox.blur(fn=out_non_first_draw_plot, inputs=[title_name_textbox] + [x_label_textbox] + [y_label_textbox] + colorpickers + legend_labels_textboxs
                                                               + [learning_curve_checkboxgroup] + [shap_beeswarm_radio] + [shap_beeswarm_type] + [data_fit_checkboxgroup] + [waterfall_radio] + [waterfall_number]
                                                               + [force_radio] + [force_number] + [dependence_radio] + [dependence_col] + [data_distribution_radio] + [data_distribution_is_rotate]
                            + [descriptive_indicators_is_rotate] + [descriptive_indicators_checkboxgroup] + [heatmap_checkboxgroup] + [heatmap_is_rotate], outputs=get_outputs())

    x_label_textbox.blur(fn=out_non_first_draw_plot, inputs=[title_name_textbox] + [x_label_textbox] + [y_label_textbox] + colorpickers + legend_labels_textboxs
                                                            + [learning_curve_checkboxgroup] + [shap_beeswarm_radio] + [shap_beeswarm_type] + [data_fit_checkboxgroup] + [waterfall_radio] + [waterfall_number]
                                                            + [force_radio] + [force_number] + [dependence_radio] + [dependence_col] + [data_distribution_radio] + [data_distribution_is_rotate]
                         + [descriptive_indicators_is_rotate] + [descriptive_indicators_checkboxgroup] + [heatmap_checkboxgroup] + [heatmap_is_rotate], outputs=get_outputs())

    y_label_textbox.blur(fn=out_non_first_draw_plot, inputs=[title_name_textbox] + [x_label_textbox] + [y_label_textbox] + colorpickers + legend_labels_textboxs
                                                            + [learning_curve_checkboxgroup] + [shap_beeswarm_radio] + [shap_beeswarm_type] + [data_fit_checkboxgroup] + [waterfall_radio] + [waterfall_number]
                                                            + [force_radio] + [force_number] + [dependence_radio] + [dependence_col] + [data_distribution_radio] + [data_distribution_is_rotate]
                         + [descriptive_indicators_is_rotate] + [descriptive_indicators_checkboxgroup] + [heatmap_checkboxgroup] + [heatmap_is_rotate], outputs=get_outputs())

    for i in range(StaticValue.max_num):
        colorpickers[i].blur(fn=out_non_first_draw_plot, inputs=[title_name_textbox] + [x_label_textbox] + [y_label_textbox] + colorpickers + legend_labels_textboxs
                                                                + [learning_curve_checkboxgroup] + [shap_beeswarm_radio] + [shap_beeswarm_type] + [data_fit_checkboxgroup] + [waterfall_radio] + [waterfall_number]
                                                                + [force_radio] + [force_number] + [dependence_radio] + [dependence_col] + [data_distribution_radio] + [data_distribution_is_rotate]
                             + [descriptive_indicators_is_rotate] + [descriptive_indicators_checkboxgroup] + [heatmap_checkboxgroup] + [heatmap_is_rotate], outputs=get_outputs())

        color_textboxs[i].blur(fn=out_non_first_draw_plot, inputs=[title_name_textbox] + [x_label_textbox] + [y_label_textbox] + color_textboxs + legend_labels_textboxs
                                                                  + [learning_curve_checkboxgroup] + [shap_beeswarm_radio] + [shap_beeswarm_type] + [data_fit_checkboxgroup] + [waterfall_radio] + [waterfall_number]
                                                                  + [force_radio] + [force_number] + [dependence_radio] + [dependence_col] + [data_distribution_radio] + [data_distribution_is_rotate]
                               + [descriptive_indicators_is_rotate] + [descriptive_indicators_checkboxgroup] + [heatmap_checkboxgroup] + [heatmap_is_rotate], outputs=get_outputs())

        legend_labels_textboxs[i].blur(fn=out_non_first_draw_plot, inputs=[title_name_textbox] + [x_label_textbox] + [y_label_textbox] + colorpickers + legend_labels_textboxs
                                                                          + [learning_curve_checkboxgroup] + [shap_beeswarm_radio] + [shap_beeswarm_type] + [data_fit_checkboxgroup] + [waterfall_radio] + [waterfall_number]
                                                                          + [force_radio] + [force_number] + [dependence_radio] + [dependence_col] + [data_distribution_radio] + [data_distribution_is_rotate]
                                       + [descriptive_indicators_is_rotate] + [descriptive_indicators_checkboxgroup] + [heatmap_checkboxgroup] + [heatmap_is_rotate], outputs=get_outputs())

if __name__ == "__main__":
    demo.launch()
