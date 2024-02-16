import copy
import os.path

import gradio as gr
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd

from analysis.shap_model import shap_calculate
from static.process import *
from analysis.linear_model import *
from visualization.draw_learning_curve_total import draw_learning_curve_total
from static.paint import *

import warnings

warnings.filterwarnings("ignore")


class Container:
    def __init__(self, x_train=None, y_train=None, x_test=None, y_test=None, hyper_params_optimize=None):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.hyper_params_optimize = hyper_params_optimize
        self.info = dict()
        self.y_pred = None
        self.train_sizes = None
        self.train_scores_mean = None
        self.train_scores_std = None
        self.test_scores_mean = None
        self.test_scores_std = None
        self.status = None
        self.model = None

    def set_info(self, info: dict):
        self.info = info

    def set_y_pred(self, y_pred):
        self.y_pred = y_pred

    def get_learning_curve_values(self):
        return [
            self.train_sizes,
            self.train_scores_mean,
            self.train_scores_std,
            self.test_scores_mean,
            self.test_scores_std
        ]

    def set_learning_curve_values(self, train_sizes, train_scores_mean, train_scores_std, test_scores_mean, test_scores_std):
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


class StaticValue:
    max_num = 10


class FilePath:
    png_base = "./buffer/{}.png"
    excel_base = "./buffer/{}.xlsx"

    # [绘图]
    display_dataset = "current_excel_data"
    learning_curve_train_plot = "learning_curve_train_plot"
    learning_curve_validation_plot = "learning_curve_validation_plot"
    shap_beeswarm_plot = "shap_beeswarm_plot"


class MN:  # ModelName
    classification = "classification"
    regression = "regression"

    linear_regression = "linear_regression"
    polynomial_regression = "polynomial_regression"
    logistic_regression = "logistic_regression"

    # [绘图]
    learning_curve_train = "learning_curve_train"
    learning_curve_validation = "learning_curve_validation"
    shap_beeswarm = "shap_beeswarm"


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
    linear_regression_model_radio = "选择线性回归的模型"
    model_optimize_radio = "选择超参数优化方法"
    model_train_button = "训练"
    select_as_model_radio = "选择所需训练的模型"

    title_name_textbox = "标题"
    x_label_textbox = "x 轴名称"
    y_label_textbox = "y 轴名称"
    colors = ["颜色 {}".format(i) for i in range(StaticValue.max_num)]
    labels = ["图例 {}".format(i) for i in range(StaticValue.max_num)]

    # [绘图]
    learning_curve_checkboxgroup = "选择所需绘制学习曲线的模型"
    learning_curve_train_button = "绘制训练集学习曲线"
    learning_curve_validation_button = "绘制验证集学习曲线"
    shap_beeswarm_radio = "选择所需绘制蜂群特征图的模型"
    shap_beeswarm_button = "绘制蜂群特征图"

    learning_curve_train_plot = "训练集学习曲线"
    learning_curve_validation_plot = "验证集学习曲线"
    shap_beeswarm_plot = "蜂群特征图"


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
        linear_regression_model_radio,
        model_optimize_radio,
        model_train_button,
        model_train_checkbox,
        select_as_model_radio,
        choose_assign_radio,
        display_dataset,
        draw_plot,
        draw_file,
        title_name_textbox,
        x_label_textbox,
        y_label_textbox,

        # [绘图]
        learning_curve_checkboxgroup,
        learning_curve_train_button,
        learning_curve_validation_button,
        shap_beeswarm_radio,
        shap_beeswarm_button,
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

            linear_regression_model_radio: gr.Radio(Dataset.get_linear_regression_model_list(), visible=Dataset.get_linear_regression_mark(), label=LN.linear_regression_model_radio),

            model_train_button: gr.Button(LN.model_train_button, visible=Dataset.check_before_train()),
            model_train_checkbox: gr.Checkbox(Dataset.get_model_container_status(), visible=Dataset.check_select_model(), label=Dataset.get_model_label()),

            draw_plot: gr.Plot(visible=False),
            draw_file: gr.File(visible=False),
            title_name_textbox: gr.Textbox(visible=False),
            x_label_textbox: gr.Textbox(visible=False),
            y_label_textbox: gr.Textbox(visible=False),

            # [绘图]
            learning_curve_checkboxgroup: gr.Checkboxgroup(Dataset.get_trained_model_list(), visible=Dataset.check_before_train(), label=LN.learning_curve_checkboxgroup),
            learning_curve_train_button: gr.Button(LN.learning_curve_train_button, visible=Dataset.check_before_train()),
            learning_curve_validation_button: gr.Button(LN.learning_curve_validation_button, visible=Dataset.check_before_train()),
            shap_beeswarm_radio: gr.Radio(Dataset.get_trained_model_list(), visible=Dataset.check_before_train(), label=LN.shap_beeswarm_radio),
            shap_beeswarm_button: gr.Button(LN.shap_beeswarm_button, visible=Dataset.check_before_train()),
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
        linear_regression_model_radio: gr.Radio(visible=False),
        model_optimize_radio: gr.Radio(visible=False),
        model_train_button: gr.Button(visible=False),
        model_train_checkbox: gr.Checkbox(visible=False),
        select_as_model_radio: gr.Radio(visible=False),
        choose_assign_radio: gr.Radio(visible=False),

        draw_plot: gr.Plot(visible=False),
        draw_file: gr.File(visible=False),
        title_name_textbox: gr.Textbox(visible=False),
        x_label_textbox: gr.Textbox(visible=False),
        y_label_textbox: gr.Textbox(visible=False),

        # [绘图]
        learning_curve_checkboxgroup: gr.Checkboxgroup(visible=False),
        learning_curve_train_button: gr.Button(visible=False),
        learning_curve_validation_button: gr.Button(visible=False),
        shap_beeswarm_radio: gr.Radio(visible=False),
        shap_beeswarm_button: gr.Button(visible=False),
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

    container_dict = {
        MN.linear_regression: Container(),
        MN.polynomial_regression: Container(),
        MN.logistic_regression: Container(),
    }

    visualize = ""

    @classmethod
    def get_dataset_list(cls):
        return ["Iris Dataset", "Wine Dataset", "Breast Cancer Dataset", "自定义"]

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
    def get_linear_regression_model_name_mapping(cls):
        return dict(zip(cls.get_linear_regression_model_list(), ["LinearRegression", "Lasso", "Ridge", "ElasticNet"]))

    @classmethod
    def train_model(cls, optimize, linear_regression_model_type=None):
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

        if cls.cur_model == MN.linear_regression:
            container = linear_regression(container, cls.get_linear_regression_model_name_mapping()[linear_regression_model_type])
        elif cls.cur_model == MN.polynomial_regression:
            container = polynomial_regression(container)
        elif cls.cur_model == MN.logistic_regression:
            container = logistic_regression(container)

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
    def get_model_chinese_name(cls):
        return ["线性回归", "多项式回归", "逻辑斯谛分类"]

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
        if cls.visualize == MN.learning_curve_train:
            return cls.draw_learning_curve_train_plot(select_model, color_list, label_list, name, x_label, y_label, is_default)
        elif cls.visualize == MN.learning_curve_validation:
            return cls.draw_learning_curve_validation_plot(select_model, color_list, label_list, name, x_label, y_label, is_default)
        elif cls.visualize == MN.shap_beeswarm:
            return cls.draw_shap_beeswarm_plot(select_model, color_list, label_list, name, x_label, y_label, is_default)

    @classmethod
    def draw_learning_curve_train_plot(cls, model_list, color_list: list, label_list: list, name: str, x_label: str, y_label: str, is_default: bool):
        learning_curve_dict = {}

        for model_name in model_list:
            model_name = cls.get_model_name_mapping_reverse()[model_name]
            learning_curve_dict[model_name] = cls.container_dict[model_name].get_learning_curve_values()

        color_cur_list = Config.COLORS if is_default else color_list
        label_cur_list = [x for x in learning_curve_dict.keys()] if is_default else label_list
        x_cur_label = "Train Sizes" if is_default else x_label
        y_cur_label = "Accuracy" if is_default else y_label
        cur_name = "" if is_default else name

        paint_object = PaintObject()
        paint_object.set_color_cur_list(color_cur_list)
        paint_object.set_label_cur_list(label_cur_list)
        paint_object.set_x_cur_label(x_cur_label)
        paint_object.set_y_cur_label(y_cur_label)
        paint_object.set_name(cur_name)

        return draw_learning_curve_total(learning_curve_dict, "train", paint_object)

    @classmethod
    def draw_learning_curve_validation_plot(cls, model_list, color_list: list, label_list: list, name: str, x_label: str, y_label: str, is_default: bool):
        learning_curve_dict = {}

        for model_name in model_list:
            model_name = cls.get_model_name_mapping_reverse()[model_name]
            learning_curve_dict[model_name] = cls.container_dict[model_name].get_learning_curve_values()

        color_cur_list = Config.COLORS if is_default else color_list
        label_cur_list = [x for x in learning_curve_dict.keys()] if is_default else label_list
        x_cur_label = "Train Sizes" if is_default else x_label
        y_cur_label = "Accuracy" if is_default else y_label
        cur_name = "" if is_default else name

        paint_object = PaintObject()
        paint_object.set_color_cur_list(color_cur_list)
        paint_object.set_label_cur_list(label_cur_list)
        paint_object.set_x_cur_label(x_cur_label)
        paint_object.set_y_cur_label(y_cur_label)
        paint_object.set_name(cur_name)

        return draw_learning_curve_total(learning_curve_dict, "validation", paint_object)

    @classmethod
    def draw_shap_beeswarm_plot(cls, model_name, color_list: list, label_list: list, name: str, x_label: str, y_label: str, is_default: bool):
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

        return shap_calculate(container.get_model(), container.x_train, cls.data.columns.values, paint_object)

    @classmethod
    def get_file(cls):
        # [绘图]
        if cls.visualize == MN.learning_curve_train:
            return FilePath.png_base.format(FilePath.learning_curve_train_plot)
        elif cls.visualize == MN.learning_curve_validation:
            return FilePath.png_base.format(FilePath.learning_curve_validation_plot)
        elif cls.visualize == MN.shap_beeswarm:
            return FilePath.png_base.format(FilePath.shap_beeswarm_plot)

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


def choose_assign(assign: str):
    Dataset.choose_assign(assign)

    return get_return(True)


def select_as_model(model_name: str):
    Dataset.select_as_model(model_name)

    return get_return(True)


# [绘图]
def shap_beeswarm_first_draw_plot(*inputs):
    Dataset.visualize = MN.shap_beeswarm
    return first_draw_plot(inputs)


def learning_curve_validation_first_draw_plot(*inputs):
    Dataset.visualize = MN.learning_curve_validation
    return first_draw_plot(inputs)


def learning_curve_train_first_draw_plot(*inputs):
    Dataset.visualize = MN.learning_curve_train
    return first_draw_plot(inputs)


def first_draw_plot(inputs):
    select_model = inputs[0]
    x_label = ""
    y_label = ""
    name = ""
    color_list = []
    label_list = []

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

    # 绘图
    if Dataset.visualize == MN.learning_curve_train:
        select_model = inputs[start_index]
    elif Dataset.visualize == MN.learning_curve_validation:
        select_model = inputs[start_index]
    elif Dataset.visualize == MN.shap_beeswarm:
        select_model = inputs[start_index+1]

    else:
        select_model = inputs[start_index: start_index+1]

    cur_plt, paint_object = Dataset.draw_plot(select_model, color_list, label_list, name, x_label, y_label, False)

    return first_draw_plot_with_non_first_draw_plot(cur_plt, paint_object)


def first_draw_plot_with_non_first_draw_plot(cur_plt, paint_object):
    extra_gr_dict = {}

    # [绘图]
    if Dataset.visualize == MN.learning_curve_train:
        cur_plt.savefig(FilePath.png_base.format(FilePath.learning_curve_train_plot), dpi=300)
        extra_gr_dict.update({draw_plot: gr.Plot(cur_plt, visible=True, label=LN.learning_curve_train_plot)})
    elif Dataset.visualize == MN.learning_curve_validation:
        cur_plt.savefig(FilePath.png_base.format(FilePath.learning_curve_validation_plot), dpi=300)
        extra_gr_dict.update({draw_plot: gr.Plot(cur_plt, visible=True, label=LN.learning_curve_validation_plot)})
    elif Dataset.visualize == MN.shap_beeswarm:
        cur_plt.savefig(FilePath.png_base.format(FilePath.shap_beeswarm_plot), dpi=300)
        extra_gr_dict.update({draw_plot: gr.Plot(cur_plt, visible=True, label=LN.shap_beeswarm_plot)})

    extra_gr_dict.update(dict(zip(colorpickers, Dataset.colorpickers_change(paint_object))))
    extra_gr_dict.update(dict(zip(color_textboxs, Dataset.color_textboxs_change(paint_object))))
    extra_gr_dict.update(dict(zip(legend_labels_textboxs, Dataset.labels_change(paint_object))))
    extra_gr_dict.update({title_name_textbox: gr.Textbox(paint_object.get_name(), visible=True, label=LN.title_name_textbox)})
    extra_gr_dict.update({x_label_textbox: gr.Textbox(paint_object.get_x_cur_label(), visible=True, label=LN.x_label_textbox)})
    extra_gr_dict.update({y_label_textbox: gr.Textbox(paint_object.get_y_cur_label(), visible=True, label=LN.y_label_textbox)})

    return get_return_extra(True, extra_gr_dict)


def train_model(optimize, linear_regression_model_type):
    Dataset.train_model(optimize, linear_regression_model_type)

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

    return get_return(True, {
        display_encode_label_dataframe: gr.Dataframe(Dataset.get_str2int_mappings_df(), type="pandas", visible=True,
                                                     label=LN.display_encode_label_dataframe)})


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


with gr.Blocks() as demo:
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
            select_as_model_radio = gr.Radio(visible=False)
            linear_regression_model_radio = gr.Radio(visible=False)
            model_optimize_radio = gr.Radio(visible=False)
            model_train_button = gr.Button(visible=False)
            model_train_checkbox = gr.Checkbox(visible=False)

        # 可视化
        with gr.Accordion("数据可视化"):
            with gr.Tab("学习曲线图"):
                learning_curve_checkboxgroup = gr.Checkboxgroup(visible=False)
                with gr.Row():
                    learning_curve_train_button = gr.Button(visible=False)
                    learning_curve_validation_button = gr.Button(visible=False)

            with gr.Tab("蜂群特征图"):
                shap_beeswarm_radio = gr.Radio(visible=False)
                shap_beeswarm_button = gr.Button(visible=False)

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
    model_train_button.click(fn=train_model, inputs=[model_optimize_radio, linear_regression_model_radio], outputs=get_outputs())

    # 可视化
    learning_curve_train_button.click(fn=learning_curve_train_first_draw_plot, inputs=[learning_curve_checkboxgroup], outputs=get_outputs())
    learning_curve_validation_button.click(fn=learning_curve_validation_first_draw_plot, inputs=[learning_curve_checkboxgroup], outputs=get_outputs())
    shap_beeswarm_button.click(fn=shap_beeswarm_first_draw_plot, inputs=[shap_beeswarm_radio], outputs=get_outputs())

    title_name_textbox.blur(fn=out_non_first_draw_plot, inputs=[title_name_textbox] + [x_label_textbox] + [y_label_textbox] + colorpickers + legend_labels_textboxs
                            + [learning_curve_checkboxgroup] + [shap_beeswarm_radio], outputs=get_outputs())
    x_label_textbox.blur(fn=out_non_first_draw_plot, inputs=[title_name_textbox] + [x_label_textbox] + [y_label_textbox] + colorpickers + legend_labels_textboxs
                         + [learning_curve_checkboxgroup] + [shap_beeswarm_radio], outputs=get_outputs())
    y_label_textbox.blur(fn=out_non_first_draw_plot, inputs=[title_name_textbox] + [x_label_textbox] + [y_label_textbox] + colorpickers + legend_labels_textboxs
                         + [learning_curve_checkboxgroup] + [shap_beeswarm_radio], outputs=get_outputs())
    for i in range(StaticValue.max_num):
        colorpickers[i].blur(fn=out_non_first_draw_plot, inputs=[title_name_textbox] + [x_label_textbox] + [y_label_textbox] + colorpickers + legend_labels_textboxs
                             + [learning_curve_checkboxgroup] + [shap_beeswarm_radio], outputs=get_outputs())
        color_textboxs[i].blur(fn=out_non_first_draw_plot, inputs=[title_name_textbox] + [x_label_textbox] + [y_label_textbox] + color_textboxs + legend_labels_textboxs
                               + [learning_curve_checkboxgroup] + [shap_beeswarm_radio], outputs=get_outputs())
        legend_labels_textboxs[i].blur(fn=out_non_first_draw_plot, inputs=[title_name_textbox] + [x_label_textbox] + [y_label_textbox] + colorpickers + legend_labels_textboxs
                                       + [learning_curve_checkboxgroup] + [shap_beeswarm_radio], outputs=get_outputs())

if __name__ == "__main__":
    demo.launch()
