import copy

import gradio as gr
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd

from analysis.shap_model import shap_calculate
from static.process import *
from analysis.linear_model import *
from visualization.draw_learning_curve_total import draw_learning_curve_total


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


class LN:  # LabelName
    choose_dataset_radio = "选择所需数据源"
    display_total_col_num_text = "总列数"
    display_total_row_num_text = "总行数"
    display_na_list_text = "存在缺失值的列"
    del_all_na_col_button = "删除所有存在缺失值的列"
    display_duplicate_num_text = "重复的行数"
    del_col_checkboxgroup = "选择所需删除的列"
    del_col_button = "删除"
    remain_row_slider = "保留的行数"
    remain_row_button = "保留"
    del_duplicate_button = "删除所有重复行"
    encode_label_checkboxgroup = "选择所需标签编码的字符型数值列"
    display_encode_label_dataframe = "标签编码信息"
    encode_label_button = "字符型转数值型"
    change_data_type_to_float_button = "将所有数据强制转换为浮点型"
    standardize_data_checkboxgroup = "选择所需标准化的列"
    standardize_data_button = "标准化"
    select_as_y_radio = "选择因变量"
    linear_regression_model_radio = "选择线性回归的模型"
    linear_regression_optimize_radio = "选择线性回归的超参数优化方法"
    linear_regression_button = "训练线性回归模型"
    linear_regression_checkbox = "线性回归模型是否完成训练"
    learning_curve_checkboxgroup = "选择所需绘制学习曲线的模型"
    learning_curve_train_button = "绘制训练集学习曲线"
    learning_curve_validation_button = "绘制验证集学习曲线"
    learning_curve_train_plot = "训练集学习曲线"
    learning_curve_validation_plot = "验证集学习曲线"
    shap_beeswarm_radio = "选择所需绘制蜂群特征图的模型"
    shap_beeswarm_plot = "蜂群特征图"


def get_outputs():
    gr_dict = {
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
        linear_regression_optimize_radio,
        linear_regression_button,
        linear_regression_checkbox,
        learning_curve_checkboxgroup,
        learning_curve_train_button,
        learning_curve_validation_button,
        learning_curve_train_plot,
        learning_curve_validation_plot,
        shap_beeswarm_radio,
        shap_beeswarm_plot,
    }

    return gr_dict


def get_return(is_visible, extra_gr_dict: dict = None):
    if is_visible:
        gr_dict = {
            display_dataset_dataframe: gr.Dataframe(add_index_into_df(Dataset.data), type="pandas", visible=True),
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
            standardize_data_checkboxgroup: gr.Checkboxgroup(Dataset.get_non_standardized_data(), visible=True, label=LN.standardize_data_checkboxgroup),
            standardize_data_button: gr.Button(LN.standardize_data_button, visible=True),
            select_as_y_radio: gr.Radio(Dataset.get_col_list(), visible=Dataset.check_select_as_y_radio(), label=LN.select_as_y_radio),
            linear_regression_model_radio: gr.Radio(Dataset.get_linear_regression_model_list(), visible=Dataset.select_y_mark, label=LN.linear_regression_model_radio),
            linear_regression_optimize_radio: gr.Radio(Dataset.get_optimize_list(), visible=Dataset.select_y_mark, label=LN.linear_regression_optimize_radio),
            linear_regression_button: gr.Button(LN.linear_regression_button, visible=Dataset.select_y_mark),
            linear_regression_checkbox: gr.Checkbox(Dataset.get_linear_regression_container_status(), visible=Dataset.select_y_mark, label=LN.linear_regression_checkbox),
            learning_curve_checkboxgroup: gr.Checkboxgroup(Dataset.get_trained_model_list(), visible=Dataset.select_y_mark, label=LN.learning_curve_checkboxgroup),
            learning_curve_train_button: gr.Button(LN.learning_curve_train_button, visible=Dataset.select_y_mark),
            learning_curve_validation_button: gr.Button(LN.learning_curve_validation_button, visible=Dataset.select_y_mark),
            shap_beeswarm_radio: gr.Radio(Dataset.get_trained_model_list(), visible=Dataset.select_y_mark, label=LN.shap_beeswarm_radio),
        }

        if extra_gr_dict:
            gr_dict.update(extra_gr_dict)

        return gr_dict

    gr_dict = {
        choose_custom_dataset_file: gr.File(None, visible=True),
        display_dataset_dataframe: gr.Dataframe(visible=False),
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
        linear_regression_optimize_radio: gr.Radio(visible=False),
        linear_regression_button: gr.Button(visible=False),
        linear_regression_checkbox: gr.Checkbox(visible=False),
        learning_curve_checkboxgroup: gr.Checkboxgroup(visible=False),
        learning_curve_train_button: gr.Button(visible=False),
        learning_curve_validation_button: gr.Button(visible=False),
        learning_curve_train_plot: gr.Plot(visible=False),
        learning_curve_validation_plot: gr.Plot(visible=False),
        shap_beeswarm_radio: gr.Radio(visible=False),
        shap_beeswarm_plot: gr.Plot(visible=False),
    }

    return gr_dict


class Dataset:
    file = ""
    data = pd.DataFrame()

    na_list = []
    non_numeric_list = []
    str2int_mappings = {}
    max_num = 0
    data_copy = pd.DataFrame()
    select_y_mark = False

    container_dict = {
        "linear_regression": Container(),
    }

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
    def encode_label(cls, col_list: list):
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

        for col in data_copy.columns.values:
            data_copy[col] = data_copy[col].astype(float)

        cls.data = data_copy

    @classmethod
    def get_non_standardized_data(cls):
        not_standardized_data_list = []

        for col in cls.data.columns.values:
            if cls.data[col].dtype.name in ["int64", "float64"]:
                if not np.array_equal(np.round(preprocessing.scale(cls.data[col]), decimals=2), np.round(cls.data[col].values.round(2), decimals=2)):
                    not_standardized_data_list.append(col)

        return not_standardized_data_list

    @classmethod
    def check_select_as_y_radio(cls):
        for col in cls.data.columns.values:
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
    def train_linear_regression(cls, optimize: str, model_name: str):
        optimize = cls.get_optimize_name_mapping()[optimize]
        model_name = cls.get_linear_regression_model_name_mapping()[model_name]

        x_train, x_test, y_train, y_test = train_test_split(
            cls.data.values[:, 1:],
            cls.data.values[:, :1],
            random_state=Config.RANDOM_STATE,
            train_size=0.8
        )
        container = Container(x_train, y_train, x_test, y_test, optimize)

        container = linear_regression(container, model_name)
        cls.container_dict["linear_regression"] = container

    @classmethod
    def get_linear_regression_container_status(cls):
        status = cls.container_dict["linear_regression"].get_status()
        if status == "trained":
            return True

        return False

    @classmethod
    def get_model_name(cls):
        return ["linear_regression"]

    @classmethod
    def get_model_chinese_name(cls):
        return ["线性回归"]

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
    def draw_learning_curve_train_plot(cls, model_list: list) -> plt.Figure:
        learning_curve_dict = {}

        for model_name in model_list:
            model_name = cls.get_model_name_mapping_reverse()[model_name]
            learning_curve_dict[model_name] = cls.container_dict[model_name].get_learning_curve_values()

        return draw_learning_curve_total(learning_curve_dict, "train")

    @classmethod
    def draw_learning_curve_validation_plot(cls, model_list: list) -> plt.Figure:
        learning_curve_dict = {}

        for model_name in model_list:
            model_name = cls.get_model_name_mapping_reverse()[model_name]
            learning_curve_dict[model_name] = cls.container_dict[model_name].get_learning_curve_values()

        return draw_learning_curve_total(learning_curve_dict, "validation")

    @classmethod
    def draw_shap_beeswarm_plot(cls, model_name) -> plt.Figure:
        model_name = cls.get_model_name_mapping_reverse()[model_name]
        container = cls.container_dict[model_name]

        return shap_calculate(container.get_model(), container.x_train, cls.data.columns.values)


def draw_shap_beeswarm_plot(model_name):
    cur_plt = Dataset.draw_shap_beeswarm_plot(model_name)

    return get_return(True, {shap_beeswarm_plot: gr.Plot(cur_plt, visible=True, label=LN.shap_beeswarm_plot)})


def draw_learning_curve_validation_plot(model_list: list):
    cur_plt = Dataset.draw_learning_curve_validation_plot(model_list)

    return get_return(True, {learning_curve_validation_plot: gr.Plot(cur_plt, visible=True, label=LN.learning_curve_validation_plot)})


def draw_learning_curve_train_plot(model_list: list):
    cur_plt = Dataset.draw_learning_curve_train_plot(model_list)

    return get_return(True, {learning_curve_train_plot: gr.Plot(cur_plt, visible=True, label=LN.learning_curve_train_plot)})


def train_linear_regression(optimize: str, model: str):
    Dataset.train_linear_regression(optimize, model)

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
            with gr.Row():
                with gr.Column():
                    del_col_checkboxgroup = gr.Checkboxgroup(visible=False)
                    del_col_button = gr.Button(visible=False)
                with gr.Column():
                    encode_label_checkboxgroup = gr.Checkboxgroup(visible=False)
                    encode_label_button = gr.Button(visible=False)
                    display_encode_label_dataframe = gr.Dataframe(visible=False)
            with gr.Row():
                with gr.Column():
                    data_type_dataframe = gr.Dataframe(visible=False)
                    change_data_type_to_float_button = gr.Button(visible=False)
                with gr.Column():
                    standardize_data_checkboxgroup = gr.Checkboxgroup(visible=False)
                    standardize_data_button = gr.Button(visible=False)
            select_as_y_radio = gr.Radio(visible=False)

        # 回归模型
        with gr.Accordion("回归模型"):
            linear_regression_model_radio = gr.Radio(visible=False)
            linear_regression_optimize_radio = gr.Radio(visible=False)
            linear_regression_button = gr.Button(visible=False)
            linear_regression_checkbox = gr.Checkbox(visible=False)

        # 可视化
        with gr.Accordion("数据可视化"):
            learning_curve_checkboxgroup = gr.Checkboxgroup(visible=False)
            with gr.Row():
                learning_curve_train_button = gr.Button(visible=False)
                learning_curve_validation_button = gr.Button(visible=False)
            learning_curve_train_plot = gr.Plot(visible=False)
            learning_curve_validation_plot = gr.Plot(visible=False)
            shap_beeswarm_radio = gr.Radio(visible=False)
            shap_beeswarm_plot = gr.Plot(visible=False)

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
    # 将所有数据强制转换为浮点型
    change_data_type_to_float_button.click(fn=change_data_type_to_float, outputs=get_outputs())
    # 标准化数据
    standardize_data_button.click(fn=standardize_data, inputs=[standardize_data_checkboxgroup], outputs=get_outputs())
    # 选择因变量
    select_as_y_radio.change(fn=select_as_y, inputs=[select_as_y_radio], outputs=get_outputs())

    # 回归模型
    linear_regression_button.click(fn=train_linear_regression, inputs=[linear_regression_optimize_radio, linear_regression_model_radio], outputs=get_outputs())

    # 可视化
    learning_curve_train_button.click(fn=draw_learning_curve_train_plot, inputs=[learning_curve_checkboxgroup], outputs=get_outputs())
    learning_curve_validation_button.click(fn=draw_learning_curve_validation_plot, inputs=[learning_curve_checkboxgroup], outputs=get_outputs())
    shap_beeswarm_radio.change(fn=draw_shap_beeswarm_plot, inputs=[shap_beeswarm_radio], outputs=get_outputs())

if __name__ == "__main__":
    demo.launch()
