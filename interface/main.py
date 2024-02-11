import copy

import gradio as gr
from sklearn import preprocessing
import pandas as pd

from coding.llh.static.process import *


class LN:  # LabelName
    choose_dataset_radio = "选择数据源"
    display_dataset_dataframe = "数据表信息"
    display_total_col_num_text = "总列数"
    display_total_row_num_text = "总行数"
    display_na_list_text = "存在缺失值的列"
    del_all_na_col_button = "删除所有存在缺失值的列"
    display_duplicate_num_text = "重复的行数"
    del_col_checkboxgroup = "选择所需删除的列"
    del_col_button = "删除"
    del_duplicate_button = "删除所有重复行"
    encode_label_checkboxgroup = "选择所需标签编码的字符型数值列"
    display_encode_label_dataframe = "标签编码信息"
    encode_label_button = "字符型转数值型"
    change_data_type_to_float_button = "将所有数据强制转换为浮点型"
    standardize_data_checkboxgroup = "选择所需标准化的列"
    standardize_data_button = "标准化"
    select_as_y_radio = "选择因变量"


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
        encode_label_button,
        display_encode_label_dataframe,
        encode_label_checkboxgroup,
        change_data_type_to_float_button,
        standardize_data_checkboxgroup,
        standardize_data_button,
        select_as_y_radio,
    }

    return gr_dict


def get_return(is_visible, extra_gr_dict: dict = None):
    if is_visible:
        gr_dict = {
            display_dataset_dataframe: gr.Dataframe(add_index_into_df(Dataset.data), type="pandas", visible=True, label=LN.display_dataset_dataframe),
            display_total_col_num_text: gr.Textbox(str(Dataset.get_total_col_num()), visible=True, label=LN.display_total_col_num_text),
            display_total_row_num_text: gr.Textbox(str(Dataset.get_total_row_num()), visible=True, label=LN.display_total_row_num_text),
            display_na_list_text: gr.Textbox(Dataset.get_na_list_str(), visible=True, label=LN.display_na_list_text),
            del_all_na_col_button: gr.Button(LN.del_all_na_col_button, visible=True),
            display_duplicate_num_text: gr.Textbox(str(Dataset.get_duplicate_num()), visible=True, label=LN.display_duplicate_num_text),
            del_duplicate_button: gr.Button(LN.del_duplicate_button, visible=True),
            del_col_checkboxgroup: gr.Checkboxgroup(Dataset.get_col_list(), visible=True, label=LN.del_col_checkboxgroup),
            del_col_button: gr.Button(LN.del_col_button, visible=True),
            encode_label_button: gr.Button(LN.encode_label_button, visible=True),
            display_encode_label_dataframe: gr.Dataframe(visible=False),
            encode_label_checkboxgroup: gr.Checkboxgroup(Dataset.get_non_numeric_list(), visible=True, label=LN.encode_label_checkboxgroup),
            change_data_type_to_float_button: gr.Button(LN.change_data_type_to_float_button, visible=True),
            standardize_data_checkboxgroup: gr.Checkboxgroup(Dataset.get_non_standardized_data(), visible=True, label=LN.standardize_data_checkboxgroup),
            standardize_data_button: gr.Button(LN.standardize_data_button, visible=True),
            select_as_y_radio: gr.Radio(Dataset.get_col_list(), visible=True, label=LN.select_as_y_radio),
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
        encode_label_button: gr.Button(visible=False),
        display_encode_label_dataframe: gr.Dataframe(visible=False),
        encode_label_checkboxgroup: gr.Checkboxgroup(visible=False),
        change_data_type_to_float_button: gr.Button(visible=False),
        standardize_data_checkboxgroup: gr.Checkboxgroup(visible=False),
        standardize_data_button: gr.Button(visible=False),
        select_as_y_radio: gr.Radio(visible=False),
    }

    return gr_dict


class Dataset:
    file = ""
    data = pd.DataFrame()
    na_list = []
    non_numeric_list = []
    str2int_mappings = {}

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
                na_list_str = na_list_str.rstrip(", ")
                na_list.append(cur_index)

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
        cls.data = cls.data.drop_duplicates().reset_index()

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
    def change_data_type_to_float(cls):
        data_copy = cls.data

        for col in data_copy.columns.values:
            data_copy[col] = data_copy[col].astype(float)

        cls.data = data_copy

    @classmethod
    def get_non_standardized_data(cls):
        not_standardized_data_list = []

        for col in cls.data.columns.values:
            if not np.array_equal(np.round(preprocessing.scale(cls.data[col]), decimals=2), np.round(cls.data[col].values.round(2), decimals=2)):
                not_standardized_data_list.append(col)

        return not_standardized_data_list

    @classmethod
    def standardize_data(cls, col_list: list):
        for col in col_list:
            cls.data[col] = preprocessing.scale(cls.data[col])

    @classmethod
    def select_as_y(cls, col: str):
        cls.data = pd.concat([cls.data[col], cls.data.drop(col, axis=1)], axis=1)


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

    # 显示数据表信息
    with gr.Column():
        display_dataset_dataframe = gr.Dataframe(visible=False)
        display_total_col_num_text = gr.Textbox(visible=False)
        display_total_row_num_text = gr.Textbox(visible=False)
        display_na_list_text = gr.Textbox(visible=False)
        display_duplicate_num_text = gr.Textbox(visible=False)
        display_encode_label_dataframe = gr.Dataframe(visible=False)

    # 选择数据源
    with gr.Column():
        with gr.Group():
            choose_dataset_radio = gr.Radio(Dataset.get_dataset_list(), label=LN.choose_dataset_radio)
            choose_custom_dataset_file = gr.File(visible=False)

    # 操作数据表
    with gr.Column():
        del_col_checkboxgroup = gr.Checkboxgroup(visible=False)
        del_col_button = gr.Button(visible=False)
        del_all_na_col_button = gr.Button(visible=False)
        del_duplicate_button = gr.Button(visible=False)
        encode_label_checkboxgroup = gr.Checkboxgroup(visible=False)
        encode_label_button = gr.Button(visible=False)
        change_data_type_to_float_button = gr.Button(visible=False)
        standardize_data_checkboxgroup = gr.Checkboxgroup(visible=False)
        standardize_data_button = gr.Button(visible=False)
        select_as_y_radio = gr.Radio(visible=False)

    '''
        监听事件
    '''

    # 选择数据源
    choose_dataset_radio.change(fn=choose_dataset, inputs=[choose_dataset_radio], outputs=get_outputs())
    choose_custom_dataset_file.upload(fn=choose_custom_dataset, inputs=[choose_custom_dataset_file], outputs=get_outputs())

    # 操作数据表

    # 删除所选列
    del_col_button.click(fn=del_col, inputs=[del_col_checkboxgroup], outputs=get_outputs())
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


if __name__ == "__main__":
    demo.launch()
