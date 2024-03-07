# 全局静态变量值存储类
class StaticValue:
    # 超参数文本框的最大组件数量
    MAX_PARAMS_NUM = 60
    # 颜色和标签显示的最大组件数量
    MAX_NUM = 20
    # 随机种子 (数据集切分+模型训练)
    RANDOM_STATE = 123

    # 参数类型
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    STR = "str"

    # 画图颜色组重复次数
    COLOR_ITER_NUM = 3

    # 颜色组
    COLORS = [
                 "#ca5353",
                 "#c874a5",
                 "#b674c8",
                 "#8274c8",
                 "#748dc8",
                 "#74acc8",
                 "#74c8b7",
                 "#74c88d",
                 "#a6c874",
                 "#e0e27e",
                 "#df9b77",
                 "#404040",
                 "#999999",
                 "#d4d4d4"
             ] * COLOR_ITER_NUM

    COLORS_0 = [
                   "#8074C8",
                   "#7895C1",
                   "#A8CBDF",
                   "#992224",
                   "#B54764",
                   "#E3625D",
                   "#EF8B67",
                   "#F0C284"
               ] * COLOR_ITER_NUM

    COLORS_1 = [
                   "#4A5F7E",
                   "#719AAC",
                   "#72B063",
                   "#94C6CD",
                   "#B8DBB3",
                   "#E29135"
               ] * COLOR_ITER_NUM

    COLORS_2 = [
                   "#4485C7",
                   "#D4562E",
                   "#DBB428",
                   "#682487",
                   "#84BA42",
                   "#7ABBDB",
                   "#A51C36"
               ] * COLOR_ITER_NUM

    COLORS_3 = [
                   "#8074C8",
                   "#7895C1",
                   "#A8CBDF",
                   "#F5EBAE",
                   "#F0C284",
                   "#EF8B67",
                   "#E3625D",
                   "#B54764"
               ] * COLOR_ITER_NUM

    COLORS_4 = [
                   "#979998",
                   "#C69287",
                   "#E79A90",
                   "#EFBC91",
                   "#E4CD87",
                   "#FAE5BB",
                   "#DDDDDF"
               ] * COLOR_ITER_NUM

    COLORS_5 = [
                   "#91CCC0",
                   "#7FABD1",
                   "#F7AC53",
                   "#EC6E66",
                   "#B5CE4E",
                   "#BD7795",
                   "#7C7979"
               ] * COLOR_ITER_NUM

    COLORS_6 = [
                   "#E9687A",
                   "#F58F7A",
                   "#FDE2D8",
                   "#CFCFD0",
                   "#B6B3D6"
               ] * COLOR_ITER_NUM


# 文件路径相关静态变量存储类
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
    # 绘图Step 15:在这里添加新的绘图方法名称


# 模型名称静态变量存储类
class MN:  # ModelName
    classification = "classification"
    regression = "regression"

    # [模型]
    linear_regressor = "linear regressor"
    polynomial_regressor = "polynomial regressor"
    logistic_classifier = "logistic classifier"
    decision_tree_classifier = "decision tree classifier"
    random_forest_classifier = "random forest classifier"
    random_forest_regressor = "random forest regressor"
    xgboost_classifier = "xgboost classifier"
    lightGBM_classifier = "lightGBM classifier"
    gradient_boosting_regressor = "gradient boosting regressor"
    svm_classifier = "svm classifier"
    svm_regressor = "svm regressor"
    knn_classifier = "knn classifier"
    knn_regressor = "knn regressor"
    naive_bayes_classifier = "naive bayes classifier"
    # 模型Step 4:在这里添加新的模型名称

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
    # 绘图Step 4:在这里添加新的绘图方法名称


# 组件标签名称静态变量存储类
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
    train_size_textbox = "分割出的训练集所占比例"
    model_optimize_radio = "选择超参数优化方法"
    model_train_input_params_dataframe = "超参数列表"
    model_train_button = "训练"
    model_train_params_dataframe = "训练后的模型参数"
    model_train_metrics_dataframe = "训练后的模型指标"
    select_as_model_radio = "选择所需训练的模型"

    # [模型]
    linear_regression_model_radio = "选择线性回归的模型"
    naive_bayes_classification_model_radio = "选择朴素贝叶斯分类的模型"
    # 模型Step 5:在这里添加新的模型额外组件名称

    title_name_textbox = "标题"
    x_label_textbox = "x 轴名称"
    y_label_textbox = "y 轴名称"
    colors = ["颜色 {}".format(i) for i in range(StaticValue.MAX_NUM)]
    labels = ["图例 {}".format(i) for i in range(StaticValue.MAX_NUM)]

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
    # 绘图Step 5:在这里添加新的绘图方法相关组件名称

    data_distribution_plot = "数据分布图"
    descriptive_indicators_plot = "箱线统计图"
    heatmap_plot = "系数热力图"
    learning_curve_plot = "学习曲线图"
    shap_beeswarm_plot = "特征蜂群图"
    data_fit_plot = "数据拟合图"
    waterfall_plot = "特征瀑布图"
    force_plot = "特征力图"
    dependence_plot = "特征依赖图"
    # 绘图Step 6:在这里添加新的绘图方法名称









