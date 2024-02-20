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


