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
