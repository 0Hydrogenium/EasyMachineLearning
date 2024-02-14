class PaintObject:
    def __init__(self):
        self.cur_num = 0
        self.cur_list = []

    def get_cur_num(self):
        return self.cur_num

    def set_cur_num(self, cur_num):
        self.cur_num = cur_num

    def get_cur_list(self):
        return self.cur_list

    def set_cur_list(self, cur_list):
        self.cur_list = cur_list