import matplotlib.pyplot as plt


def draw_line_graph(nums, labels, paint_object):
    plt.figure(figsize=(10, 8), dpi=300)

    plt.plot(
        nums,
        labels,
        "-o",
        color=paint_object.get_color_cur_list()[0]
    )

    plt.title(paint_object.get_name())

    plt.xlabel(paint_object.get_x_cur_label())
    plt.ylabel(paint_object.get_y_cur_label())

    paint_object.set_color_cur_num(1)

    return plt, paint_object



