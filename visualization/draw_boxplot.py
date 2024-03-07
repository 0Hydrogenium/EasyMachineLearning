import matplotlib.pyplot as plt


def draw_boxplot(x_data, paint_object, will_rotate=False):
    plt.figure(figsize=(10, 8), dpi=300)

    plt.grid(True)

    plt.boxplot(
        x_data,
        meanline=True,
        showmeans=True,
        medianprops={"color": paint_object.get_color_cur_list()[0], "linewidth": 1.5},
        meanprops={"color": paint_object.get_color_cur_list()[1], "ls": "--", "linewidth": 1.5},
        flierprops={"marker": "o", "markerfacecolor": paint_object.get_color_cur_list()[2]},
        labels=x_data.columns.values
    )

    if will_rotate:
        plt.xticks(rotation=-45)

    plt.title(paint_object.get_name())

    plt.xlabel(paint_object.get_x_cur_label())
    plt.ylabel(paint_object.get_y_cur_label())

    paint_object.set_color_cur_num(3)

    return plt, paint_object

