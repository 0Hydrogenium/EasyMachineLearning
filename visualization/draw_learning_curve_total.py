from matplotlib import pyplot as plt


def draw_learning_curve_total(input_dict, paint_object):
    plt.clf()
    plt.figure(figsize=(10, 8), dpi=300)

    for i, values in enumerate(input_dict.values()):
        train_sizes = values[0]
        train_scores_mean = values[1]
        train_scores_std = values[2]
        test_scores_mean = values[3]
        test_scores_std = values[4]

        plt.fill_between(
            train_sizes,
            train_scores_mean - train_scores_std,
            train_scores_mean + train_scores_std,
            alpha=0.1,
            color=paint_object.get_color_cur_list()[2*i]
        )

        plt.plot(
            train_sizes,
            train_scores_mean,
            "o-",
            color=paint_object.get_color_cur_list()[2*i],
            label=paint_object.get_label_cur_list()[2*i]
        )

        plt.fill_between(
            train_sizes,
            test_scores_mean - test_scores_std,
            test_scores_mean + test_scores_std,
            alpha=0.1,
            color=paint_object.get_color_cur_list()[2*i+1]
        )
        plt.plot(
            train_sizes,
            test_scores_mean,
            "o-",
            color=paint_object.get_color_cur_list()[2*i+1],
            label=paint_object.get_label_cur_list()[2*i+1]
        )

    plt.title(paint_object.get_name())

    plt.xlabel(paint_object.get_x_cur_label())
    plt.ylabel(paint_object.get_y_cur_label())
    plt.legend()

    paint_object.set_color_cur_num(2*len(input_dict.values()))
    paint_object.set_label_cur_num(2*len(input_dict.values()))

    return plt, paint_object

