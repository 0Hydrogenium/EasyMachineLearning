import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from coding.llh.static.config import Config


# Draw scatter
def draw_scatter_2D(x_data, y_data, centers, title):
    num_clusters = np.unique(y_data)

    plt.figure(figsize=(10, 8))

    for i in range(len(num_clusters)):
        plt.scatter(x_data[y_data == i][:, 0], x_data[y_data == i][:, 1], s=1)
    for i in range(len(num_clusters)):
        plt.scatter(centers[i, 0], centers[i, 1], marker="*", s=50, c="black")

    plt.title(title)

    plt.savefig("./diagram/{}.png".format(title), dpi=300)

    plt.show()


def draw_scatter_2D_1(x_data, title):
    plt.figure(figsize=(10, 8))

    plt.scatter(x_data[:, 0], x_data[:, 1], s=1)

    plt.title(title)

    plt.savefig("./diagram/{}.png".format(title), dpi=300)

    plt.show()


def draw_scatter_3D(x_data, y_data, centers, title):
    num_clusters = np.unique(y_data)

    fig = plt.figure(figsize=(10, 8))

    ax = Axes3D(fig)
    fig.add_axes(ax)

    for i in range(len(num_clusters)):
        ax.scatter(x_data[y_data == i][:, 0], x_data[y_data == i][:, 1], x_data[y_data == i][:, 2], s=1)
    for i in range(len(num_clusters)):
        ax.scatter(centers[i, 0], centers[i, 1], centers[i, 2], marker="*", s=50, c="black")

    plt.title(title)

    plt.savefig("./diagram/{}.png".format(title), dpi=300)

    plt.show()


def draw_scatter_3D_1(x_data, title):
    fig = plt.figure(figsize=(10, 8))

    ax = Axes3D(fig)
    fig.add_axes(ax)

    ax.scatter(x_data[:, 0], x_data[:, 1], x_data[:, 2], s=1)

    plt.title(title)

    plt.savefig("./diagram/{}.png".format(title), dpi=300)

    plt.show()