import numpy as np
import sklearn.metrics
from sklearn.cluster import KMeans
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo

from coding.llh.visualization.draw_heat_map import draw_heat_map
from coding.llh.visualization.draw_scatter import draw_scatter_2D, draw_scatter_2D_1, draw_scatter_3D_1, draw_scatter_3D


# K-means
def k_means(array: np.ndarray):
    info = {}

    draw_scatter_2D_1(array, "2D scatter data before k-means")
    draw_scatter_3D_1(array, "3D scatter data before k-means")

    K = 60

    info["Number of clustering centers"] = K

    k_means_model = KMeans(n_clusters=K, init='k-means++')

    k_means_model.fit(array)

    sum_of_squared_errors = k_means_model.inertia_

    info["SSE"] = sum_of_squared_errors

    draw_scatter_2D(array, k_means_model.labels_, k_means_model.cluster_centers_, "2D scatter data after k-means")
    draw_scatter_3D(array, k_means_model.labels_, k_means_model.cluster_centers_, "3D scatter data after k-means")

    result = k_means_model.fit_predict(array[:200])

    silhouette_score = sklearn.metrics.silhouette_score(array[:200], result)

    info["Silhouette score"] = silhouette_score

    return info


# Bartlett sphericity test
def bartlett_test(df):
    _, p_value = calculate_bartlett_sphericity(df)

    return p_value


# KMO test
def kmo_test(df):
    _, kmo_score = calculate_kmo(df)

    return kmo_score


# Principal component analysis
def pca(df):
    # Only consider the correlation of the independent variables
    info = {}

    # array_x = df.iloc[:, 1:]
    array_x = df.iloc[:, :]
    array_y = df.iloc[:, :1]

    # Bartlett sphericity test
    p_value = bartlett_test(array_x)
    info["p value of bartlett sphericity test"] = p_value
    if p_value < 0.05:
        info["Result of bartlett sphericity test"] = "Accept"
    else:
        info["Result of bartlett sphericity test"] = "Reject"

    # KMO test
    kmo_score = kmo_test(array_x)
    info["Score of KMO test"] = kmo_score
    if kmo_score > 0.5:
        info["Result of KMO test"] = "Accept"
    else:
        info["Result of KMO test"] = "Reject"

    # get the matrix of correlation coefficients
    covX = np.around(np.corrcoef(array_x.T), decimals=3)

    # 计算协方差矩阵的对角线元素的标准差
    std_dev = np.sqrt(np.diag(covX))

    # 计算皮尔逊相关系数矩阵
    pearson_matrix = covX / np.outer(std_dev, std_dev)

    # draw_heat_map(pearson_matrix, "pearson matrix", True, df.columns.values)

    # Solve the eigenvalues and eigenvectors of the coefficient correlation matrix
    eigenvalues, eigenvectors = np.linalg.eig(covX.T)

    eigenvalues = np.around(eigenvalues, decimals=3)

    eigenvalues_dict = dict(zip(eigenvalues.tolist(), list(range(0, len(eigenvalues)))))

    # Sort feature values in descending order
    eigenvalues = sorted(eigenvalues, reverse=True)

    for i, value in enumerate(eigenvalues):
        if i == 0:
            sorted_eigenvectors = eigenvectors[:, eigenvalues_dict[value]].reshape(-1, 1)
        else:
            sorted_eigenvectors = np.concatenate((sorted_eigenvectors, eigenvectors[:, eigenvalues_dict[value]].reshape(-1, 1)), axis=1)

    # draw_line_graph(range(1, len(eigenvalues) + 1), eigenvalues, "Eigenvalue")

    # get the contribution of the eigenvalues
    contribution = eigenvalues / np.sum(eigenvalues)

    # get the cumulative contribution of the eigenvalues
    cumulative_contribution = np.cumsum(contribution)

    # Selection of principal components
    main_factors_index = [i for i in range(len(cumulative_contribution)) if cumulative_contribution[i] < 0.80]

    main_factor_num = len(main_factors_index)

    info["Main factor num"] = main_factor_num

    # Get the projection matrix
    projected_array = array_x.dot(sorted_eigenvectors[:, :main_factor_num])
    projected_array = np.concatenate((array_y.values, projected_array), axis=1)

    return projected_array, info



