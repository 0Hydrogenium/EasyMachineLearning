import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture


def gaussian_mix(x):
    x = x.reshape(-1, 1)
    n_components = 2000  # 你可以根据需要调整混合组件的数量
    gmm = GaussianMixture(n_components=n_components, covariance_type='full')

    # 拟合模型
    gmm.fit(x)

    # 预测每个数据点所属的组件
    continuous_data = gmm.sample(len(x))[0].reshape(-1)

    return continuous_data

    # 使用高斯混合模型拟合数据
    # gmm = GaussianMixture(n_components=50)  # 选择混合成分的数量
    # gmm.fit(x.reshape(-1, 1))

    # 生成连续数据
    # return np.linspace(min(x), max(x), len(x)).flatten()

    # z = np.exp(gmm.score_samples(y.reshape(-1, 1)))

    # return z
