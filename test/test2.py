import numpy as np
from sklearn.mixture import GaussianMixture

# 输入离散数据
discrete_data = np.array([2, 2, 2, 5, 5, 5, 2, 2, 2, 1, 1, 3, 4]).reshape(-1, 1)

# 定义高斯混合模型
n_components = 10  # 你可以根据需要调整混合组件的数量
gmm = GaussianMixture(n_components=n_components, covariance_type='full')

# 拟合模型
gmm.fit(discrete_data)

# 预测每个数据点所属的组件
continuous_data = gmm.sample(len(discrete_data))[0].reshape(-1)

# 输出结果
print("离散数据:", discrete_data.flatten())
print("连续数据:", continuous_data)

a = discrete_data.flatten().reshape(-1, 1)
b = continuous_data.reshape(-1, 1)
c = np.concatenate((a, b), axis=1)
pass
