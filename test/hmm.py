import numpy as np
import pandas as pd
from hmmlearn import hmm


# 示例数据
# states = ['E', 'F', 'G', 'H', 'I']
observations = np.array([[1, 3, 2, 4], [1, 1, 2, 3], [3, 2, 2, 1], [3, 4, 2, 1], [3, 2, 2, 1], [1, 2, 4, 4], [3, 2, 2, 2]])

# 一种常见的方法是使用Baum-Welch算法，也称为Expectation-Maximization（EM）算法，它是一种用于训练隐马尔科夫模型的无监督学习算法。该算法通过迭代估计模型参数，包括初始状态概率、状态转移概率矩阵和观测状态生成概率矩阵
hidden_markov_model = hmm.MultinomialHMM(n_components=5, n_iter=100, tol=0.01)

# 使用Baum-Welch算法训练模型
# em.fit(observations.reshape(-1, 1))
length = np.array([len(seq) for seq in observations])

hidden_markov_model.fit(observations)

# 获取训练后的模型参数
start_prob = hidden_markov_model.startprob_
transition_prob = hidden_markov_model.transmat_
emission_prob = hidden_markov_model.emissionprob_

print("初始状态概率分布:", start_prob)
print("状态转移概率矩阵:")
print(transition_prob)
print("观测状态生成概率矩阵:")
print(emission_prob)

length1 = np.array([4])
predicted_states = hidden_markov_model.decode(np.array([[1, 2, 1, 1]]))
# predicted_states1 = em.predict(np.array([[1, 2, 1, 1]]).reshape(-1, 1), length1)

# start_prob = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
# transition_prob = np.array([[0.2, 0.2, 0.2, 0.2, 0.2],
#                             [0.2, 0.2, 0.2, 0.2, 0.2],
#                             [0.2, 0.2, 0.2, 0.2, 0.2],
#                             [0.2, 0.2, 0.2, 0.2, 0.2],
#                             [0.2, 0.2, 0.2, 0.2, 0.2]])
# emission_prob = np.array([[0.25, 0.25, 0.25, 0.25],
#                           [0.25, 0.25, 0.25, 0.25],
#                           [0.25, 0.25, 0.25, 0.25],
#                           [0.25, 0.25, 0.25, 0.25],
#                           [0.25, 0.25, 0.25, 0.25]])

# observations = observations.tolist()[0]

observations = ["A", "C", "B", "D", "D", "C", "B", "A"]

# 创建HMM模型
hmm = HiddenMarkovModel(states, observations, start_prob, transition_prob, emission_prob)

# 创建示例数据的DataFrame
data = pd.DataFrame({'Observations': observations})

# 预测最可能的状态序列
predicted_states = hmm.decode(data['Observations'])
print("Predicted States:", predicted_states)
