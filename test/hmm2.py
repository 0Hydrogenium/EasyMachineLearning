import numpy as np
from hmmlearn import hmm

# 观测序列
obs_sequences = np.array([
    np.array([0, 1, 2, 3]),  # 代表A、B、C、D中的一个
    np.array([3, 2, 1, 0]),
    np.array([0, 0, 1, 1]),
])

# 隐状态
hidden_states = 5

# 构建并训练HMM模型
model = hmm.MultinomialHMM(n_components=hidden_states, n_iter=100,  tol=0.01, random_state=42)
obs_lengths = np.array([len(seq) for seq in obs_sequences])
obs_concatenated = np.concatenate(obs_sequences)
model.fit(obs_concatenated.reshape(-1, 1))

# 获取训练后的模型参数
start_prob = model.startprob_
transition_prob = model.transmat_
emission_prob = model.emissionprob_

# 提供的预测用的观测序列
provided_observation_sequence = np.array([0, 2, 2, 3]).reshape(-1, 1)

# 将提供的观测序列中的值映射到训练数据的范围内
# provided_observation_sequence = np.clip(provided_observation_sequence, 0, 3)

# 预测隐状态
predicted_hidden_states = model.predict(provided_observation_sequence, [4])

print("提供的观测序列:", provided_observation_sequence.flatten())
print("预测的隐状态序列:", predicted_hidden_states)
