import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 生成示例数据
# 假设有一个包含多个时间序列段的DataFrame，每个段从零开始
# 在实际应用中，你需要替换这里的数据处理步骤
data = pd.DataFrame({
    'timestamp': np.tile(np.arange(10), 5),
    'value': np.random.rand(50)
})

# 数据预处理
def preprocess_data(df):
    scaler = MinMaxScaler()
    df['scaled_value'] = scaler.fit_transform(df['value'].values.reshape(-1, 1))
    return df

data = data.groupby('timestamp').apply(preprocess_data)

# 特征工程
def create_sequences(df, sequence_length):
    sequences = []
    for i in range(len(df) - sequence_length + 1):
        seq = df.iloc[i:i+sequence_length]
        sequences.append(seq[['scaled_value']].values)
    return np.array(sequences)

sequence_length = 5  # 设置你的时间序列长度
sequences = create_sequences(data, sequence_length)

# 划分数据集
X = sequences[:, :-1, :]  # 输入序列
y = sequences[:, -1, 0]    # 输出值取最后一个时间步的值

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


'''
    Extra Function
'''


# def calculate_momentum(df, i):
#     alpha = 0.5
#     beta = 0.5
#
#     t_point_victor_col = df.loc[:i, "point_victor"]
#     if i == 0:
#         t_1_point_victor_col = df.loc[:i, "point_victor"]
#     else:
#         t_1_point_victor_col = df.loc[:i - 1, "point_victor"]
#
#     max_value = max(t_point_victor_col)
#     min_value = min(t_point_victor_col)
#
#     t_momentum = alpha * t_point_victor_col.where(t_point_victor_col == min_value).apply(lambda x: x - min_value + 1).sum() - \
#                  beta * t_point_victor_col.where(t_point_victor_col == max_value).apply(lambda  x: x - max_value + 1).sum()
#
#     t_1_momentum = alpha * t_1_point_victor_col.where(t_1_point_victor_col == min_value).apply(lambda x: x - min_value + 1).sum() - \
#                  beta * t_1_point_victor_col.where(t_1_point_victor_col == max_value).apply(lambda  x: x - max_value + 1).sum()
#
#     delta_t_momentum = t_momentum - t_1_momentum
#
#     return delta_t_momentum


