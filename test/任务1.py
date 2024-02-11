import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

# 创建包含列表的DataFrame
data = {'col1': [[1, 2, 3], [4, 5, 6], [7, 8, 9, 10]]}
df = pd.DataFrame(data)

# 使用explode将列表拆分成多行
df_exploded = df.explode('col1')

# 根据索引进行分组，并将每个组中的元素组合成一个新的列表
df_grouped = df_exploded.groupby(df_exploded.index)['col1'].apply(list)

# 创建一个新的DataFrame
result_df = pd.DataFrame(df_grouped.tolist(), columns=['col1'])

print(result_df)


# 生成演示用的随机数据
np.random.seed(42)
n_points = 100
players = ['选手A', '选手B']
points_won = np.random.choice(players, size=n_points)
game_scores = np.random.choice(['15-0', '15-15', '30-15', '30-30', '40-30', '40-40', '选手A占优势', '选手B占优势', '选手A获胜', '选手B获胜'], size=n_points)

# 创建DataFrame
df = pd.DataFrame({'Point_Winner': points_won, 'Game_Score': game_scores})

# 根据比分计算性能分数的函数
def calculate_performance_score(game_score):
    if '占优势' in game_score:
        return 2
    elif '获胜' in game_score:
        return 3
    else:
        return 1

# 计算每个点的性能分数
df['Performance_Score'] = df['Game_Score'].apply(calculate_performance_score)

# 计算每个选手的累积性能分数
df['选手A分数'] = np.where(df['Point_Winner'] == '选手A', df['Performance_Score'], 0).cumsum()
df['选手B分数'] = np.where(df['Point_Winner'] == '选手B', df['Performance_Score'], 0).cumsum()

# 绘制比赛流程图
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['选手A分数'], label='选手A')
plt.plot(df.index, df['选手B分数'], label='选手B')
plt.title('比赛流程')
plt.xlabel('点数')
plt.ylabel('累积性能分数')
plt.legend()
plt.show()
