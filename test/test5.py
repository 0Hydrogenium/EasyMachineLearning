import matplotlib.pyplot as plt
from matplotlib.table import Table

# 数据
data = [
    ["Header 1", "Header 2", "Header 3"],
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

# 创建子图
fig, ax = plt.subplots()

# 隐藏坐标轴
ax.axis('off')

# 创建表格
table = Table(ax, loc='center', cellText=data, colLabels=None, cellLoc='center')

# 添加表格到子图
ax.add_table(table)

# 调整表格布局
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)

# 保存图片
plt.savefig('table_image.png', bbox_inches='tight', pad_inches=0.05)
plt.show()
