import matplotlib.pyplot as plt
import matplotlib

# 设置字体为Times New Roman并加粗
# plt.rcParams['font.family'] = '微软雅黑'
plt.rcParams['font.weight'] = 'bold'

# 数据数组
data = [0, 30, 30, 30, 32, 30, 31, 32, 31, 31, 32, 31, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 31, 32]

# 找到第一次出现32的位置
first_occurrence = data.index(32)

# 创建图形
plt.plot(data, marker='o', linestyle='-', linewidth=2)

# 标记第一次出现32的地方
plt.axvline(x=first_occurrence, color='red', linestyle='--', linewidth=2, label=f'EarlyStop at epoch {first_occurrence}')
# plt.text(first_occurrence, max(data), 'Stop training', color='red', verticalalignment='bottom', horizontalalignment='right', weight='bold')

# 设置标签
plt.xlabel('epoch', weight='bold')
plt.ylabel('TSM', weight='bold')
plt.legend()

# 显示图形
plt.show()
