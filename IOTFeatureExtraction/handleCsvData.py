import pandas as pd

# 读取CSV文件
df = pd.read_csv('C_feature.csv')

# 计算总行数
total_rows = len(df)

# 计算每个标签的出现次数
label_counts = df.iloc[:, -1].value_counts()

# 数量小于10行数
threshold = 10 #

# 找出出现次数少于阈值的标签
labels_to_drop = label_counts[label_counts < threshold].index.tolist()

# 删除这些标签的数据
df = df[~df.iloc[:, -1].isin(labels_to_drop)]

# 再次计算每个标签的出现次数
final_label_counts = df.iloc[:, -1].value_counts()

# 输出被删除的标签和剩下的标签及其出现次数
print("删除的标签:", labels_to_drop)
print("剩下的标签及其出现次数:")
print(final_label_counts)

# 将处理后的数据保存到新的CSV文件
df.to_csv('processed_file.csv', index=False)
