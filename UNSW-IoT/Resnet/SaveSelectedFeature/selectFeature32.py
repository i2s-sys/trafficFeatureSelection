import pandas as pd

# 保留的特征下标和标签的位置
important_features = [10, 15, 25, 64, 20, 44, 42, 16, 21, 26, 35, 22, 39, 43, 17,  9, 27, 31, 45,  6, 14, 18, 12, 67, 49, 24, 38, 40, 19, 66, 68, 48]
label_index = 72  # 标签的下标位置，第73列

# 加载训练和测试数据
train_data = pd.read_csv('../train_data.csv', header=None)
test_data = pd.read_csv('../test_data.csv', header=None)

# 选择重要特征和标签
important_features.append(label_index)  # 保留标签
train_data_new = train_data.iloc[:, important_features]
test_data_new = test_data.iloc[:, important_features]

# 保存新的数据到文件
train_data_new.to_csv('train_data_new.csv', index=False, header=False)
test_data_new.to_csv('test_data_new.csv', index=False, header=False)

print("新的训练数据和测试数据已保存。")
