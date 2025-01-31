import csv
import pandas as pd
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

train_data = []
test_data = []
label_status = {}
filename = '../train_data2.csv'
csv_reader = csv.reader(open(filename))
label_data = {i: [] for i in range(29)}

for row in csv_reader:
    data = [0 if char == 'None' else np.float32(char) for char in row]
    label = int(data[-1])
    label_data[label].append(data)
    if str(label) in label_status:
        label_status[str(label)] += 1
    else:
        label_status[str(label)] = 1

train_data = sum(label_data.values(), [])
filename = '../test_data2.csv'
csv_reader = csv.reader(open(filename))
for row in csv_reader:
    data = [0 if char == 'None' else np.float32(char) for char in row]
    test_data.append(data)
train_length = len(train_data)
test_length = len(test_data)
  
print('init data completed!')

def load_data(datas):
    data = np.delete(datas, -1, axis=1)
    label = np.array(datas, dtype=np.int32)[:, -1]
    return data, label
# 加载数据
X_train, y_train = load_data(train_data)
X_test, y_test = load_data(test_data)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# 预测
y_pred = knn.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

macro_f1 = f1_score(y_test, y_pred, average='macro')
micro_f1 = f1_score(y_test, y_pred, average='micro')
print(f"Macro-F1: {macro_f1}")
print(f"Micro-F1: {micro_f1}")
