# 使用早停策略
import argparse
import sys
import time
import tensorflow as tf
import numpy as np
import csv, os
from AEAddRF2 import AE2 # 导入DNN类
import matplotlib.pyplot as plt

# 获取当前脚本的文件名
file_name = os.path.basename(__file__)
print(f"当前脚本的文件名是: {file_name}")
SEED = 25
K = 32 # topk 特征
TRAIN_EPOCH = 30
ES_THRESHOLD = 3

feature_widths = [ 64, 16, 16, 16, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 64, 64, 64, 64, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
    32, 32, 32, 32, 32, 32, 32, 1, 32, 32, 32, 32, 1]
scaling_factor_value =[0.0410, 0.0396, 0.0771, 0.0433, 0.0294, 0.0006, -0.0000, 0.0004, 0.0463, 0.1864, 0.1226, 0.0176, 0.0344, 0.0001, 0.0313, 0.0185, 0.0120, 0.0253, 0.0286, 0.0892, 0.0912, 0.0909, 0.0624, 0.0000, 0.0333, 0.0313, 0.0521, 0.0550, -0.0000, 0.0076, 0.0631, 0.0622, 0.0580, 0.0553, 0.0492, 0.0608, 0.0139, -0.0000, 0.0293, 0.0572, 0.0638, 0.0125]

# selected_features= [9]
# selected_features= [9, 10]
# selected_features= [9, 10, 20, 21]
# selected_features= [9, 10, 20, 21, 19,  2, 40, 30]
# selected_features= [9, 10, 20, 21, 19,  2, 40, 30, 22, 31, 35, 32, 39, 33, 27, 26]
# selected_features= [9, 10, 20, 21, 19,  2, 40 ,30, 22, 31, 35, 32, 39, 33, 27, 26, 34,  8,  3,  0,  1, 12, 24, 14]
all_features = [9, 10, 20, 21, 19,  2, 40, 30, 22, 31, 35, 32, 39, 33, 27, 26, 34,  8 , 3,  0,  1, 12, 24, 14, 25, 4, 38, 18, 17, 15, 11, 36]

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
parser = argparse.ArgumentParser(description='Specify number of features.')
parser.add_argument('--featureNums', type=int, default=16, help='Number of features to select')
parser.add_argument('--ES', type=bool, default=False, help='Whether to use ES feather and factor')
parser.add_argument('--packetSize', type=int, default=256, help='Whether to use ES feather and factor')
parser.add_argument('--strategy', type=str, default='no', help='Whether to use ES feather and factor')
args = parser.parse_args()
WIDTHLITMIT = args.packetSize
selected_features = all_features[:args.featureNums]
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
sess = tf.compat.v1.Session(config=config)
curr_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
def knapsack(values, weights, max_weight):
    n = len(values) - 1 # 真实物品的个数
    dp = np.zeros((n + 1, max_weight + 1))
    keep = np.empty((n + 1, max_weight + 1), dtype=object)
    for i in range(n + 1):
        for j in range(max_weight + 1):
            keep[i][j] = []

    for i in range(1, n + 1):
        for j in range(0, max_weight + 1):
            dp[i][j] = dp[i - 1][j]
            keep[i][j] = keep[i - 1][j].copy()
            if(j >= weights[i]):
                if(dp[i - 1][j - weights[i]] + values[i] > dp[i][j]):
                    dp[i][j] = dp[i - 1][j - weights[i]] + values[i]
                    keep[i][j] = keep[i - 1][j - weights[i]].copy()
                    keep[i][j].append(i)
    total_weight = sum(weights[i] for i in keep[n][max_weight])
    print("total_weight",total_weight,"keep[n][max_weight]",keep[n][max_weight])
    return keep[n][max_weight]
def greed1(values, weights, max_weight):
    # 将物品按体积从小到大排序
    items = sorted(range(len(weights)), key=lambda i: weights[i])
    total_weight = 0
    selected_items = []
    for i in items:
        if total_weight + weights[i] <= max_weight:
            selected_items.append(i)
            total_weight += weights[i]
        else:
            break
    return selected_items
def greed2(values, weights, max_weight):
    # 将物品按价值从大到小排序
    items = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
    total_weight = 0
    selected_items = []
    for i in items:
        if total_weight + weights[i] <= max_weight:
            selected_items.append(i)
            total_weight += weights[i]
        else:
            break
    return selected_items

top_k_values = np.array(scaling_factor_value)[selected_features] # 获取对应因子值
top_k_weights = np.array(feature_widths)[selected_features]  # 获取对应特征的位宽
print("top_k_values",top_k_values)
print("top_k_weights",top_k_weights)
if(args.strategy == 'dp'):
    top_k_values = np.insert(top_k_values, 0, -1)
    top_k_weights = np.insert(top_k_weights, 0, -1)
    selected_indices = knapsack(top_k_values, top_k_weights, WIDTHLITMIT)
    selected_features = [selected_features[i - 1] for i in selected_indices]
elif(args.strategy == 'greed1'):
    selected_indices = greed1(top_k_values, top_k_weights, WIDTHLITMIT)
    selected_features = [selected_features[i] for i in selected_indices]
elif(args.strategy == 'greed2'):
    selected_indices = greed2(top_k_values, top_k_weights, WIDTHLITMIT)
    selected_features = [selected_features[i] for i in selected_indices]

if __name__ == '__main__':
    print("nums:",len(selected_features),"selected_features",selected_features)
    model2 = AE2(dim=len(selected_features), selected_features=selected_features,seed=SEED)
    start_time = time.time()
    for _ in range(TRAIN_EPOCH):
        model2.train()
        model2.epoch_count += 1
    model2.train_classifier()
    end_time = time.time()  # 记录训练结束时间
    total_training_time = end_time - start_time  # 计算训练总时长