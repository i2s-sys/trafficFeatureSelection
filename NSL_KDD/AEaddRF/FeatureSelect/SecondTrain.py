# 使用早停策略
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
WIDTHLITMIT = 512 # 位宽限制加大
TRAIN_EPOCH = 30
ES_THRESHOLD = 3

feature_widths = [32, 16, 32, 16, 32, 32, 1, 32, 32, 32, 32, 1, 32, 32, 32, 32, 32, 32, 32, 32, 1, 1, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32]
scaling_factor_value = [0.0408, 0.3373, 0.2965, 0.4220, 0.0000, 0.0000, 0.0021, 0.0547, 0.0000, 0.0429, 0.0000, 0.2912, 0.0000, 0.0389, 0.0440, 0.0000, 0.0671, 0.0002, 0.0800, 0.0000, 0.0000, 0.0598, 0.1516, 0.0916, 0.2123, 0.2095, 0.1414, 0.1317, 0.3981, 0.0986, 0.1326, 0.4273, 0.3334, 0.3613, 0.1159, 0.1613, 0.0764, 0.2042, 0.2044, 0.1370, 0.1230]

# selected_features = [31]
# selected_features = [31,  3]
# selected_features = [31,  3, 28, 33]
# selected_features = [31,  3, 28, 33,  1, 32,  2, 11]
# selected_features = [31,  3, 28, 33,  1, 32,  2, 11, 24, 25, 38, 37, 35, 22, 26, 39]
# selected_features = [31,  3, 28, 33,  1, 32,  2, 11, 24, 25, 38, 37, 35, 22, 26, 39, 30, 27, 40, 34, 29, 23, 18, 36]
selected_features = [31,  3, 28, 33,  1, 32,  2, 11, 24, 25, 38, 37, 35, 22, 26, 39, 30, 27, 40, 34, 29, 23, 18, 36, 16, 21,  7, 14,  9,  0, 13,  6]

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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

if __name__ == '__main__':
    top_k_values = np.array(scaling_factor_value)[selected_features]  # 获取对应因子值
    top_k_weights = np.array(feature_widths)[selected_features]  # 获取对应特征的位宽
    top_k_values = np.insert(top_k_values, 0, -1)
    top_k_weights = np.insert(top_k_weights, 0, -1)
    print("top_k_values", top_k_values)
    print("top_k_weights", top_k_weights)
    selected_indices = knapsack(top_k_values, top_k_weights, WIDTHLITMIT)
    selected_features = [selected_features[i - 1] for i in selected_indices]
    # 无背包算法如下：
    print("selected_features",selected_features)
    model2 = AE2(dim=len(selected_features), selected_features=selected_features,seed=SEED)
    start_time = time.time()
    for _ in range(TRAIN_EPOCH):
        model2.train()
        model2.epoch_count += 1
    model2.train_classifier()
    end_time = time.time()  # 记录训练结束时间
    total_training_time = end_time - start_time  # 计算训练总时长