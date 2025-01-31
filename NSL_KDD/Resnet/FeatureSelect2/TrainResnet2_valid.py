# 测试没有早停策略所选特征的 模型精度
import sys
import time
import tensorflow as tf
import numpy as np
import csv, os

from pcapResnet2 import Resnet2
from tensorflow.keras.backend import clear_session
import matplotlib.pyplot as plt

file_name = os.path.basename(__file__)
print(f"当前脚本的文件名是: {file_name}")

SEED = 25
WIDTHLITMIT = 512 # 位宽限制加大
TRAIN_EPOCH = 30
# selected_features = [9]  # 无早停特征
# selected_features = [9, 39]
# selected_features = [9, 39, 36, 40]
# selected_features = [9, 39, 36, 40, 23, 37, 0, 22]
# selected_features = [9, 39, 36, 40, 23, 37, 0, 22, 31, 18, 30, 16, 1, 10, 2, 14]
# selected_features = [9, 39, 36, 40, 23, 37, 0, 22, 31, 18, 30, 16, 1, 10, 2, 14, 34, 35, 32, 27, 33, 28, 3, 21]
selected_features = [9, 39, 36, 40, 23, 37, 0, 22, 31, 18, 30, 16, 1, 10, 2, 14, 34, 35, 32, 27, 33, 28, 3, 21, 26, 24, 38, 13, 25, 17, 11, 29]

# selected_features = [9] # 早停策略下的特征
# selected_features = [9, 39]
# selected_features = [9, 39, 40, 36]
# selected_features = [9, 39, 40, 36, 7, 22, 2, 31]
# selected_features = [9, 39, 40, 36, 7, 22, 2, 31, 23, 1, 10, 37, 30, 27, 33, 0]
# selected_features = [9, 39, 40, 36, 7, 22, 2, 31, 23, 1, 10, 37, 30, 27, 33, 0, 18, 14, 28, 35, 38, 21, 34, 32]
# selected_features = [9, 39, 40, 36, 7, 22, 2, 31, 23, 1, 10, 37, 30, 27, 33, 0, 18, 14, 28, 35, 38, 21, 34, 32, 3, 24, 25, 6, 13, 17, 26, 11]


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
sess = tf.compat.v1.Session(config=config)

feature_widths = [32, 3, 70, 11, 32, 32, 1, 32, 32, 32, 32, 1, 32, 32, 32, 32, 32, 32, 32, 32, 1, 1, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32]

scaling_factor_value = [1.0487, 1.0259, 1.0195, 0.9803, 0.7445, 0.0000, 0.8669, 0.8538, 0.7943, 1.1773, 1.0249, 0.9313, 0.00001, 0.9624, 1.0129, 0.2961, 1.0282, 0.9346, 1.0377, 0.00001, 0.00001, 0.9793, 1.0486, 1.0727, 0.9709, 0.9496, 0.9786, 1.0023, 0.9832, 0.9296, 1.0292, 1.0410, 1.0044, 0.9965, 1.0114, 1.0107, 1.1005, 1.0498, 0.9642, 1.1563, 1.0844]

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
    top_k_values = np.array(scaling_factor_value)[selected_features] # 获取对应因子值
    top_k_weights = np.array(feature_widths)[selected_features]  # 获取对应特征的位宽
    top_k_values = np.insert(top_k_values, 0, -1)
    top_k_weights = np.insert(top_k_weights, 0, -1)
    print("top_k_values",top_k_values)
    print("top_k_weights",top_k_weights)
    selected_indices = knapsack(top_k_values, top_k_weights, WIDTHLITMIT)
    selected_features = [selected_features[i - 1] for i in selected_indices]
    # 无背包算法 如下
    print("f WIDTHLITMIT: {WIDTHLITMIT}  selected_features: ", selected_features,"len",len(selected_features))
    model2 = Resnet2(dim=len(selected_features), selected_features=selected_features, seed=SEED)
    print('start retraining...')
    start_time = time.time()
    for _ in range(TRAIN_EPOCH):
        delta_loss, count = model2.train()
        model2.epoch_count += 1
    end_time = time.time()
    total_training_time = end_time - start_time
    print("dnn2_loss_history", model2.loss_history)
    print("dnn2_macro_F1List", model2.macro_F1List)
    print("dnn2_micro_F1List", model2.micro_F1List)
    print('start testing...')
    accuracy2 = model2.test()