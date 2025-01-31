import sys
import time
import tensorflow as tf
import numpy as np
import csv, os
from pcapVGG2 import VGG2 # 导入DNN类
import argparse
import matplotlib.pyplot as plt

# 获取当前脚本的文件名
file_name = os.path.basename(__file__)
print(f"当前脚本的文件名是: {file_name}")

SEED = 25
K = 32 # topk 特征

TRAIN_EPOCH = 30
ES_THRESHOLD = 3

parser = argparse.ArgumentParser(description='Specify number of features.')
parser.add_argument('--packetSize', type=int, default=256, help='Whether to use ES feather and factor')
parser.add_argument('--strategy', type=str, default='dp', help='Whether to use ES feather and factor')
args = parser.parse_args()

feature_widths = [32, 16, 32, 16, 32, 32, 1, 32, 32, 32, 32, 1, 32, 32, 32, 32, 32, 32, 32, 32, 1, 1, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32]
scaling_factor_value = [0.9607, 1.0098, 1.0229, 0.9839, 0.3409, 0.0000, 0.7592, 1.0127, 0.5368, 1.1380, 0.6932, 0.9426, 0.0000, 0.9923, 0.7372, 0.0000, 0.8345, 0.8512, 0.6501, 0.0000, 0.0000, 0.9374, 1.0724, 1.1263, 0.9380, 0.9290, 0.9789, 0.9866, 0.9674, 0.9865, 0.9329, 0.9778, 1.0238, 1.0059, 1.0467, 0.9951, 1.0739, 1.0046, 1.0065, 1.1395, 1.1166]

# selected_features = [39] # 无早停
# selected_features = [39, 9]
# selected_features = [39, 9, 23, 40]
# selected_features = [39, 9, 23, 40, 36, 22, 34, 32]
# selected_features = [39, 9, 23, 40, 36, 22, 34, 32, 2, 7, 1, 38, 33, 37, 35, 13]
# selected_features = [39, 9, 23, 40, 36, 22, 34, 32, 2, 7, 1, 38, 33, 37, 35, 13, 27, 29, 3, 26, 31, 28, 0, 11]
selected_features = [39, 9, 23, 40, 36, 22, 34, 32, 2, 7, 1, 38, 33, 37, 35, 13, 27, 29, 3, 26, 31, 28, 0, 11, 24, 21, 30, 25, 17, 16, 6, 14]

# selected_features = [39] 早停
# selected_features = [39, 9]
# selected_features = [39, 9, 40, 7]
# selected_features = [39, 9, 40, 7, 36, 23, 34, 14]
# selected_features = [39, 9, 40, 7, 36, 23, 34, 14, 1, 27, 22, 2, 38, 33, 28, 32]
# selected_features = [39, 9, 40, 7, 36, 23, 34, 14, 1, 27, 22, 2, 38, 33, 28, 32, 37, 11, 35, 31, 13, 21, 26, 3]
# selected_features = [39, 9, 40, 7, 36, 23, 34, 14, 1, 27, 22, 2, 38, 33, 28, 32, 37, 11, 35, 31, 13, 21, 26, 3, 25, 29, 24, 17, 30, 10, 0, 6]


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
sess = tf.compat.v1.Session(config=config)
WIDTHLITMIT = args.packetSize
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

model_dir = "./model"
curr_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
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
    print(args.strategy)
    print("len", len(selected_features), "f WIDTHLITMIT: {WIDTHLITMIT}  selected_features: ", selected_features)
    model2 = VGG2("cb_focal_loss",dim=len(selected_features), selected_features=selected_features,seed=SEED)
    print('start retraining...')
    start_time = time.time()
    # 验证集参数初始化
    best_accuracy = 0.0  # 初始化最高accuracy
    best_model_path = None
    new_folder = "model_" + curr_time
    new_folder2 = new_folder + "best_model"
    os.mkdir(os.path.join(model_dir, new_folder2))
    os.mkdir(os.path.join(model_dir, new_folder))
    saver = tf.compat.v1.train.Saver()
    for _ in range(TRAIN_EPOCH):
        accuracy = model2.train()
        model2.epoch_count += 1
        # 保存验证集最佳模型
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_path = os.path.join(model_dir, new_folder2, f"model_best_epoch_{model2.epoch_count}.ckpt")
            saver.save(model2.sess, best_model_path)
    end_time = time.time()
    total_training_time = end_time - start_time
    saver.restore(sess, best_model_path)
    accuracy2 = model2.test()