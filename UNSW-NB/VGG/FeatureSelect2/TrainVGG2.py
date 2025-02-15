import argparse
import sys
import time
import tensorflow as tf
import numpy as np
import csv, os
from VGGRetrain import VGG2 # 导入DNN类
import matplotlib.pyplot as plt

# 获取当前脚本的文件名
file_name = os.path.basename(__file__)
print(f"当前脚本的文件名是: {file_name}")

SEED = 25
K = 32 # topk 特征

TRAIN_EPOCH = 30
ES_THRESHOLD = 3
parser = argparse.ArgumentParser(description='Specify number of features.')
parser.add_argument('--featureNums', type=int, default=16, help='Number of features to select')
parser.add_argument('--ES', type=bool, default=False, help='Whether to use ES feather and factor')
parser.add_argument('--packetSize', type=int, default=256, help='Whether to use ES feather and factor')
parser.add_argument('--strategy', type=str, default='no', help='Whether to use ES feather and factor')
args = parser.parse_args()

WIDTHLITMIT = args.packetSize
feature_widths = [
    64, 16, 16, 16, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 64, 64, 64, 64, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
    32, 32, 32, 32, 32, 32, 32, 1, 32, 32, 32, 32, 1]
scaling_factor_value = [1.0860, 1.2350, 1.0333, 0.9220, 0.1548, 0.3742, 0.1788, 0.6643, 0.9358, 0.9713, 0.9804, 0.8391, 0.4930, 0.6350, 0.3378, 0.9367, 0.7206, 0.9029, 0.2443, 0.9748, 0.9618, 0.8946, 0.9154, 1.0751, 1.0522, 0.8772, 1.0443, 1.0569, 0.5200, 0.4351, 1.0352, 0.9801, 0.9404, 0.9883, 0.9963, 1.1315, 0.8631, 0.6984, 1.0195, 0.8689, 1.1190, 0.1694]

# selected_features = [1]  # 无早停
# selected_features = [1, 35]
# selected_features = [1, 35, 40, 0]
# selected_features = [1, 35, 40, 0, 23, 27, 24, 26]
# selected_features = [1, 35, 40, 0, 23, 27, 24, 26, 30, 2, 38, 34, 33, 10, 31, 19]
# selected_features = [1, 35, 40, 0, 23, 27, 24, 26, 30, 2, 38, 34, 33, 10, 31, 19, 9, 20, 32, 15, 8, 3, 22, 17]
all_features = [1, 35, 40, 0, 23, 27, 24, 26, 30, 2, 38, 34, 33, 10, 31, 19, 9, 20, 32, 15, 8, 3, 22, 17, 21, 25, 39, 36, 11, 16, 37, 7]
selected_features = all_features[:args.featureNums]
# selected_features = [39] 早停

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
sess = tf.compat.v1.Session(config=config)

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
    model2 = VGG2("cb_focal_loss",dim=len(selected_features), selected_features=selected_features,seed=SEED)
    print('start retraining...')
    start_time = time.time()
    # 验证集参数初始化
    model_dir = "./model"
    curr_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
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
    # 获取验证集最佳模型
    saver.restore(sess, best_model_path)
    accuracy2 = model2.test()