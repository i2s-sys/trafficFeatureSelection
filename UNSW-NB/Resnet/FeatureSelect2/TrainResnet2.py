import sys
import time
import tensorflow as tf
import numpy as np
import csv, os
import argparse

from pcapResnet2 import Resnet2
from tensorflow.keras.backend import clear_session
import matplotlib.pyplot as plt

file_name = os.path.basename(__file__)
print(f"当前脚本的文件名是: {file_name}")

SEED = 25

TRAIN_EPOCH = 30

# 从命令行读取 featureNums 参数
parser = argparse.ArgumentParser(description='Specify number of features.')
parser.add_argument('--featureNums', type=int, default=32, help='Number of features to select')
parser.add_argument('--Q', type=int, default=3, help='Number of Q')
parser.add_argument('--packetSize', type=int, default=1024, help='Whether to use ES feather and factor')
parser.add_argument('--strategy', type=str, default='no', help='Whether to use ES feather and factor')
parser.add_argument('--device', type=int, default='1', help='Whether to use ES feather and factor')
args = parser.parse_args()

if(args.Q ==3): # 8
    all_features = [
        1, 24, 35, 40, 0, 27, 30, 33, 23, 2, 26, 8, 32, 38, 39, 20,
        17, 25, 10, 9, 19, 34, 3, 4, 15, 11, 36, 31, 37, 21, 16, 22
    ]
    scaling_factor_value = [1.1003, 1.2429, 1.0352, 0.9458, 0.9355, 0.8145, -0.0000, 0.4196, 1.0261, 0.9696, 0.9778,
                            0.9273, 0.6965, 0.5279, 0.6262, 0.9291, 0.8593, 0.9931, 0.7850, 0.9671, 1.0083, 0.8682,
                            0.8565, 1.0461, 1.1745, 0.9787, 1.0313, 1.0808, 0.7615, 0.6339, 1.0587, 0.8804, 1.0260,
                            1.0531, 0.9667, 1.1717, 0.9172, 0.8776, 1.0257, 1.0139, 1.1097, 0.5797]
elif(args.Q ==2): # 4 ES
    all_features = [1, 35, 40, 24, 30, 27, 23, 26, 2, 8, 9, 32, 38, 10, 19, 3, 33, 0, 4, 20, 34, 15, 25, 5, 11, 17, 22, 41, 21, 31, 37, 18]
    scaling_factor_value = [
    0.9861, 1.1277, 1.0054, 0.9936, 0.9859, 0.9593,
    0.8670, 0.8302, 1.0049, 1.0030, 0.9971, 0.9555,
    0.8107, 0.8785, 0.8140, 0.9760, 0.8991, 0.9518,
    0.9204, 0.9942, 0.9773, 0.9406, 0.9506, 1.0142,
    1.0450, 0.9629, 1.0061, 1.0264, 0.7730, 0.7861,
    1.0375, 0.9302, 1.0010, 0.9929, 0.9766, 1.1182,
    0.9102, 0.9248, 1.0001, 0.9172, 1.1080, 0.9424]
elif(args.Q ==4): # 9 ES
    all_features = [35, 1, 40, 24, 30, 27, 2, 26, 23, 38, 9, 0, 8, 32, 10, 19, 3, 33, 20, 25, 34, 15, 39, 22, 31, 21, 17, 37, 36, 11, 5, 4]
    scaling_factor_value = [
    0.9918, 1.1626, 1.0251, 0.9713, 0.8198, 0.8367,
    0.1923, 0.5949, 0.9883, 0.9954, 0.9798, 0.8499,
    0.6583, 0.7569, 0.5917, 0.9365, 0.7948, 0.8881,
    0.5092, 0.9792, 0.9564, 0.8985, 0.9177, 1.0211,
    1.0807, 0.9508, 1.0241, 1.0301, 0.7329, 0.5385,
    1.0446, 0.9039, 0.9823, 0.9670, 0.9381, 1.1672,
    0.8832, 0.8839, 1.0151, 0.9289, 1.1151, 0.6254]
elif(args.Q ==5): # 10
    all_features = [35, 1, 40, 24, 27, 23, 26, 30, 38, 2, 0, 8, 9, 33, 10, 19, 32, 3, 20, 25, 34, 22, 15, 39, 31, 11, 37, 17, 21, 36, 4, 5]
    scaling_factor_value = [
    1.0155, 1.1486, 1.0266, 0.9606, 0.8491, 0.8106,
    0.1950, 0.4992, 0.9953, 0.9890, 0.9807, 0.8999,
    0.7422, 0.7757, 0.4668, 0.9227, 0.7549, 0.8955,
    0.6354, 0.9750, 0.9536, 0.8662, 0.9250, 1.0319,
    1.0814, 0.9496, 1.0310, 1.0508, 0.7189, 0.4740,
    1.0309, 0.9073, 0.9739, 0.9824, 0.9368, 1.1571,
    0.8634, 0.8984, 1.0282, 0.9117, 1.1245, 0.1880]
elif(args.Q ==6): # 14
    all_features = [1, 35, 40, 24, 27, 30, 0, 26, 2, 23, 33, 38, 32, 9, 10, 19, 8, 20, 25, 3, 15, 39, 34, 17, 22, 11, 31, 21, 36, 37, 16, 4]
    scaling_factor_value = [1.0445, 1.1856, 1.0256, 0.9357, 0.7373, 0.5878, 0.0000, 0.2858, 0.9685, 0.9734, 0.9702, 0.8759, 0.5076, 0.6166, 0.2814, 0.9297, 0.7688, 0.9122, 0.4976, 0.9690, 0.9573, 0.8664, 0.8804, 1.0248, 1.0913, 0.9445, 1.0297, 1.0657, 0.6712, 0.3468, 1.0472, 0.8718, 0.9854, 1.0038, 0.9202, 1.1638, 0.8562, 0.8557, 0.9944, 0.9271, 1.1432, 0.3054]
# 根据 featureNums 参数设置 selected_features 因为24精度最高
selected_features = all_features[:args.featureNums]

os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.device}"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
sess = tf.compat.v1.Session(config=config)

feature_widths = [
    64, 16, 16, 16, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 64, 64, 64, 64, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
    32, 32, 32, 32, 32, 32, 32, 1, 32, 32, 32, 32, 1]
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
    print("total_weight", total_weight, "keep[n][max_weight]", keep[n][max_weight])
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
WIDTHLITMIT = args.packetSize
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
    print("len", len(selected_features), "f WIDTHLITMIT: {WIDTHLITMIT} selected_features: ", selected_features)
    model2 = Resnet2(dim=len(selected_features), selected_features=selected_features, seed=SEED)
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
    print("dnn2_loss_history", model2.loss_history)
    print("dnn2_macro_F1List", model2.macro_F1List)
    print("dnn2_micro_F1List", model2.micro_F1List)
    print('start testing...')
    # 获取验证集最佳模型
    saver.restore(sess, best_model_path)
    accuracy2 = model2.test()
