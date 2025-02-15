import time
import tensorflow as tf
import numpy as np
import csv, os
import argparse

from ResnetRetrain import Resnet2
from tensorflow.keras.backend import clear_session
import matplotlib.pyplot as plt

file_name = os.path.basename(__file__)
print(f"当前脚本的文件名是: {file_name}")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
sess = tf.compat.v1.Session(config=config)

SEED = 25
TRAIN_EPOCH = 30

# 从命令行读取 featureNums 参数
parser = argparse.ArgumentParser(description='Specify number of features.')
parser.add_argument('--featureNums', type=int, default=32, help='Number of features to select')
parser.add_argument('--ES', type=bool, default=False, help='Whether to use ES feather and factor')
parser.add_argument('--packetSize', type=int, default=256, help='Whether to use ES feather and factor')
parser.add_argument('--strategy', type=str, default='dp', help='Whether to use ES feather and factor')
args = parser.parse_args()

# 根据 featureNums 参数设置 selected_features
if args.ES == False:
    selected_features = [20, 19, 15, 18, 70, 25, 21, 22, 53, 17, 51, 62, 35, 56, 36, 4, 24, 44, 26, 37, 33, 13, 41, 50, 49, 16, 68, 38, 40, 14, 29, 39][:args.featureNums]
else:
    # 早停特征:
    selected_features = [15, 20, 25, 56, 18, 37, 70, 35, 53, 19, 21, 51, 17, 22, 36, 4, 49, 44, 62, 41, 68, 26, 40, 33, 45, 13, 50, 14, 52, 38, 29, 58]
WIDTHLITMIT = args.packetSize

feature_widths = [
    32, 32, 32, 32,  # fiat_mean, fiat_min, fiat_max, fiat_std
    32, 32, 32, 32,  # biat_mean, biat_min, biat_max, biat_std
    32, 32, 32, 32,  # diat_mean, diat_min, diat_max, diat_std
    32,              # duration 13
    64, 32, 32, 32, 32,  # fwin_total, fwin_mean, fwin_min, fwin_max, fwin_std
    64, 32, 32, 32, 32,  # bwin_total, bwin_mean, bwin_min, bwin_max, bwin_std
    64, 32, 32, 32, 32,  # dwin_total, dwin_mean, dwin_min, dwin_max, dwin_std
    16, 16, 16,         # fpnum, bpnum, dpnum
    32, 32, 32, 32,         # bfpnum_rate, fpnum_s, bpnum_s, dpnum_s 22
    64, 32, 32, 32, 32,  # fpl_total, fpl_mean, fpl_min, fpl_max, fpl_std
    64, 32, 32, 32, 32,  # bpl_total, bpl_mean, bpl_min, bpl_max, bpl_std
    64, 32, 32, 32, 32,  # dpl_total, dpl_mean, dpl_min, dpl_max, dpl_std
    32, 32, 32, 32,         # bfpl_rate, fpl_s, bpl_s, dpl_s  19
    16, 16, 16, 16, 16, 16, 16, 16,  # fin_cnt, syn_cnt, rst_cnt, pst_cnt, ack_cnt, urg_cnt, cwe_cnt, ece_cnt
    16, 16, 16, 16,     # fwd_pst_cnt, fwd_urg_cnt, bwd_pst_cnt, bwd_urg_cnt
    16, 16, 16,         # fp_hdr_len, bp_hdr_len, dp_hdr_len
    32, 32, 32          # f_ht_len, b_ht_len, d_ht_len 18
]
scaling_factor_value = [0.0907, 0.00, 0.0044, 1.1608, 1.2640, 0.0000, 0.0574, 0.00, 0.9546, 0.9535, 1.2512, 0.9870, 0.9581, 1.0570, 1.0325, 1.2393, 1.0356, 0.9685, 1.0092, 1.1580, 1.1437, 1.0051, 0.9036, 1.0227, 1.0778, 0.9133, 0.9975, 0.9214, 0.9366, 0.9785, 0.9234, 0.9901, 0.9476, 1.0309, 1.0111, 1.0083, 1.0355, 1.0412, 1.0127, 0.9837, 0.9980, 0.9662, 0.9913, 0.9390, 1.0201, 0.8630, 0.9452, 0.9676, 0.9760, 0.9783, 1.0139, 0.9504, 0.9292, 0.9886, 1.0013, 1.1094, 1.2258, 0.8541, 0.9693, 0.0000, 0.8995, 0.7024, 1.0598, 0.0000, 1.0703, 0.0000, 0.9571, 0.8953, 0.9689, 0.9754, 1.0697, 0.8523]
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
            if(j >= weights[i] and dp[i - 1][j - weights[i]] + values[i] > dp[i][j]):
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
            continue
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
            continue
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
    print("len",len(selected_features),"f WIDTHLITMIT: {WIDTHLITMIT}  selected_features: ", selected_features)
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
    # 获取验证集最佳模型
    saver.restore(sess, best_model_path)
    accuracy2 = model2.test()