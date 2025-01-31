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

parser = argparse.ArgumentParser(description='Specify number of features.')
parser.add_argument('--packetSize', type=int, default=256, help='Whether to use ES feather and factor')
parser.add_argument('--strategy', type=str, default='dp', help='Whether to use ES feather and factor')
args = parser.parse_args()
WIDTHLITMIT = args.packetSize

# selected_features = [12]
# selected_features = [12, 48]
# selected_features = [12, 48, 9, 66]
# selected_features = [12, 48, 9, 66, 35, 38, 43, 49]
# selected_features = [12, 48, 9, 66, 35, 38, 43, 49, 26, 69, 70, 67, 16, 21, 37, 40]
# selected_features = [12, 48, 9, 66, 35, 38, 43, 49, 26, 69, 70, 67, 16, 21, 37, 40, 64, 62, 27, 42, 39, 17, 44, 22]
selected_features = [12, 48, 9, 66, 35, 38, 43, 49, 26, 69, 70, 67, 16, 21, 37, 40, 64, 62, 27, 42, 39, 17, 44, 22, 58, 71, 29, 20, 15, 34, 54, 55]

# 早停特征
# selected_features = [12, 48, 67, 9, 28, 66, 29, 43, 38, 26, 58, 49, 69, 70, 21, 16, 68, 40, 42, 37, 46, 32, 27, 64, 44, 39, 30, 62, 36, 33, 15, 17]

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
sess = tf.compat.v1.Session(config=config)

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
scaling_factor_value = [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.1198, 0.0457, 0.0374, 0.1921, 0.0360, 0.0308, 0.0506,
         0.0854, 0.0660, 0.0408, 0.0304, 0.0557, 0.0838, 0.0648, 0.0000,
         0.0130, 0.0195, 0.1067, 0.0741, 0.0001, 0.0573, 0.0000, 0.0001,
         0.0000, 0.0000, 0.0482, 0.1146, 0.0000, 0.0798, 0.1103, 0.0703,
         0.0793, 0.0389, 0.0726, 0.1088, 0.0651, 0.0000, 0.0170, 0.0295,
         0.1417, 0.1079, 0.0000, 0.0053, 0.0200, 0.0000, 0.0477, 0.0470,
         0.0385, 0.0000, 0.0615, 0.0000, 0.0177, 0.0160, 0.0751, 0.0000,
         0.0769, 0.0000, 0.1186, 0.0925, 0.0087, 0.0953, 0.0932, 0.0584]

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
    print("selected_features",selected_features,"len",len(selected_features))
    model2 = AE2(dim=len(selected_features), selected_features=selected_features,seed=SEED)
    start_time = time.time()
    for _ in range(TRAIN_EPOCH):
        model2.train()
        model2.epoch_count += 1
    model2.train_classifier()
    end_time = time.time()  # 记录训练结束时间
    total_training_time = end_time - start_time  # 计算训练总时长