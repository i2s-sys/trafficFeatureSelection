# 测试没有早停策略所选特征的 模型精度
import sys
import time
from xml.sax.handler import all_features

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

# 从命令行读取 featureNums 参数
parser = argparse.ArgumentParser(description='Specify number of features.')
parser.add_argument('--featureNums', type=int, default=32, help='Number of features to select')
parser.add_argument('--ES', type=bool, default=True, help='Whether to use ES Feature')
parser.add_argument('--packetSize', type=int, default=1024, help='packetSize')
parser.add_argument('--TRAIN_EPOCH', type=int, default=30, help='TRAIN_EPOCH')
parser.add_argument('--Q', type=int, default=3, help='Q')
parser.add_argument('--strategy', type=str, default='dp', help='Whether to use ES feather and factor')
args = parser.parse_args()
feature_widths = [32, 16, 32, 16, 32, 32, 1, 32, 32, 32, 32, 1, 32, 32, 32, 32, 32, 32, 32, 32, 1, 1, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32]
# 根据 featureNums 参数设置 selected_features
TRAIN_EPOCH = args.TRAIN_EPOCH
# 早停策略下的特征
if(args.ES == True):
    if(args.Q ==3): # 12
        all_feature = [9, 39, 40, 36, 7, 22, 2, 31, 23, 1, 10, 37, 30, 27, 33, 0, 18, 14, 28, 35, 38, 21, 34, 32, 3, 24, 25, 6, 13, 17, 26, 11][:args.featureNums]
        scaling_factor_value = [1.0279, 1.0202, 1.0148, 0.9838, 0.9803, 0.6089,
            0.9670, 1.0053, 0.9449, 1.1440, 1.0337, 0.9602,
            0.7815, 0.9689, 1.0077, 0.7839, 0.9778, 0.9690,
            1.0152, 0.0159, 0.1307, 0.9882, 1.0393, 1.0355,
            0.9809, 0.9710, 0.9847, 1.0090, 1.0016, 0.9302,
            1.0196, 1.0266, 0.9898, 1.0062, 0.9982, 1.0009,
            1.0626, 1.0371, 0.9793, 1.1092, 1.0585
        ]
    elif(args.Q ==2): # 4
        all_feature =  [
    9, 39, 40, 36, 7, 22, 2, 31, 1, 10, 23, 33, 37, 27, 30, 38, 28, 35,
    21, 24, 0, 34, 25, 3, 32, 11, 18, 14, 17, 6, 13, 26]
        scaling_factor_value = [
    0.9914, 1.0141, 1.0162, 0.9903, 0.9785, 0.8677,
    0.9833, 1.0264, 0.9763, 1.0894, 1.0135, 0.9885,
    0.9766, 0.9819, 0.9864, 0.9684, 0.9729, 0.9845,
    0.9865, 0.7048, 0.8014, 0.9936, 1.0259, 1.0123,
    0.9923, 0.9907, 0.9818, 1.0096, 1.0003, 0.9723,
    1.0060, 1.0155, 0.9896, 1.0103, 0.9907, 0.9972,
    1.0314, 1.0096, 1.0014, 1.0540, 1.0442]
    elif(args.Q ==4): # 16
        all_feature = [
    9, 39, 36, 40, 0, 10, 23, 22, 37, 30, 31, 18, 1, 2, 27, 7, 14, 34, 33, 35,
    28, 32, 3, 21, 38, 16, 4, 26, 17, 24, 13, 25]
        scaling_factor_value =[
    1.0597, 1.0212, 1.0194, 0.9855, 0.9741, 0.2219,
    0.9371, 1.0123, 0.8945, 1.1504, 1.0521, 0.9621,
    0.6155, 0.9702, 1.0084, 0.6321, 0.9805, 0.9730,
    1.0268, 0.0000, 0.0000, 0.9818, 1.0400, 1.0408,
    0.9728, 0.9676, 0.9740, 1.0135, 0.9952, 0.9251,
    1.0369, 1.0280, 0.9893, 1.0064, 1.0074, 1.0037,
    1.0784, 1.0370, 0.9807, 1.1310, 1.0748]
    elif(args.Q ==5): # 9
        all_feature = [9, 39, 40, 36, 7, 22, 2, 31, 1, 27, 10, 30, 23, 33, 37, 38, 28, 21,
    35, 34, 24, 14, 25, 18, 3, 32, 11, 6, 0, 17, 12, 26]
        scaling_factor_value = [
    0.9858, 1.0139, 1.0171, 0.9889, 0.9780, 0.8570,
    0.9859, 1.0310, 0.9752, 1.0936, 1.0112, 0.9865,
    0.9802, 0.9769, 0.9920, 0.9720, 0.9721, 0.9854,
    0.9894, 0.7048, 0.7654, 0.9975, 1.0225, 1.0108,
    0.9924, 0.9897, 0.9800, 1.0120, 1.0008, 0.9713,
    1.0110, 1.0162, 0.9885, 1.0092, 0.9943, 0.9969,
    1.0324, 1.0090, 1.0012, 1.0542, 1.0436]
    elif(args.Q ==6): # 13
        all_feature = [9, 39, 40, 36, 7, 22, 2, 31, 1, 33, 30, 23, 10, 37, 27, 38, 28, 35, 21, 14, 24, 0, 34, 25, 18, 11, 3, 32, 26, 17, 6, 13]
        scaling_factor_value = [
    0.9920, 1.0136, 1.0197, 0.9892, 0.9760, 0.8555,
    0.9843, 1.0268, 0.9757, 1.0738, 1.0103, 0.9905,
    0.9781, 0.9812, 0.9939, 0.9739, 0.9739, 0.9848,
    0.9906, 0.7048, 0.7712, 0.9948, 1.0250, 1.0105,
    0.9929, 0.9910, 0.9849, 1.0081, 1.0000, 0.9728,
    1.0109, 1.0157, 0.9885, 1.0111, 0.9916, 0.9974,
    1.0344, 1.0085, 1.0024, 1.0536, 1.0462]
else:
    all_feature = [9, 39, 36, 40, 23, 37, 0, 22, 31, 18, 30, 16, 1, 10, 2, 14, 34, 35, 32, 27, 33, 28, 3, 21, 26,
     24, 38, 13, 25, 17, 11, 29]
    scaling_factor_value = [1.0487, 1.0259, 1.0195, 0.9803, 0.7445, 0.0000, 0.8669, 0.8538, 0.7943, 1.1773, 1.0249, 0.9313, 0.00001, 0.9624, 1.0129, 0.2961, 1.0282, 0.9346, 1.0377, 0.00001, 0.00001, 0.9793, 1.0486, 1.0727, 0.9709, 0.9496, 0.9786, 1.0023, 0.9832, 0.9296, 1.0292, 1.0410, 1.0044, 0.9965, 1.0114, 1.0107, 1.1005, 1.0498, 0.9642, 1.1563, 1.0844]

selected_features = all_feature[:args.featureNums]
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
    print("f WIDTHLITMIT: {WIDTHLITMIT}  selected_features: ", selected_features,"len",len(selected_features))
    model2 = Resnet2(dim=len(selected_features), selected_features=selected_features, seed=SEED)
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
    end_time = time.time()
    total_training_time = end_time - start_time
    print("dnn2_loss_history", model2.loss_history)
    print("dnn2_macro_F1List", model2.macro_F1List)
    print("dnn2_micro_F1List", model2.micro_F1List)
    print('start testing...')
    # 获取验证集最佳模型
    saver.restore(sess, best_model_path)
    accuracy2 = model2.test()