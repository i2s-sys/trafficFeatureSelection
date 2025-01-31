# 使用早停策略
import sys
import time
import tensorflow as tf
import numpy as np
import csv, os
from LSTMPacket_1F import LSTM,LSTM2 # 导入DNN类
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# 获取当前脚本的文件名
file_name = os.path.basename(__file__)
print(f"当前脚本的文件名是: {file_name}")

IS_EARLY_STOP = False
EPOCH_NUM = 30
DATA_DIM = 72 # 特征数
SEED = 25
K = 8 # topk 特征
ES_THRESHOLD = 3
WIDTHLITMIT = 1024 # 位宽限制加大

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

curr_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
model = LSTM("cb_focal_loss",SEED,K,ES_THRESHOLD)
start_time = time.time()  # 记录训练开始时间
# 训练模型
model.train_start = 0
for _ in range(EPOCH_NUM):
    delta_loss, count = model.train()
    model.epoch_count += 1
    if model.earlyStop == True:
        print("model.earlyStop == True 看top K factor")
        scaling_factor_value = model.sess.run(model.scaling_factor)
        sorted_indices = np.argsort(scaling_factor_value.flatten())[::-1]
        top_k_indices = sorted_indices[:K]
        print("K=", K, "top_k_indices", top_k_indices)
        if IS_EARLY_STOP == True:
            break
end_time = time.time()  # 记录训练结束时间
total_training_time = end_time - start_time  # 计算训练总时长
print("TSMRecord—100",model.TSMRecord)
print("loss_history—100",model.loss_history)
print(f"Total training time: {total_training_time:.2f} seconds")

model_dir = "./model"
new_folder = "model_" + curr_time
os.mkdir(os.path.join(model_dir, new_folder))

# 拼接模型文件的完整路径
model_path = os.path.join(model_dir, new_folder, "model.ckpt")
saver = tf.compat.v1.train.Saver()

def knapsack(values, weights, max_weight):
    n = len(values)
    dp = np.zeros((n + 1, max_weight + 1))
    keep = np.zeros((n + 1, max_weight + 1), dtype=int)

    for i in range(1, n + 1):
        for w in range(max_weight + 1):
            if weights[i - 1] <= w:
                if dp[i - 1][w] < dp[i - 1][w - weights[i - 1]] + values[i - 1]:
                    dp[i][w] = dp[i - 1][w - weights[i - 1]] + values[i - 1]
                    keep[i][w] = 1
                else:
                    dp[i][w] = dp[i - 1][w]
            else:
                dp[i][w] = dp[i - 1][w]

    selected_items = []
    total_weight = 0
    total_value = 0
    w = max_weight
    for i in range(n, 0, -1):
        if keep[i][w] == 1:
            selected_items.append(i - 1)
            total_weight += weights[i - 1]
            total_value += values[i - 1]
            w -= weights[i - 1]
    print("total_weight",total_weight,"total_value",total_value,"selected_items",selected_items)
    return selected_items

with model.sess as sess:
    saver.save(sess, model_path)
    reTrainAccuracy_history = {} # 存储对应的 删掉特征数量和精度
    scaling_factor_value = model.sess.run(model.scaling_factor)
    print('scaling_factor_value：',scaling_factor_value)
    print('start testing...')
    # 扁平化矩阵，并返回排序后的索引
    sorted_indices = np.argsort(scaling_factor_value.flatten())[::-1] #
    print("sorted_indices：", sorted_indices)
    macroF1, microF1 = model.test2()
    print("loss_history", model.loss_history)
    print("macro_F1List", model.macro_F1List)
    print("micro_F1List", model.micro_F1List)
    print('starting retraining')  # 重新训练网络，只用 top_k_indices 对应的矩阵
    k = K
    # 提取前 k+n 个特征的下标和因子值
    top_k_indices = sorted_indices[:k]
    print("K=", k, "top_k_indices", top_k_indices)
    selected_features = top_k_indices
    dnn2 = LSTM2("cb_focal_loss",SEED,dim=len(selected_features), selected_features=selected_features)
    print('start retraining...')
    start_time = time.time()
    for _ in range(EPOCH_NUM):  # 调用train函数，并获取损失的变化
        delta_loss, count = dnn2.train()  # 判断损失的变化是否小于阈值
        dnn2.epoch_count += 1
    end_time = time.time()  # 记录训练结束时间
    total_training_time = end_time - start_time  # 计算训练总时长
    print("dnn2_loss_history", dnn2.loss_history)
    print("dnn2_macro_F1List", dnn2.macro_F1List)
    print("dnn2_micro_F1List", dnn2.micro_F1List)
    print('start testing...')
    accuracy2 = dnn2.test()
