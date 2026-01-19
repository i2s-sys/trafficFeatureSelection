# TensorFlow 2.9.0 compatible second training script for UNSW-IoT VGG
import sys
import time
import tensorflow as tf
import numpy as np
import csv, os
from pcapVGG2Seed import VGG2
import matplotlib.pyplot as plt

# 设置环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 减少TensorFlow日志输出

# 配置GPU内存增长
def configure_gpu():
    """配置GPU设置"""
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU配置成功: {len(gpus)} 个GPU")
            return True
        else:
            print("未检测到GPU，将使用CPU")
            return False
    except Exception as e:
        print(f"GPU配置失败: {e}")
        return False

# 执行GPU配置
gpu_available = configure_gpu()

# 获取当前脚本的文件名
file_name = os.path.basename(__file__)
print(f"当前脚本的文件名是: {file_name}")

SEED = 25
WIDTHLITMIT = 1024 # 位宽限制加大
TRAIN_EPOCH = 30

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
    64, 32, 32, 32, 32,  # dpl_total, dpl_mean, dpl_min, dpl_max, dwin_std
    32, 32, 32, 32,         # bfpl_rate, fpl_s, bpl_s, dpl_s  19
    16, 16, 16, 16, 16, 16, 16, 16,  # fin_cnt, syn_cnt, rst_cnt, pst_cnt, ack_cnt, urg_cnt, cwe_cnt, ece_cnt
    16, 16, 16, 16,     # fwd_pst_cnt, fwd_urg_cnt, bwd_pst_cnt, bwd_urg_cnt
    16, 16, 16,         # fp_hdr_len, bp_hdr_len, dp_hdr_len
    32, 32, 32          # f_ht_len, b_ht_len, d_ht_len 18
]

curr_time = time.strftime("%Y%m%d%H%M%S", time.localtime())

K = 16 # topk 特征
# 使用预定义的特征选择结果
# sorted_indices = [50,23,18,13,45,40,35,26,16,21,31,12,19,14,20,24,15,27,17,22,25,10,9,68,66,67,38,43,41,36,42,47] # infs
# sorted_indices = [71, 69, 68, 65, 61, 59, 58, 57, 55, 53, 50, 48, 43, 37, 34, 30, 25, 24, 23, 21, 20, 19, 17, 16, 15, 14, 13, 9, 8, 5, 4, 0] # pso
# sorted_indices = [14, 70, 3, 12, 62, 55, 23, 25, 61, 20, 51, 56, 24, 18, 15, 21, 48, 13, 17, 9, 59, 26, 32, 68, 5, 67, 66, 71, 8, 7, 69, 65] # sca
sorted_indices = [60, 34, 13, 62, 11, 10, 24, 70, 61, 12, 30, 27, 28, 14, 15, 68, 26, 2, 52, 65, 22, 7, 18, 45, 67, 53, 4, 35, 20, 55, 21, 19] # fpa
# sorted_indices = [20,15,25,19,18,21,51,53,33,26,17,31,69,62,70,35,36,49,44,41,38,13,16,68,24,64,29,50,37,57,9,22] # factor

top_k_indices = sorted_indices[:K]
print("K=",K,"top_k_indices",top_k_indices)
selected_features = top_k_indices

vgg2 = VGG2("ce",dim=len(selected_features), selected_features=selected_features,seed=SEED)
print('start retraining...')

start_time = time.time()
for _ in range(TRAIN_EPOCH):
    vgg2.train()
    vgg2.epoch_count += 1
end_time = time.time()
total_training_time = end_time - start_time
print("vgg2_loss_history", vgg2.loss_history)
print("vgg2_macro_F1List", vgg2.macro_F1List)
print("vgg2_micro_F1List", vgg2.micro_F1List)
print('start testing...')
accuracy2 = vgg2.test()
