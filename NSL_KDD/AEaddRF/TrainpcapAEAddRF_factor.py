# 使用早停策略
import sys
import time
import tensorflow as tf
import numpy as np
import csv, os
from pcapAEAddRF_factor import AE
import matplotlib.pyplot as plt

# 获取当前脚本的文件名
file_name = os.path.basename(__file__)
print(f"当前脚本的文件名是: {file_name}")

TRAIN_EPOCH = 30 # epoch num
TRAIN_EPOCH2 = 3
K = 32 # topk 特征
WIDTHLITMIT = 1024 # 位宽限制加大

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
curr_time = time.strftime("%Y%m%d%H%M", time.localtime())
model = AE()
start_time = time.time()
model.train_start = 0
for _ in range(TRAIN_EPOCH): # 先训练自编码器
    loss = model.train()
    model.epoch_count += 1
model.epoch_count = 0

model.train_classifier()

end_time = time.time()  # 记录训练结束时间
total_training_time = end_time - start_time  # 计算训练总时长
print("TSMRecord—100",model.TSMRecord)
print("loss_history—100",model.loss_history)
print(f"Total training time: {total_training_time:.2f} seconds")

model_dir = "model"
new_folder = "model_" + curr_time
os.mkdir(os.path.join(model_dir, new_folder))

# 拼接模型文件的完整路径
model_path = os.path.join(model_dir, new_folder, "model.ckpt")
saver = tf.compat.v1.train.Saver()

with model.sess as sess:
    saver.save(sess, model_path)
    reTrainAccuracy_history = {} # 存储对应的 删掉特征数量和精度
    scaling_factor_value = model.sess.run(model.scaling_factor)
    scaling_factor_value = scaling_factor_value.flatten()
    print('scaling_factor_value：',scaling_factor_value)
    print('start testing...')
    # 扁平化矩阵，并返回排序后的索引
    sorted_indices = np.argsort(scaling_factor_value.flatten())[::-1]
    K_values = [1, 2, 4, 8, 16, 24, 32]
    for K in K_values:
        top_k_indices = sorted_indices[:K]
        print(f"K = {K}, top_k_indices = {top_k_indices}")