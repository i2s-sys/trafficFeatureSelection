# 为了得到最好的特征 先不用早停策略 后续对比用了早停策略的精度和所选特征
import sys
import time
import tensorflow as tf
import numpy as np
import csv, os
from pcapResnetSeed_factor import Resnet
from tensorflow.keras.backend import clear_session
import matplotlib.pyplot as plt

# 获取当前脚本的文件名
file_name = os.path.basename(__file__)
print(f"当前脚本的文件名是: {file_name}")

SEED = 25

K = 32  # topk 特征
WIDTHLITMIT = 1024  # 位宽限制加大
TRAIN_EPOCH = 30
ES_THRESHOLD = 3

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
sess = tf.compat.v1.Session(config=config)

feature_widths = [
    64, 64, 64, 64,  # fiat_mean, fiat_min, fiat_max, fiat_std
    64, 64, 64, 64,  # biat_mean, biat_min, biat_max, biat_std
    64, 64, 64, 64,  # diat_mean, diat_min, diat_max, diat_std
    64,  # duration 13
    64, 64, 64, 64, 64,  # fwin_total, fwin_mean, fwin_min, fwin_max, fwin_std
    64, 64, 64, 64, 64,  # bwin_total, bwin_mean, bwin_min, bwin_max, bwin_std
    64, 64, 64, 64, 64,  # dwin_total, dwin_mean, dwin_min, dwin_max, dwin_std
    16, 16, 16,  # fpnum, bpnum, dpnum
    64, 64, 64, 64,  # bfpnum_rate, fpnum_s, bpnum_s, dpnum_s 22
    64, 64, 64, 64, 64,  # fpl_total, fpl_mean, fpl_min, fpl_max, fpl_std
    64, 64, 64, 64, 64,  # bpl_total, bpl_mean, bpl_min, bpl_max, bpl_std
    64, 64, 64, 64, 64,  # dpl_total, dpl_mean, dpl_min, dpl_max, dpl_std
    64, 64, 64, 64,  # bfpl_rate, fpl_s, bpl_s, dpl_s  19
    16, 16, 16, 16, 16, 16, 16, 16,  # fin_cnt, syn_cnt, rst_cnt, pst_cnt, ack_cnt, urg_cnt, cwe_cnt, ece_cnt
    16, 16, 16, 16,  # fwd_pst_cnt, fwd_urg_cnt, bwd_pst_cnt, bwd_urg_cnt
    16, 16, 16,  # fp_hdr_len, bp_hdr_len, dp_hdr_len
    64, 64, 64  # f_ht_len, b_ht_len, d_ht_len 18
]

curr_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
model = Resnet(K, ES_THRESHOLD, SEED)
start_time = time.time()

model_dir = "./model"
new_folder = "model_" + curr_time
os.mkdir(os.path.join(model_dir, new_folder))
best_accuracy = 0.0  # 初始化最高accuracy
best_model_path = None  # 初始化最佳模型路径
new_folder2 = new_folder + "best_model"
os.mkdir(os.path.join(model_dir, new_folder2))
saver = tf.compat.v1.train.Saver()

model.train_start = 0
for _ in range(TRAIN_EPOCH):
    accuracy = model.train()
    model.epoch_count += 1
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_path = os.path.join(model_dir, new_folder2, f"model_best_epoch_{model.epoch_count}.ckpt")
        saver.save(model.sess, best_model_path)
    if model.earlyStop == True:
        print("model.earlyStop == True")
        break

end_time = time.time()
total_training_time = end_time - start_time  # 计算训练总时长
model.test()
print("TSMRecord—100", model.TSMRecord)
print("loss_history—100", model.loss_history)
print(f"Total training time: {total_training_time:.2f} seconds")

model_path = os.path.join(model_dir, new_folder, "model.ckpt")
scaling_factor_value = model.sess.run(model.scaling_factor)
print('scaling_factor_value：', scaling_factor_value)

with model.sess as sess:
    saver.restore(sess, best_model_path)
    print('best Model testing...')
    model.test()
    # 扁平化矩阵，并返回排序后的索引
    sorted_indices = np.argsort(scaling_factor_value.flatten())[::-1]
    '''choose top K feature '''
    K_values = [1, 2, 4, 8, 16, 24, 32]
    for K in K_values:
        top_k_indices = sorted_indices[:K]
        print(f"K = {K}, selected_features = {top_k_indices}")