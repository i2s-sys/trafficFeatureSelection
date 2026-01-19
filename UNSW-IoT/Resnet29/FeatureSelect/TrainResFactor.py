# TensorFlow 2.9.0 compatible training script with early stopping for UNSW-IoT (no stop version)
import sys
import time
import tensorflow as tf
import numpy as np
import csv, os
from pcapResnetPacketSeed import Resnet, Resnet2
import matplotlib.pyplot as plt

# 设置环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
DATA_DIM = 72 # 特征数
K = 32 # topk 特征
WIDTHLITMIT = 1024 # 位宽限制加大
TRAIN_EPOCH = 30
ES_THRESHOLD = 2

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

model = Resnet(K, ES_THRESHOLD, SEED)
start_time = time.time()

# 训练模型
for _ in range(TRAIN_EPOCH):
    delta_loss, count = model.train()
    model.epoch_count += 1
    # if model.earlyStop == True:
    #     print("model.earlyStop == True")
    #     break

end_time = time.time()
total_training_time = end_time - start_time
print("TSMRecord—100", model.TSMRecord)
print("loss_history—100", model.loss_history)
print(f"Total training time: {total_training_time:.2f} seconds")

# 保存模型
model_dir = "model"
new_folder = "model_" + curr_time
os.makedirs(os.path.join(model_dir, new_folder), exist_ok=True)

model_path = os.path.join(model_dir, new_folder, "model")
model.model.save_weights(model_path)

# 获取scaling factor值
scaling_factor_value = model.model.scaling_factor.numpy()
print('scaling_factor_value：', scaling_factor_value)
print('start testing...')

sorted_indices = np.argsort(scaling_factor_value.flatten())[::-1]
print("sorted_indices：", sorted_indices)
macroF1, microF1 = model.test2()
print("loss_history", model.loss_history)
print("macro_F1List", model.macro_F1List)
print("micro_F1List", model.micro_F1List)

print('starting retraining')
'''choose top K feature '''
k = K
top_k_indices = sorted_indices[:k]
print("K=", k, "top_k_indices", top_k_indices)

