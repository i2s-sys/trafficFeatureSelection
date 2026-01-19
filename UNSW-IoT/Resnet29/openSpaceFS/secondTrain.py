# TensorFlow 2.9.0 compatible training script with early stopping for UNSW-IoT
import sys
import time
import tensorflow as tf
import numpy as np
import csv, os
from pcapResnetPacketSeed import Resnet2
import matplotlib.pyplot as plt

# 设置环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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
DATA_DIM = 72  # 全量特征
K = DATA_DIM  # 使用全部特征，缩放因子负责选择
WIDTHLITMIT = 1024 # 位宽限制加大
TRAIN_EPOCH = 30
MODEL_DIR = "checkpoints"

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
k = K

# 全量特征，缩放因子固定：加载 moni_scaling_factor.npy，取最大32，其余置零
scaling_path = os.path.join(os.path.dirname(__file__), "moni_scaling_factor.npy")
fixed_scaling = None
if os.path.exists(scaling_path):
    full = np.load(scaling_path)
    if full.ndim == 2 and full.shape[1] >= DATA_DIM:
        masked = np.zeros_like(full, dtype=np.float32)
        top_idx = np.argsort(full.flatten())[::-1][:32]
        masked.reshape(-1)[top_idx] = full.reshape(-1)[top_idx]
        fixed_scaling = masked.astype(np.float32)
        print(f"Loaded scaling_factor from {scaling_path}, shape {fixed_scaling.shape}")
        print(f"Top-32 indices: {top_idx}")
        print(f"Scaling factor (head): {fixed_scaling[:, :8]}")
    else:
        print(f"[WARN] scaling_factor shape mismatch: expect (1,{DATA_DIM}) got {full.shape}, skip freeze.")
else:
    print(f"[WARN] scaling_factor file not found: {scaling_path}, will run without scaling freeze.")

selected_features = list(range(K))
model2 = Resnet2(dim=len(selected_features), selected_features=selected_features, seed=SEED, fixed_scaling=fixed_scaling)
print('start retraining...')

start_time = time.time()
for _ in range(TRAIN_EPOCH):
    delta_loss, count = model2.train()
    model2.epoch_count += 1
# print('start testing...')
# accuracy2 = model2.test()

# 训练完成后保存模型权重
os.makedirs(MODEL_DIR, exist_ok=True)
ckpt_path = os.path.join(MODEL_DIR, f"ow_resnet2_k{k}_{curr_time}.weights.h5")
model2.model.save_weights(ckpt_path)
print(f"模型权重已保存到: {ckpt_path}")
