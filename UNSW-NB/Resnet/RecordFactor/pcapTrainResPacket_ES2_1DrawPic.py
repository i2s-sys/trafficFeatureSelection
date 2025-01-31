# 使用早停策略
import sys
import time
import tensorflow as tf
import numpy as np
import csv, os

from pcapResnetPacketSeed import Resnet,Resnet2 # 导入DNN类
import matplotlib.pyplot as plt

# 获取当前脚本的文件名
file_name = os.path.basename(__file__)
print(f"当前脚本的文件名是: {file_name}")

SEED = 25
K = 1 # topk 特征
WIDTHLITMIT = 1024 # 位宽限制加大
TRAIN_EPOCH = 3
ES_THRESHOLD = 2

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

curr_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
model = Resnet(K,ES_THRESHOLD,SEED)
start_time = time.time()
model.train_start = 0
for _ in range(TRAIN_EPOCH):
    delta_loss, count = model.train()
    model.epoch_count += 1
    if model.earlyStop == True:
        print("model.earlyStop == True")
        break
end_time = time.time()
total_training_time = end_time - start_time  # 计算训练总时长
print("TSMRecord—100",model.TSMRecord)
print("loss_history—100",model.loss_history)
print(f"Total training time: {total_training_time:.2f} seconds")

model_dir = "./model"
new_folder = "model_" + curr_time
os.mkdir(os.path.join(model_dir, new_folder))

model_path = os.path.join(model_dir, new_folder, "model.ckpt")
saver = tf.compat.v1.train.Saver()

with model.sess as sess:
    saver.save(sess, model_path)
    reTrainAccuracy_history = {} # 存储对应的 删掉特征数量和精度
    scaling_factor_value = model.sess.run(model.scaling_factor)
    print('scaling_factor_value：',scaling_factor_value)
    print('start testing...')
    # 扁平化矩阵，并返回排序后的索引
    sorted_indices = np.argsort(scaling_factor_value.flatten())[::-1] #
    print("sorted_indices：",sorted_indices)
    model.test()
    print("loss_history", model.loss_history)
    print("macro_F1List", model.macro_F1List)
    print("micro_F1List", model.micro_F1List)
    print('starting retraining')  # 重新训练网络，只用 top_k_indices 对应的矩阵
    K2 = 5  # 设定要获取的前K个和后K个下标数量
    top_k_indices = sorted_indices[:K2]
    bottom_k_indices = sorted_indices[-K2:]

    # 提取每个epoch下这些下标对应的因子值
    top_k_factors = []
    bottom_k_factors = []

    for epoch_idx, scaling_factor in enumerate(model.scaling_factors_history):
        print(f"Epoch {epoch_idx + 1} 的选择因子: {scaling_factor}")

    for scaling_factors in model.scaling_factors_history:
        top_k_factors.append([scaling_factors[0, idx] for idx in top_k_indices])
        bottom_k_factors.append([scaling_factors[0, idx] for idx in bottom_k_indices])

    # 转换为numpy数组，方便绘图
    # 保存图片路径
    # 保存图片路径
    fig_dir = "figs"
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    top_k_fig_path = os.path.join(fig_dir, f"{K2}top_k_factors_{curr_time}.png")
    bottom_k_fig_path = os.path.join(fig_dir, f"{K2}bottom_k_factors_{curr_time}.png")
    tb_k_fig_path = os.path.join(fig_dir, f"{K2}tb_k_factors_{curr_time}.png")

    top_k_factors = np.array(top_k_factors)
    bottom_k_factors = np.array(bottom_k_factors)

    epochs = np.arange(1, len(model.scaling_factors_history) + 1)

    # 绘制前K2个因子值随epoch变化的点图
    plt.figure(figsize=(12, 6))
    for i in range(K2):
        plt.scatter(epochs, top_k_factors[:, i], label=f'Index {top_k_indices[i]}')
        plt.plot(epochs, top_k_factors[:, i])
    plt.xlabel('Epoch')
    plt.ylabel('Scaling Factor Value')
    # plt.title('Top K Scaling Factor Values Over Epochs')
    plt.legend()
    plt.savefig(top_k_fig_path)
    plt.show()

    K2 = 6  # 设定要获取的前K个和后K个下标数量
    top_k_indices = sorted_indices[:K2]
    bottom_k_indices = sorted_indices[-K2:]

    # 提取每个epoch下这些下标对应的因子值
    top_k_factors = []
    bottom_k_factors = []

    for epoch_idx, scaling_factor in enumerate(model.scaling_factors_history):
        print(f"Epoch {epoch_idx + 1} 的选择因子: {scaling_factor}")

    for scaling_factors in model.scaling_factors_history:
        top_k_factors.append([scaling_factors[0, idx] for idx in top_k_indices])
        bottom_k_factors.append([scaling_factors[0, idx] for idx in bottom_k_indices])

    # 转换为numpy数组，方便绘图
    # 保存图片路径
    # 保存图片路径
    fig_dir = "figs"
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    top_k_fig_path = os.path.join(fig_dir, f"{K2}top_k_factors_{curr_time}.png")
    bottom_k_fig_path = os.path.join(fig_dir, f"{K2}bottom_k_factors_{curr_time}.png")
    tb_k_fig_path = os.path.join(fig_dir, f"{K2}tb_k_factors_{curr_time}.png")

    top_k_factors = np.array(top_k_factors)
    bottom_k_factors = np.array(bottom_k_factors)

    epochs = np.arange(1, len(model.scaling_factors_history) + 1)

    # 绘制前K2个因子值随epoch变化的点图
    plt.figure(figsize=(12, 6))
    for i in range(K2):
        plt.scatter(epochs, top_k_factors[:, i], label=f'Index {top_k_indices[i]}')
        plt.plot(epochs, top_k_factors[:, i])
    plt.xlabel('Epoch')
    plt.ylabel('Scaling Factor Value')
    # plt.title('Top K Scaling Factor Values Over Epochs')
    plt.legend()
    plt.savefig(top_k_fig_path)
    plt.show()

    # 绘制后K2个因子值随epoch变化的点图
    plt.figure(figsize=(12, 6))
    for i in range(K2):
        plt.scatter(epochs, bottom_k_factors[:, i], label=f'Index {bottom_k_indices[i]}')
        plt.plot(epochs, bottom_k_factors[:, i])
    plt.xlabel('Epoch')
    plt.ylabel('Scaling Factor Value')
    plt.legend()
    plt.savefig(bottom_k_fig_path)
    plt.show()

    plt.figure(figsize=(12, 6))
    for i in range(K2):
        plt.scatter(epochs, top_k_factors[:, i], label=f'Index {top_k_indices[i]}')
        plt.scatter(epochs, bottom_k_factors[:, i], label=f'Index {bottom_k_indices[i]}')
        plt.plot(epochs, top_k_factors[:, i])
        plt.plot(epochs, bottom_k_factors[:, i])
    plt.xlabel('Epoch')
    plt.ylabel('Scaling Factor Value')
    # plt.title('tb K Scaling Factor Values Over Epochs')
    plt.legend()
    plt.savefig(tb_k_fig_path)
    plt.show()

    '''choose top K feature '''
    # k = K
    # 提取前 k 个特征的下标和因子值
    # top_k_indices = sorted_indices[:k]
    # print("K=",k,"top_k_indices",top_k_indices)
    # selected_features = top_k_indices
    # dnn2 = Resnet2(dim=len(selected_features), selected_features=selected_features,seed=SEED)
    # print('start retraining...')
    # start_time = time.time()
    # for _ in range(TRAIN_EPOCH):
    #     delta_loss, count = dnn2.train()
    #     dnn2.epoch_count += 1
    # end_time = time.time()
    # total_training_time = end_time - start_time
    # print("dnn2_loss_history", dnn2.loss_history)
    # print("dnn2_macro_F1List", dnn2.macro_F1List)
    # print("dnn2_micro_F1List", dnn2.micro_F1List)
    # print('start testing...')
    # accuracy2 = dnn2.test()