import sys
import time
import tensorflow as tf
import numpy as np
import csv, os
from pcapResnetPacketSeed import Resnet  # 导入DNN类
import matplotlib.pyplot as plt

# 获取当前脚本的文件名
file_name = os.path.basename(__file__)
print(f"当前脚本的文件名是: {file_name}")

SEED = 25
K = 16  # topk 特征
WIDTHLITMIT = 1024  # 位宽限制加大
TRAIN_EPOCH = 30
ES_THRESHOLD = 3

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
sess = tf.compat.v1.Session(config=config)

curr_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
model = Resnet(K,ES_THRESHOLD,SEED)
start_time = time.time()
model.train_start = 0
for _ in range(TRAIN_EPOCH):
    delta_loss, count = model.train()
    model.epoch_count += 1
    # if model.earlyStop == True:
    #     print("model.earlyStop == True")
    #     break
print(f"K = {K}: TSMRecord—100", model.TSMRecord)
print("loss_history—100", model.loss_history)

model_dir = "model"
new_folder = "model_" + curr_time
os.mkdir(os.path.join(model_dir, new_folder))

model_path = os.path.join(model_dir, new_folder, "model.ckpt")
saver = tf.compat.v1.train.Saver()

with model.sess as sess:
    saver.save(sess, model_path)
    reTrainAccuracy_history = {}  # 存储对应的删掉特征数量和精度
    scaling_factor_value = model.sess.run(model.scaling_factor)
    print('scaling_factor_value：', scaling_factor_value)
    print('start testing...')
    # 扁平化矩阵，并返回排序后的索引
    sorted_indices = np.argsort(scaling_factor_value.flatten())[::-1]
    print("sorted_indices：", sorted_indices)
    model.test()
    print("loss_history", model.loss_history)
    print("macro_F1List", model.macro_F1List)
    print("micro_F1List", model.micro_F1List)
    print('starting retraining')  # 重新训练网络，只用 top_k_indices 对应的矩阵
    K2 = 5  # 设定要获取的前K个和后K个下标数量
    top_k_indices = sorted_indices[:K2]
    bottom_k_indices = sorted_indices[-K2:]

    # 初始化第0个epoch的因子值为1
    top_k_factors = [np.ones(K2)]
    bottom_k_factors = [np.ones(K2)]
    scaling_factors_history_rounded = [np.ones((1, len(scaling_factor_value.flatten())))]

    for epoch_idx, scaling_factor in enumerate(model.scaling_factors_history):
        rounded_scaling_factor = np.round(scaling_factor.flatten(), 2)  # 将scaling_factor扁平化并四舍五入
        scaling_factors_history_rounded.append(rounded_scaling_factor)
        print(f"Epoch {epoch_idx + 1} = {rounded_scaling_factor.tolist()}")  # 以列表形式输出

    for scaling_factors in scaling_factors_history_rounded[1:]:
        top_k_factors.append([scaling_factors[idx] for idx in top_k_indices])
        bottom_k_factors.append([scaling_factors[idx] for idx in bottom_k_indices])

    # 定义图片存放路径
    fig_dir = "figs"
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    tb_k_fig_path1 = os.path.join(fig_dir, f"0-2_{K2}tb_k_factors_{curr_time}.png")
    tb_k_fig_path2 = os.path.join(fig_dir, f"0-1.5_{K2}tb_k_factors_{curr_time}.png")

    top_k_factors = np.array(top_k_factors)  # 转化成数组
    bottom_k_factors = np.array(bottom_k_factors)
    # 设置横坐标和纵坐标范围以及刻度
    x_ticks = np.arange(0, 31, 5)
    y_ticks = np.arange(0, 2, 0.5)
    epochs = np.arange(0, len(model.scaling_factors_history) + 1)

    # 设置字体
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["font.size"] = 14  # 将字体大小设为14

    plt.figure(figsize=(6, 6))
    for i in range(K2):
        plt.scatter(epochs, top_k_factors[:, i], label=f'feature {top_k_indices[i]}')
        plt.plot(epochs, top_k_factors[:, i])
    for i in range(K2):
        plt.scatter(epochs, bottom_k_factors[:, i], label=f'feature {bottom_k_indices[i]}', color='brown')
        plt.plot(epochs, bottom_k_factors[:, i], color='brown')
    plt.xlabel('Epoch', fontweight='bold')
    plt.ylabel('Scaling Factor Value', fontweight='bold')
    plt.ylim(0, 2)
    plt.legend(ncol=2, bbox_to_anchor=(1, 1))  # 将图例分成两列并放置在图外
    plt.savefig(tb_k_fig_path2)
    plt.show()
