# 为了得到最好的特征 先不用早停策略 后续对比用了早停策略的精度和所选特征
import sys
import time
import tensorflow as tf
import numpy as np
import csv, os
from Resnet import Resnet
import matplotlib.pyplot as plt

# 获取当前脚本的文件名
file_name = os.path.basename(__file__)
print(f"当前脚本的文件名是: {file_name}")

SEED = 25

K = 32 # topk 特征
WIDTHLITMIT = 1024 # 位宽限制加大
TRAIN_EPOCH = 30
ES_THRESHOLD = 3

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
sess = tf.compat.v1.Session(config=config)

curr_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
model = Resnet(K,ES_THRESHOLD,SEED)
model_dir = "./model"
new_folder = "model_" + curr_time
new_folder2 = new_folder + "best_model"
os.mkdir(os.path.join(model_dir, new_folder2))
os.mkdir(os.path.join(model_dir, new_folder))
start_time = time.time()
best_accuracy = 0.0  # 初始化最高accuracy
best_model_path = None  # 初始化最佳模型路径
model.train_start = 0
for _ in range(TRAIN_EPOCH):
    delta_loss,count, accuracy  = model.train()
    model.epoch_count += 1
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_path = os.path.join(model_dir, new_folder2, f"model_best_epoch_{model.epoch_count}.ckpt")
        saver = tf.compat.v1.train.Saver()
        saver.save(model.sess, best_model_path)
end_time = time.time()
total_training_time = end_time - start_time  # 计算训练总时长
scaling_factor_value = model.sess.run(model.scaling_factor)
print('scaling_factor_value last：',scaling_factor_value)

with model.sess as sess:
    saver.restore(sess, best_model_path)
    scaling_factor_value = model.sess.run(model.scaling_factor)
    model.test()
    print('scaling_factor_value best_model：',scaling_factor_value)
    print('start testing...')
    # 扁平化矩阵，并返回排序后的索引
    sorted_indices = np.argsort(scaling_factor_value.flatten())[::-1]
    '''choose top K feature '''
    K_values = [1, 2, 4, 8, 16, 24, 32]
    for K in K_values:
        top_k_indices = sorted_indices[:K]
        print(f"K = {K}, selected_features = {top_k_indices}")