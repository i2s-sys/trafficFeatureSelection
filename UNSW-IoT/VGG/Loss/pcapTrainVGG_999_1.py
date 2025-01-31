import sys
import time
import tensorflow as tf
import numpy as np
import csv, os, random
import argparse
from pcapVGGSeed_Factor import VGG # 导入DNN类
import matplotlib.pyplot as plt
import fwr13y.seeder.tf as seeder

# 获取当前脚本的文件名
file_name = os.path.basename(__file__)
print(f"当前脚本的文件名是: {file_name}")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
sess = tf.compat.v1.Session(config=config)

# 命令行参数解析
parser = argparse.ArgumentParser(description='Train a DNN model with specified parameters.')
parser.add_argument('--BETA', type=float, default=0.999, help='Class balance loss parameter BETA')
parser.add_argument('--GAMMA', type=float, default=1, help='Class balance loss parameter GAMMA')
parser.add_argument('--seed', type=int, default=25, help='Random seed')
parser.add_argument('--EPOCH_NUM', type=int, default=30, help='Number of epochs')
args = parser.parse_args()

curr_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
model_dir = "./model"
new_folder = "model_" + curr_time
best_accuracy = 0.0  # 初始化最高accuracy
best_model_path = None  # 初始化最佳模型路径
new_folder2 = new_folder + "best_model"
os.mkdir(os.path.join(model_dir, new_folder2))

seeder.init(master_seed=args.seed,
            ngpus=1,
            local_rank=0)
args.seed = seeder.get_master_seed()
seeder.reseed(0, 0)

BETA = args.BETA  # 类平衡损失的β
GAMMA = args.GAMMA
SEED = args.seed
EPOCH_NUM = args.EPOCH_NUM

model = VGG("cb_focal_loss",SEED, BETA, GAMMA)
saver = tf.compat.v1.train.Saver()
for epoch in range(EPOCH_NUM):
    seeder.reseed(1, epoch)
    accuracy = model.train()
    model.epoch_count += 1
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_path = os.path.join(model_dir, new_folder2, f"model_best_epoch_{model.epoch_count}.ckpt")
        saver.save(model.sess, best_model_path)

new_folder = "model_" + curr_time
os.mkdir(os.path.join(model_dir, new_folder))
model_path = os.path.join(model_dir, new_folder, "model.ckpt")
print("loss_history", model.loss_history)
print("macro_F1List", model.macro_F1List)
print("micro_F1List", model.micro_F1List)

saver = tf.compat.v1.train.Saver()
with model.sess as sess:
    saver.restore(sess, best_model_path)
    print('best Model testing...')
    model.test()
