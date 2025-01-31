# 纯resnet 就检查全部特征的时候 resnet的准确率
import sys
import time
import tensorflow as tf
import numpy as np
import csv, os, random
from pcapVGGSeed_Factor import VGG # 导入DNN类
import matplotlib.pyplot as plt

# 获取当前脚本的文件名
file_name = os.path.basename(__file__)
print(f"当前脚本的文件名是: {file_name}")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
sess = tf.compat.v1.Session(config=config)

BETA = 0.999 # 类平衡损失的β
GAMMA = 2
SEED = 25
EPOCH_NUM = 10
def set_seed(seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

curr_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
model_dir = "./model"
new_folder = "model_" + curr_time
best_accuracy = 0.0  # 初始化最高accuracy
best_model_path = None  # 初始化最佳模型路径
new_folder2 = new_folder + "best_model"
os.mkdir(os.path.join(model_dir, new_folder2))


if __name__ == '__main__':
    set_seed(SEED)
    model = VGG("cb_focal_loss", SEED, BETA, GAMMA)
    saver = tf.compat.v1.train.Saver()
    for _ in range(EPOCH_NUM):
        accuracy = model.train()
        model.epoch_count += 1
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_path = os.path.join(model_dir, new_folder2, f"model_best_epoch_{model.epoch_count}.ckpt")
            saver.save(model.sess, best_model_path)
    model_dir = "./model"
    new_folder = "model_" + curr_time
    os.mkdir(os.path.join(model_dir, new_folder))
    model_path = os.path.join(model_dir, new_folder, "model.ckpt")
    print("loss_history",model.loss_history)
    print("macro_F1List",model.macro_F1List)
    print("micro_F1List",model.micro_F1List)

    saver = tf.compat.v1.train.Saver()
    with model.sess as sess:
        saver.restore(sess, best_model_path)
        print('best Model testing...')
        model.test()
