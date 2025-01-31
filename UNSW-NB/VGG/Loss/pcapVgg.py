# 纯resnet 检查
import time

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from sklearn.metrics import f1_score
import numpy as np
import csv, random

# 禁用 Eager Execution
tf.compat.v1.disable_eager_execution()
DATA_DIM = 115 # 特征
OUTPUT_DIM = 29  # 0-28类
BETA = 0.9999 # 类平衡损失的β
GAMMA = 1
THRESHOLD = 32

# Hyper Parameters
LEARNING_RATE = 0.0001
BATCH_SIZE = 128

MODEL_SAVE_PATH = './model/'
MODEL_SAVE_PATH2 = './model2/'
MODEL_NAME = 'model'
MODEL_NAME2 = 'model2'
KEEP_PROB = 0.5
top_k_values=[]
top_k_indice=[]
NUM_ATTENTION_CHANNELS=1

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

def weights_variable(shape):
    w = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(w)
def bias_variable(shape):
    b = tf.constant(0.1, shape=shape)
    return tf.Variable(b)
def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

class VGG():
    def __init__(self):  # 不输入维度 默认是DATA_DIM
        print(f"BETA = {BETA}, GAMMA = {GAMMA}")
        self.learning_rate = LEARNING_RATE
        self.batch_size = BATCH_SIZE
        self.epoch_count = 0
        self.x_input = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, DATA_DIM])
        self.target = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None])
        self.train_step = tf.Variable(0, trainable=False)
        self.loss_history = []
        self.micro_F1List = []
        self.macro_F1List = []
        self.init_data()
        self.total_iterations_per_epoch = self.train_length // BATCH_SIZE

        self.create_VGG()
        beta = BETA    # cb数据预处理
        ClassNum = len(self.label_status)
        effective_num = {}
        for key, value in self.label_status.items():
            new_value = (1.0 - beta) / (1.0 - np.power(beta, value))
            effective_num[key] = new_value
        # 计算好有效数 之后 使用的是有效数的权重
        total_effective_num = sum(effective_num.values())
        self.weights = {}
        for key, value in effective_num.items():
            new_value = effective_num[key] / total_effective_num * ClassNum
            self.weights[key] = new_value
        self.sess = tf.compat.v1.Session()
        self.build_loss()  # 构建损失

        self.ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        self.saver = tf.compat.v1.train.Saver()
        self.train_start = 0
        if self.ckpt and self.ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, self.ckpt.model_checkpoint_path)
            self.train_start = self.sess.run(self.train_step)

    def create_VGG(self):
        self.scaling_factor = tf.Variable(
            tf.constant(1, dtype=tf.float32, shape=[1, DATA_DIM]))
        self.scaling_factor_extended = tf.tile(self.scaling_factor, [BATCH_SIZE, 1])
        scaled_input = tf.multiply(self.x_input, self.scaling_factor_extended)
        scaled_input = tf.reshape(scaled_input, [BATCH_SIZE, DATA_DIM, 1, 1])
        scaled_input = tf.keras.layers.Input(tensor=scaled_input)  # 使用Input
        # 构建模型层
        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_1')(scaled_input)
        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_2')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool_1')(x)
        x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv_3')(x)
        x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv_4')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool_2')(x)
        x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv_5')(x)
        x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv_6')(x)
        x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv_7')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool_3')(x)
        x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv_8')(x)
        # 全连接层
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(1024, activation='relu', name='fc_1')(x)
        keep_prob = tf.keras.layers.Dropout(0.5)(x)  # 使用 Dropout 层

        x = tf.keras.layers.Dense(1024, activation='relu', name='fc_2')(keep_prob)
        x = tf.keras.layers.Dropout(0.5)(x)

        self.y = tf.keras.layers.Dense(OUTPUT_DIM, name='fc_3')(x)

    def build_loss(self):
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.y, labels=self.target)
        self.loss = tf.reduce_sum(ce)
        self.train_op = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(self.loss,global_step=self.train_step)
        self.sess.run(tf.compat.v1.global_variables_initializer())

    def train(self):  # 每次训练都是一个epoch遍历完整个训练集
        total_loss = 0
        num_batches = 0
        self.train_start = 0
        epoch_start_time = time.time()  # 记录epoch开始时间
        for _ in range(self.total_iterations_per_epoch):
            step = self.train_start
            batch = self.get_a_train_batch(step)
            data, label = self.get_data_label(batch)
            _, y = self.sess.run([self.train_op, self.y], feed_dict={self.x_input: data, self.target: label})
            curr_loss, y = self.sess.run([self.loss, self.y], feed_dict={self.x_input: data, self.target: label})
            total_loss += curr_loss
            num_batches += 1
            self.train_start += 1
        average_loss = total_loss / num_batches
        self.loss_history.append(average_loss)
        epoch_end_time = time.time()  # 记录epoch结束时间
        epoch_duration = epoch_end_time - epoch_start_time  # 计算epoch时长
        print(f'Epoch {self.epoch_count + 1} completed, average loss: {average_loss}, duration: {epoch_duration:.2f} seconds')
        micro_F1, macro_F1 = self.test2()
        self.micro_F1List.append(micro_F1)
        self.macro_F1List.append(macro_F1)
        if self.epoch_count == 0: # 第一个epoch
            # 初始化损失的变化
            delta_loss = 0
            self.count = 0
            self.curr_loss = average_loss
        else:
            # 计算损失的变化pca
            self.prev_loss = self.curr_loss # 上一个epoch的损失
            self.curr_loss = average_loss # 新epoch的损失
            delta_loss = abs(self.curr_loss - self.prev_loss)
            # 判断损失的变化是否小于阈值
            if delta_loss <= 0.03:
                # 次数加一
                self.count += 1
            else:
                # 次数归零
                self.count = 0
        # 返回损失的变化
        return delta_loss, self.count

    def get_a_train_batch(self, step):
        '''从训练数据集中获取一个批次的数据 '''
        min_index = step * self.batch_size
        max_index = min_index + self.batch_size
        if max_index > self.train_length:
            max_index = self.train_length
        return self.train_data[min_index:max_index]  # 返回min到max之间的数据样本

    def get_data_label(self, batch):
        '''划分数据特征 和 最后一列标签'''
        data = np.delete(batch, -1, axis=1) # 从输入的批次数据中删除最后一列，以提取仅包含特征的数据
        label = np.array(batch, dtype=np.int32)[:, -1]
        return data, label

    def init_data(self):
        self.train_data = []
        self.test_data = []  # init train and test data
        self.label_status = {}
        filename = '../train_data.csv'
        csv_reader = csv.reader(open(filename))
        label0_data = label1_data = label2_data = label3_data = label4_data = label5_data = []
        label6_data = label7_data = label8_data = label9_data = label10_data = label11_data = []
        label12_data = label13_data = label14_data = label15_data = label16_data = label17_data = []
        label18_data = label19_data = label20_data = label21_data = label22_data = label23_data = []
        label24_data = label25_data = label26_data = label27_data = label28_data =[]

        for row in csv_reader:
            data = []
            for char in row:
                if char == 'None':
                    data.append(0)
                else:
                    data.append(np.float32(char))  # transform data from format of string to float32 将数据从字符串格式转换为float32格式
            if data[-1] == 0:
                label0_data.append(data)
            if data[-1] == 1:
                label1_data.append(data)
            if data[-1] == 2:
                label2_data.append(data)
            if data[-1] == 3:
                label3_data.append(data)
            if data[-1] == 4:
                label4_data.append(data)
            if data[-1] == 5:
                label5_data.append(data)  #
            if data[-1] == 6:
                label6_data.append(data)  #
            if data[-1] == 7:
                label7_data.append(data)  # 没有标签是7的 因为只捕获到1条数据
            if data[-1] == 8:
                label8_data.append(data)
            if data[-1] == 9:
                label9_data.append(data)
            if data[-1] == 10:
                label10_data.append(data)
            if data[-1] == 11:
                label11_data.append(data)
            if data[-1] == 12:
                label12_data.append(data)
            if data[-1] == 13:
                label13_data.append(data)
            if data[-1] == 14:
                label14_data.append(data)
            if data[-1] == 15:
                label15_data.append(data)
            if data[-1] == 16:
                label16_data.append(data)
            if data[-1] == 17:
                label17_data.append(data)
            if data[-1] == 18:
                label18_data.append(data)
            if data[-1] == 19:
                label19_data.append(data)
            if data[-1] == 20:
                label20_data.append(data)
            if data[-1] == 21:
                label21_data.append(data)
            if data[-1] == 22:
                label22_data.append(data)
            if data[-1] == 23:
                label23_data.append(data)
            if data[-1] == 24:
                label24_data.append(data)
            if data[-1] == 25:
                label25_data.append(data)
            if data[-1] == 26:
                label26_data.append(data)
            if data[-1] == 27:
                label27_data.append(data)
            if data[-1] == 28:
                label28_data.append(data)
            if self.label_status.get(str(int(data[-1])), 0) > 0:
                self.label_status[str(int(data[-1]))] += 1
            else:
                self.label_status[str(int(data[-1]))] = 1
        self.train_data = label0_data + label1_data + label2_data + label3_data + label4_data + label5_data + \
                          label6_data + label7_data + label8_data + label9_data + label10_data + label11_data + \
                          label12_data + label13_data + label14_data + label15_data + label16_data + label17_data + \
                          label18_data + label19_data + label20_data + label21_data + label22_data + label23_data + \
                          label24_data + label25_data + label26_data + label27_data + label28_data
        filename = '../test_data.csv'
        csv_reader = csv.reader(open(filename))
        for row in csv_reader:
            data = []
            for char in row:
                if char == 'None':
                    data.append(0)
                else:
                    data.append(np.float32(char))  # transform data from format of string to float32
            self.test_data.append(data)
        self.train_data = self.normalization(self.train_data)
        self.test_data = self.normalization(self.test_data)
        self.train_length = len(self.train_data)
        self.test_length = len(self.test_data)
        np.random.shuffle(self.train_data)
        print('init data completed!')

    def predict_top_k(self, x_feature, k=5):
        predict = self.sess.run(self.y, feed_dict={self.x_input: [x_feature]})[0]
        top_k_indices = np.argsort(predict)[-k:][::-1]  # 获取前k个预测标签的索引
        return top_k_indices

    def test(self):
        label_count = {}  # 标签的总数量
        label_correct = {}  # 标签预测正确的数量
        length = len(self.test_data)
        count = 0
        top1_correct = 0
        top3_correct = 0
        top5_correct = 0
        y_true = []
        y_pred = []
        for row in self.test_data:
            feature = row[0:-1]
            label = row[-1]
            count += 1
            x_feature = np.array(feature)
            top_k_labels = self.predict_top_k(x_feature, k=5)  # 获取前5个预测标签
            y_true.append(label)
            y_pred.append(top_k_labels[0])  # 记录TOP-1预测结果

            if str(int(label)) not in label_count:
                label_count[str(int(label))] = 0
                label_correct[str(int(label))] = 0
            if label == top_k_labels[0]:  # TOP-1预测  逐个判断top1-top5
                label_correct[str(int(label))] += 1
                top1_correct += 1
            if label in top_k_labels[:3]:  # TOP-3预测
                top3_correct += 1
            if label in top_k_labels[:5]:  # TOP-5预测
                top5_correct += 1
            label_count[str(int(label))] += 1
            if count % 10000 == 0:
                print(f"Processed {count} rows")

        accuracy1 = {}
        for label in sorted(label_count):
            accuracy1[label] = label_correct[label] / label_count[label]
            print(label, accuracy1[label], label_correct[label], label_count[label])

        top1_accuracy = top1_correct / length
        top3_accuracy = top3_correct / length
        top5_accuracy = top5_correct / length
        print(f"Top-1 accuracy: {top1_accuracy}")
        print(f"Top-3 accuracy: {top3_accuracy}")
        print(f"Top-5 accuracy: {top5_accuracy}")

        # 计算Macro-F1和Micro-F1
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        micro_f1 = f1_score(y_true, y_pred, average='micro')
        print(f"Macro-F1: {macro_f1}")
        print(f"Micro-F1: {micro_f1}")
        return top1_accuracy

    def test2(self): # 仅返回top1测试结果
        y_true = []
        y_pred = []
        for row in self.test_data:
            feature = row[0:-1]
            label = row[-1]
            x_feature = np.array(feature)
            top_k_labels = self.predict_top_k(x_feature, k=5)  # 获取前5个预测标签
            y_true.append(label)
            y_pred.append(top_k_labels[0])  # 记录TOP-1预测结果
        # 计算Macro-F1和Micro-F1
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        micro_f1 = f1_score(y_true, y_pred, average='micro')
        print(f"Macro-F1: {macro_f1}")
        print(f"Micro-F1: {micro_f1}")
        return macro_f1,micro_f1