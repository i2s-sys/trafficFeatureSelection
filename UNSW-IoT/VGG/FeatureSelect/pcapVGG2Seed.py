# 用于二次训练 有必要分开
import gc
import time

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from sklearn.metrics import f1_score
import numpy as np
import csv, random

tf.compat.v1.disable_eager_execution()
DATA_DIM = 72 # 特征
OUTPUT_DIM = 29
LEARNING_RATE = 0.0001
BETA = 0.999
GAMMA = 1
# MIN_LEARNINGRATE = 0.00001 # 学习率最小衰减至此
BATCH_SIZE = 128
TRAIN_FILE = '../../train_data_S.csv'
TEST_FILE = '../../test_data_S.csv'

MODEL_SAVE_PATH = './model/'
MODEL_SAVE_PATH2 = './model2/'
MODEL_NAME = 'model'
MODEL_NAME2 = 'model2'
top_k_values=[]
top_k_indice=[]

def set_seed(seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

class VGG2(): # 第二次训练
    def __init__(self,lossType,dim,selected_features=[],seed=25):  # 需输入维度 即当前特征数
        self.lossType = lossType
        set_seed(seed)
        self.gamma = GAMMA
        self.dim = dim
        self.top_k_indices = selected_features
        self.seed = seed
        self.learning_rate = LEARNING_RATE
        self.batch_size = BATCH_SIZE
        self.epoch_count = 0
        self.x_input = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, dim])
        self.target = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None])
        self.train_step = tf.Variable(0, trainable=False)
        self.loss_history = []
        self.micro_F1List = []
        self.macro_F1List = []
        self.init_data()
        self.total_iterations_per_epoch = self.train_length // BATCH_SIZE
        self.create_VGG(dim)
        beta = BETA  # cb数据预处理
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
        self.ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH2)
        self.saver = tf.compat.v1.train.Saver()
        self.train_start = 0
        if self.ckpt and self.ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, self.ckpt.model_checkpoint_path)
            self.train_start = self.sess.run(self.train_step)

    def create_VGG(self, NEW_DIM):
        scaled_input = tf.reshape(self.x_input, [-1, NEW_DIM, 1, 1])  # 确保输入形状正确
        scaled_input = tf.keras.layers.Input(tensor=scaled_input)
        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                                   name='conv_1',
                                   kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.seed))(
            scaled_input)
        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                                   name='conv_2',
                                   kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.seed))(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool_1')(x)
        x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                                   name='conv_3',
                                   kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.seed))(x)
        x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                                   name='conv_4',
                                   kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.seed))(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool_2')(x)
        x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same',
                                   name='conv_5',
                                   kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.seed))(x)
        x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same',
                                   name='conv_6',
                                   kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.seed))(x)
        x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same',
                                   name='conv_7',
                                   kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.seed))(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool_3')(x)
        x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same',
                                   name='conv_8',
                                   kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.seed))(x)
        # 全连接层，添加种子
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(1024, activation='relu', name='fc_1',
                                  kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.seed))(x)
        x = tf.keras.layers.Dropout(0.5)(x)  # 使用 Dropout 层
        x = tf.keras.layers.Dense(1024, activation='relu', name='fc_2',
                                  kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.seed))(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        self.y = tf.keras.layers.Dense(OUTPUT_DIM, name='fc_3',
                                       kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.seed))(x)

    def build_loss(self):
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.y, labels=self.target)
        class_weights = tf.constant([self.weights[str(i)] for i in range(len(self.weights))], dtype=tf.float32)
        # cb
        sample_weights = tf.gather(class_weights, self.target)
        cbce = tf.multiply(ce, sample_weights)
        # cbfocalLoss
        softmax_probs = tf.nn.softmax(self.y)
        labels_one_hot = tf.one_hot(self.target, depth=len(self.weights))
        pt = tf.reduce_sum(labels_one_hot * softmax_probs, axis=1)
        modulator = tf.pow(1.0 - pt, self.gamma)
        focal_loss = modulator * ce
        cb_focal_loss = tf.multiply(focal_loss, sample_weights)
        if (self.lossType == "ce"):
            self.loss = tf.reduce_sum(ce)
        elif (self.lossType == "cb"):
            self.loss = tf.reduce_sum(cbce)
        elif (self.lossType == "cb_focal_loss"):
            self.loss = tf.reduce_sum(cb_focal_loss)
        self.train_op = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(self.loss,
                                                                                      global_step=self.train_step)
        self.sess.run(tf.compat.v1.global_variables_initializer())

    def train(self):  # 每次训练都是一个epoch遍历完整个训练集
        total_loss = 0
        num_batches = 0
        self.train_start = 0
        epoch_start_time = time.time()  # 记录epoch开始时间
        for i in range(self.total_iterations_per_epoch):
            step = self.train_start
            batch = self.get_a_train_batch(step)
            data, label = self.get_data_label(batch)
            _, y = self.sess.run([self.train_op, self.y], feed_dict={self.x_input: data, self.target: label})
            curr_loss, y = self.sess.run([self.loss, self.y], feed_dict={self.x_input: data, self.target: label})
            total_loss += curr_loss
            num_batches += 1
            self.train_start += 1
        average_loss = total_loss / num_batches
        epoch_end_time = time.time()  # 记录epoch结束时间
        epoch_duration = epoch_end_time - epoch_start_time  # 计算epoch时长
        print(f'Epoch {self.epoch_count + 1} completed, average loss: {average_loss}, duration: {epoch_duration:.2f} seconds')
        self.loss_history.append(average_loss)
        self.test2()

    def get_a_train_batch(self, step):
        '''从训练数据集中获取一个批次的数据 '''
        min_index = step * self.batch_size
        max_index = min_index + self.batch_size
        if max_index > self.train_length:
            max_index = self.train_length
        return self.train_data[min_index:max_index]  # 返回min到max之间的数据样本

    def get_data_label(self, batch):
        data = np.delete(batch, -1, axis=1)
        label = np.array(batch, dtype=np.int32)[:, -1]
        return data, label

    def init_data(self):  # 此处的init_data和第一个DNN是不一样的，需要按照排序数组筛选
        self.train_data = []
        self.test_data = []  # 初始化训练和测试数据
        self.label_status = {}

        # 初始化标签数据字典
        label_data = {i: [] for i in range(29)}

        # 读取训练数据
        filename = TRAIN_FILE
        with open(filename, 'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                data = []
                for i, char in enumerate(row):  # 在indices里面才加入data
                    if i in self.top_k_indices or i == len(row) - 1:
                        if char == 'None':
                            data.append(0)
                        else:
                            data.append(np.float32(char))  # 将数据从字符串格式转换为float32格式

                label = int(data[-1])
                if label in label_data:
                    label_data[label].append(data)

                if str(label) not in self.label_status:
                    self.label_status[str(label)] = 0
                self.label_status[str(label)] += 1

        self.train_data = [item for sublist in label_data.values() for item in sublist]

        # 读取测试数据
        filename = TEST_FILE
        with open(filename, 'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                data = []
                for i, char in enumerate(row):
                    if i in self.top_k_indices or i == len(row) - 1:
                        if char == 'None':
                            data.append(0)
                        else:
                            data.append(np.float32(char))  # 将数据从字符串格式转换为float32格式
                self.test_data.append(data)

        self.train_length = len(self.train_data)
        self.test_length = len(self.test_data)
        self.train_data = self.normalization(self.train_data)
        self.test_data = self.normalization(self.test_data)
         
        print('init data completed!')

    def normalization(self, minibatch):
        data = np.delete(minibatch, -1, axis=1)
        labels = np.array(minibatch, dtype=np.int64)[:, -1]
        mmax = np.max(data, axis=0)
        mmin = np.min(data, axis=0)
        for i in range(len(mmax)):
            if mmax[i] == mmin[i]:
                mmax[i] += 0.000001  # avoid getting devided by 0
        res = (data - mmin) / (mmax - mmin)
        res = np.c_[res, labels]
        return res

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
        del y_true, y_pred, macro_f1, micro_f1
        gc.collect()
