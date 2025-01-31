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
DATA_DIM = 41 # 特征
OUTPUT_DIM = 2
LEARNING_RATE = 0.0001
# MIN_LEARNINGRATE = 0.00001 # 学习率最小衰减至此
BATCH_SIZE = 128
TRAIN_FILE = '../../train_data.csv'
TEST_FILE = '../../test_data.csv'

MODEL_SAVE_PATH = './model/'
MODEL_SAVE_PATH2 = './model2/'
MODEL_NAME = 'model'
MODEL_NAME2 = 'model2'
top_k_values=[]
top_k_indice=[]

class VGG():
    def __init__(self,lossType,seed,beta,gamma):  # 不输入维度 默认是DATA_DIM
        print(f"BETA = {beta}, GAMMA = {gamma}")
        self.beta = beta
        self.gamma = gamma
        self.learning_rate = LEARNING_RATE
        self.batch_size = BATCH_SIZE
        self.lossType = lossType
        self.seed = seed
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
        if(self.lossType == "ce"):
            self.loss = tf.reduce_sum(ce)
        elif(self.lossType == "cb"):
            self.loss = tf.reduce_sum(cbce)
        elif(self.lossType == "cb_focal_loss"):
            self.loss = tf.reduce_sum(cb_focal_loss)
        self.train_op = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(self.loss,global_step=self.train_step)
        self.sess.run(tf.compat.v1.global_variables_initializer())

    def train(self):
        total_loss = 0
        num_batches = 0
        self.train_start = 0
        epoch_start_time = time.time()
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
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        print(f'Epoch {self.epoch_count + 1} completed, average loss: {average_loss}, duration: {epoch_duration:.2f} seconds')
        micro_F1, macro_F1 = self.test2()
        self.micro_F1List.append(micro_F1)
        self.macro_F1List.append(macro_F1)

    def get_a_train_batch(self, step):
        min_index = step * self.batch_size
        max_index = min_index + self.batch_size
        if max_index > self.train_length:
            max_index = self.train_length
        return self.train_data[min_index:max_index]

    def get_data_label(self, batch):
        data = np.delete(batch, -1, axis=1)
        label = np.array(batch, dtype=np.int32)[:, -1]
        return data, label

    def init_data(self):
        self.train_data = []
        self.test_data = []
        self.label_status = {}
        filename = TRAIN_FILE
        csv_reader = csv.reader(open(filename))
        label_data = [[] for _ in range(6)]
        for row in csv_reader:
            data = []
            for char in row:
                if char == 'None':
                    data.append(0)
                else:
                    data.append(np.float32(char))  # 将数据从字符串格式转换为float32格式
            label = int(data[-1])
            if label < 2:  # 只保留2类的数据
                label_data[label].append(data)
            if self.label_status.get(str(label), 0) > 0:
                self.label_status[str(label)] += 1
            else:
                self.label_status[str(label)] = 1
        self.train_data = sum(label_data, [])
        filename = TEST_FILE
        csv_reader = csv.reader(open(filename))
        for row in csv_reader:
            data = []
            for char in row:
                if char == 'None':
                    data.append(0)
                else:
                    data.append(np.float32(char))
            self.test_data.append(data)
        self.train_data = self.normalization(self.train_data)
        self.test_data = self.normalization(self.test_data)
        self.train_length = len(self.train_data)
        self.test_length = len(self.test_data)
        np.random.shuffle(self.train_data)
        print('init data completed!')

    def init_data2class(self):
        self.train_data = []
        self.test_data = []
        self.label_status = {}
        def process_file(filename):
            normal_count = 0
            abnormal_count = 0
            data_list = []
            csv_reader = csv.reader(open(filename))
            for row in csv_reader:
                data = []
                for char in row[:-1]:
                    if char == 'None':
                        data.append(0)
                    else:
                        data.append(np.float32(char))
                label = row[-1]
                if label == 'normal':
                    data.append(0)
                    normal_count += 1
                    label_1 = "0"
                else:
                    data.append(1)
                    abnormal_count += 1
                    label_1 = "1"
                if self.label_status.get(label_1, 0) > 0:
                    self.label_status[label_1] += 1
                else:
                    self.label_status[label_1] = 1
                data_list.append(data)
            return data_list, normal_count, abnormal_count
        self.train_data, normal_count, abnormal_count = process_file(TRAIN_FILE)
        if (normal_count + abnormal_count) > 0:
            self.threshold = (normal_count / (normal_count + abnormal_count)) * 100
        self.test_data, _, _ = process_file(TEST_FILE)
        self.train_data = self.normalization(self.train_data)
        self.test_data = self.normalization(self.test_data)
        self.train_length = len(self.train_data)
        self.test_length = len(self.test_data)
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

    def test2(self):
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