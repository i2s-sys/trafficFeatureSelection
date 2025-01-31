import time
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from sklearn.metrics import f1_score
import numpy as np
import csv, random

# 使用3层神经网络的分类器效果不好
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, GlobalMaxPool2D, Reshape, Multiply, Lambda
tf.compat.v1.disable_eager_execution()
DATA_DIM = 41
K = 32
OUTPUT_DIM = 2
BETA = 0.999
GAMMA = 1
TRAIN_FILE = '../train_data.csv'
TEST_FILE = '../test_data.csv'
LEARNING_RATE = 0.0001
BATCH_SIZE = 128
MODEL_SAVE_PATH = '../model/'
MODEL_SAVE_PATH2 = './model2/'
MODEL_NAME = 'model'
MODEL_NAME2 = 'model2'
KEEP_PROB = 0.5
top_k_values=[]
top_k_indice=[]
NUM_ATTENTION_CHANNELS=1
class AE():
    def __init__(self):
        self.loss_history = []
        self.micro_F1List = []
        self.macro_F1List = []
        self.intersection_sets = []
        self.TSMRecord = []
        self.batch_size = BATCH_SIZE
        self.earlyStop = False
        self.learning_rate = LEARNING_RATE
        self.epoch_count = 0
        self.x_input = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None,DATA_DIM])
        self.target = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None,DATA_DIM])
        self.classifier_target = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None])
        self.train_step = tf.Variable(0, trainable=False)
        self.init_data()
        self.total_iterations_per_epoch = self.train_length // BATCH_SIZE
        self.create_AE()
        self.build_loss()
        self.build_classifier()
        self.sess = tf.compat.v1.Session()
        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        self.saver = tf.compat.v1.train.Saver()
        self.train_start = 0
        if self.ckpt and self.ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, self.ckpt.model_checkpoint_path)
            self.train_start = self.sess.run(self.train_step)

    def create_AE(self):
        self.encoded = layers.Dense(128, activation='relu')(self.x_input)
        self.encoded = layers.Dense(64, activation='relu')(self.encoded)
        self.decoded = layers.Dense(128, activation='relu')(self.encoded)
        self.decoded = layers.Dense(DATA_DIM)(self.decoded)
        self.output = self.decoded

    def build_classifier(self):
        self.encoded = layers.Dense(128, activation='relu')(self.x_input)
        self.encoded = layers.Dense(64, activation='relu')(self.encoded)

    def build_loss(self):
        self.loss = tf.sqrt(tf.reduce_mean(tf.square(self.x_input - self.output)))
        self.train_op = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(self.loss,global_step=self.train_step)

    def train_classifier(self):
        num_batches = 0
        self.train_start = 0
        epoch_start_time = time.time()
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        all_encoded_features = []
        all_labels = []
        for _ in range(self.total_iterations_per_epoch):
            step = self.train_start
            batch = self.get_a_train_batch(step)
            data, label = self.getBatch_data_label(batch)
            encoded_feature = self.sess.run(self.encoded, feed_dict={self.x_input: data})
            encoded_feature = np.squeeze(encoded_feature)
            all_encoded_features.append(encoded_feature)
            all_labels.append(label)
            num_batches += 1
            self.train_start += 1
        # 在所有批次数据累积后训练分类器
        all_encoded_features = np.vstack(all_encoded_features)
        all_labels = np.hstack(all_labels)
        self.classifier.fit(all_encoded_features, all_labels)
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        print(f'duration: {epoch_duration:.2f} seconds')
        self.test_classifier()

    def test_classifier(self):
        self.train_start = 0
        all_encoded_features = []
        all_labels = []
        label_count = {}
        label_correct = {}
        self.test_iterations = self.test_length // BATCH_SIZE
        for _ in range(self.test_iterations):
            step = self.train_start
            batch = self.get_a_test_batch(step)
            data, label = self.getBatch_data_label(batch)
            encoded_feature = self.sess.run(self.encoded, feed_dict={self.x_input: data})
            encoded_feature = np.squeeze(encoded_feature)
            all_encoded_features.append(encoded_feature)
            all_labels.append(label)
            self.train_start += 1
        all_encoded_features = np.vstack(all_encoded_features)
        all_labels = np.hstack(all_labels)
        predictions = self.classifier.predict(all_encoded_features)
        # 统计每个类的正确预测次数和总预测次数
        for true_label, pred_label in zip(all_labels, predictions):
            true_label_str = str(int(true_label))
            if true_label_str not in label_count:
                label_count[true_label_str] = 0
                label_correct[true_label_str] = 0
            label_count[true_label_str] += 1
            if true_label == pred_label:
                label_correct[true_label_str] += 1

        # 计算并打印每个类的准确率
        for label in sorted(label_count):
            accuracy = label_correct[label] / label_count[label]
            print(f'Label {label}: Accuracy {accuracy:.2f} ({label_correct[label]}/{label_count[label]})')

        macro_f1 = f1_score(all_labels, predictions, average='macro')
        micro_f1 = f1_score(all_labels, predictions, average='micro')
        print(f'Macro-F1: {macro_f1}')
        print(f'Micro-F1: {micro_f1}')

    def train(self):
        self.epoch_rmse = []
        total_loss = 0
        num_batches = 0
        self.train_start = 0
        epoch_start_time = time.time()
        for _ in range(self.total_iterations_per_epoch):
            step = self.train_start
            batch = self.get_a_train_batch(step)
            data, label = self.getBatch_data_label(batch)
            _, output = self.sess.run([self.train_op, self.output],feed_dict={self.x_input: data, self.target: data})
            curr_loss, output = self.sess.run([self.loss, self.output],feed_dict={self.x_input: data, self.target: data})
            self.epoch_rmse.append(curr_loss)
            total_loss += curr_loss
            num_batches += 1
            self.train_start += 1
        average_loss = total_loss / self.train_length
        self.loss_history.append(average_loss)
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        print(f'Epoch {self.epoch_count + 1} completed, average loss: {average_loss}, duration: {epoch_duration:.2f} seconds')
        return average_loss

    def getBatch_data_label(self, batch):
        data = np.delete(batch, -1, axis=1)
        label = np.array(batch, dtype=np.int32)[:, -1]
        return data, label

    def get_a_train_batch(self, step):
        min_index = step * self.batch_size
        max_index = min_index + self.batch_size
        if max_index > self.train_length:
            max_index = self.train_length
        return self.train_data[min_index:max_index]

    def get_a_test_batch(self, step):
        min_index = step * self.batch_size
        max_index = min_index + self.batch_size
        if max_index > self.test_length:
            max_index = self.test_length
        return self.test_data[min_index:max_index]

    def init_data(self):
        self.train_data = []
        self.test_data = []
        self.label_status = {}
        filename = TRAIN_FILE
        csv_reader = csv.reader(open(filename))
        label_data = [[] for _ in range(OUTPUT_DIM)]
        for row in csv_reader:
            data = []
            for char in row:
                if char == 'None':
                    data.append(0)
                else:
                    data.append(np.float32(char))
            label = int(data[-1])
            if label < OUTPUT_DIM:
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

    def normalization(self, minibatch):
        data = np.delete(minibatch, -1, axis=1)
        labels = np.array(minibatch, dtype=np.int64)[:, -1]
        mmax = np.max(data, axis=0)
        mmin = np.min(data, axis=0)
        for i in range(len(mmax)):
            if mmax[i] == mmin[i]:
                mmax[i] += 0.000001
        res = (data - mmin) / (mmax - mmin)
        res = np.c_[res, labels]
        return res

    def predict_top_k(self, x_feature, k=5):
        predict = self.sess.run(self.classifier_output, feed_dict={self.x_input: [x_feature]})[0]
        top_k_indices = np.argsort(predict)[-k:][::-1]
        return top_k_indices

    def test(self):
        label_count = {}
        label_correct = {}
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
            top_k_labels = self.predict_top_k(x_feature, k=5)
            y_true.append(label)
            y_pred.append(top_k_labels[0])
            if str(int(label)) not in label_count:
                label_count[str(int(label))] = 0
                label_correct[str(int(label))] = 0
            if label == top_k_labels[0]:
                label_correct[str(int(label))] += 1
                top1_correct += 1
            if label in top_k_labels[:3]:
                top3_correct += 1
            if label in top_k_labels[:5]:
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
            top_k_labels = self.predict_top_k(x_feature, k=5)
            y_true.append(label)
            y_pred.append(top_k_labels[0])
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        micro_f1 = f1_score(y_true, y_pred, average='micro')
        print(f"Macro-F1: {macro_f1}")
        print(f"Micro-F1: {micro_f1}")
