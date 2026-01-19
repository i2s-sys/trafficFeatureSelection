# TensorFlow 2.9.0 compatible AutoEncoder + RandomForest with feature scaling factor (AEaddRF)
import time
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras import layers, Sequential, Model
from sklearn.metrics import f1_score
import numpy as np
import csv, random, os

# 默认保持图执行（graph mode），性能优于强制 eager
tf.config.run_functions_eagerly(False)

# 设置随机种子确保结果可复现
def set_deterministic_seed(seed):
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

DATA_DIM = 72
K = 32
OUTPUT_DIM = 29
BETA = 0.999
GAMMA = 1
TRAIN_FILE = '../../train_data.csv'   # 29 类全集
TEST_FILE = '../../test_data.csv'
LEARNING_RATE = 0.0001
BATCH_SIZE = 512
MODEL_SAVE_PATH = '../model/'
MODEL_SAVE_PATH2 = './model2/'
MODEL_NAME = 'model'
MODEL_NAME2 = 'model2'
KEEP_PROB = 0.5
LOG_INTERVAL = 500  # 每多少个 batch 打印一次进度

class AEModelWithFactor(Model):
    def __init__(self, data_dim, seed=None):
        super().__init__()
        self.data_dim = data_dim
        self.seed = seed

        # 可训练的特征缩放因子，形状 [1, data_dim]，初始为 1
        self.scaling_factor = tf.Variable(
            tf.ones([1, data_dim], dtype=tf.float32),
            trainable=True,
            name='scaling_factor'
        )

        # 编码器
        self.encoder1 = layers.Dense(128, activation='relu',
                                     kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.seed))
        self.encoder2 = layers.Dense(64, activation='relu',
                                     kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.seed))

        # 解码器
        self.decoder1 = layers.Dense(128, activation='relu',
                                     kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.seed))
        self.decoder2 = layers.Dense(data_dim,
                                     kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.seed))

    def call(self, inputs, training=None):
        batch_size = tf.shape(inputs)[0]
        scaling_factor_ext = tf.tile(self.scaling_factor, [batch_size, 1])
        scaled_input = inputs * scaling_factor_ext
        encoded = self.encoder1(scaled_input)
        encoded = self.encoder2(encoded)
        decoded = self.decoder1(encoded)
        decoded = self.decoder2(decoded)
        return decoded, encoded


class AE():
    def __init__(self, seed=25):
        set_deterministic_seed(seed)
        self.seed = seed
        self.loss_history = []
        self.ES_THRESHOLD = 3
        self.micro_F1List = []
        self.macro_F1List = []
        self.intersection_sets = []
        self.TSMRecord = []
        self.batch_size = BATCH_SIZE
        self.earlyStop = False
        self.learning_rate = LEARNING_RATE
        self.epoch_count = 0

        self.init_data()
        self.total_iterations_per_epoch = max(1, (self.train_length + BATCH_SIZE - 1) // BATCH_SIZE)

        self.model = AEModelWithFactor(data_dim=DATA_DIM, seed=self.seed)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def compute_loss(self, inputs, outputs):
        l1_regularizer = tf.keras.regularizers.l1(0.001)
        reg_penalty = l1_regularizer(self.model.scaling_factor)
        recon_loss = tf.sqrt(tf.reduce_mean(tf.square(inputs - outputs)))
        return recon_loss + reg_penalty

    @tf.function
    def train_step(self, data):
        with tf.GradientTape() as tape:
            decoded, encoded = self.model(data, training=True)
            loss = self.compute_loss(data, decoded)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss, encoded

    def train_classifier(self):
        num_batches = 0
        epoch_start_time = time.time()
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        all_encoded_features = []
        all_labels = []

        for step in range(self.total_iterations_per_epoch):
            batch = self.get_a_train_batch(step)
            data, label = self.getBatch_data_label(batch)
            _, encoded_feature = self.train_step(data)
            encoded_feature = np.squeeze(encoded_feature.numpy())
            all_encoded_features.append(encoded_feature)
            all_labels.append(label)
            num_batches += 1

        all_encoded_features = np.vstack(all_encoded_features)
        all_labels = np.hstack(all_labels)
        self.classifier.fit(all_encoded_features, all_labels)
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        print(f'classifier fit duration: {epoch_duration:.2f} seconds')
        self.test_classifier()

    def test_classifier(self):
        all_encoded_features = []
        all_labels = []
        label_count = {}
        label_correct = {}
        self.test_iterations = max(1, self.test_length // BATCH_SIZE)

        for step in range(self.test_iterations):
            batch = self.get_a_test_batch(step)
            data, label = self.getBatch_data_label(batch)
            _, encoded_feature = self.model(data, training=False)
            encoded_feature = np.squeeze(encoded_feature.numpy())
            all_encoded_features.append(encoded_feature)
            all_labels.append(label)

        all_encoded_features = np.vstack(all_encoded_features)
        all_labels = np.hstack(all_labels)
        predictions = self.classifier.predict(all_encoded_features)

        for t, p in zip(all_labels, predictions):
            key = str(int(t))
            label_count[key] = label_count.get(key, 0) + 1
            if t == p:
                label_correct[key] = label_correct.get(key, 0) + 1

        for key in sorted(label_count):
            acc = label_correct.get(key, 0) / label_count[key]
            print(f'Label {key}: Accuracy {acc:.2f} ({label_correct.get(key,0)}/{label_count[key]})')

        macro_f1 = f1_score(all_labels, predictions, average='macro')
        micro_f1 = f1_score(all_labels, predictions, average='micro')
        print(f'Macro-F1: {macro_f1}')
        print(f'Micro-F1: {micro_f1}')

    def train(self):
        self.epoch_rmse = []
        total_loss = 0
        num_batches = 0
        epoch_start_time = time.time()

        np.random.shuffle(self.train_data)

        for step in range(self.total_iterations_per_epoch):
            batch = self.get_a_train_batch(step)
            data, label = self.getBatch_data_label(batch)
            loss, _ = self.train_step(data)
            self.epoch_rmse.append(loss.numpy())
            total_loss += loss.numpy()
            num_batches += 1
            if (step + 1) % LOG_INTERVAL == 0 or (step + 1) == self.total_iterations_per_epoch:
                print(f"  train batch {step + 1}/{self.total_iterations_per_epoch}, loss={loss.numpy():.6f}", flush=True)

        average_loss = total_loss / self.train_length
        self.loss_history.append(average_loss)
        epoch_duration = time.time() - epoch_start_time
        print(f'Epoch {self.epoch_count + 1} completed, avg loss: {average_loss:.6f}, duration: {epoch_duration:.2f}s')

        # 简单早停：记录前 K 大因子位置的交集
        scaling_val = self.model.scaling_factor.numpy()
        max_indices = np.argsort(scaling_val.flatten())[::-1][:K]
        max_set = set(max_indices)
        self.intersection_sets.append(max_set)
        if len(self.intersection_sets) > self.ES_THRESHOLD:
            self.intersection_sets.pop(0)
        if len(self.intersection_sets) > 0:
            inter = set.intersection(*self.intersection_sets)
            print(f"Epoch {self.epoch_count + 1}, Intersection size: {len(inter)}")
            if len(inter) == K and self.epoch_count > self.ES_THRESHOLD:
                self.earlyStop = True
            self.TSMRecord.append(len(inter))

        return average_loss

    def getBatch_data_label(self, batch):
        data = np.delete(batch, -1, axis=1).astype(np.float32)
        label = np.array(batch, dtype=np.int32)[:, -1]
        return data, label

    def get_a_train_batch(self, step):
        min_index = step * self.batch_size
        max_index = min_index + self.batch_size
        max_index = min(max_index, self.train_length)
        return self.train_data[min_index:max_index]

    def get_a_test_batch(self, step):
        min_index = step * self.batch_size
        max_index = min_index + self.batch_size
        max_index = min(max_index, self.test_length)
        return self.test_data[min_index:max_index]

    def init_data(self):
        self.train_data = []
        self.test_data = []
        self.label_status = {}
        label_data = [[] for _ in range(OUTPUT_DIM)]

        with open(TRAIN_FILE, 'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                data = []
                for char in row:
                    if char in ('None', ''):
                        data.append(0)
                    else:
                        data.append(np.float32(char))
                label = int(data[-1])
                if label < OUTPUT_DIM:
                    label_data[label].append(data)
                self.label_status[str(label)] = self.label_status.get(str(label), 0) + 1
        self.train_data = sum(label_data, [])

        with open(TEST_FILE, 'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                data = []
                for char in row:
                    if char in ('None', ''):
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
