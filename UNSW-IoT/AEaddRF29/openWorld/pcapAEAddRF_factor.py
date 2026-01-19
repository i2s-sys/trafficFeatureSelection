# TensorFlow 2.9.0 compatible AutoEncoder + RandomForest with feature selection for UNSW-IoT
import time
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from tensorflow import keras
from tensorflow.keras import layers, Sequential, Model
from sklearn.metrics import f1_score
import numpy as np
import csv, random

# 默认保持图执行（graph mode），比强制 eager 更快
tf.config.run_functions_eagerly(False)

# 设置随机种子确保结果可复现
def set_deterministic_seed(seed):
    """设置所有随机种子确保结果可复现"""
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

DATA_DIM = 72
K = 32
OUTPUT_DIM = 25
BETA = 0.999
GAMMA = 1
TRAIN_FILE = '../../OWtrain_data2.csv'  #封闭世界 数据集 25个设备
TEST_FILE = '../../OWtest_data2.csv'
LEARNING_RATE = 0.0001
BATCH_SIZE = 512
MODEL_SAVE_PATH = '../model/'
MODEL_SAVE_PATH2 = './model2/'
MODEL_NAME = 'model'
MODEL_NAME2 = 'model2'
KEEP_PROB = 0.5
top_k_values=[]
top_k_indice=[]
NUM_ATTENTION_CHANNELS=1
LOG_INTERVAL = 500  # 每多少个 batch 打印一次进度

class AEModelWithFactor(Model):
    def __init__(self, data_dim, seed=None):
        super(AEModelWithFactor, self).__init__()
        self.data_dim = data_dim
        self.seed = seed
        
        # Feature scaling factor for feature selection
        self.scaling_factor = tf.Variable(
            tf.constant(1, dtype=tf.float32, shape=[1, data_dim]), 
            trainable=True,
            name='scaling_factor'
        )
        
        # Encoder layers
        self.encoder1 = layers.Dense(128, activation='relu', 
                                   kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.seed))
        self.encoder2 = layers.Dense(64, activation='relu',
                                   kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.seed))
        
        # Decoder layers
        self.decoder1 = layers.Dense(128, activation='relu',
                                   kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.seed))
        self.decoder2 = layers.Dense(data_dim,
                                    kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.seed))
    
    def call(self, inputs, training=None):
        # Apply scaling factor
        batch_size = tf.shape(inputs)[0]
        scaling_factor_extended = tf.tile(self.scaling_factor, [batch_size, 1])
        scaled_input = tf.multiply(inputs, scaling_factor_extended)
        
        # Encoder
        encoded = self.encoder1(scaled_input)
        encoded = self.encoder2(encoded)
        
        # Decoder
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
        # ceil 取整，至少 1 个 batch
        self.total_iterations_per_epoch = max(1, (self.train_length + BATCH_SIZE - 1) // BATCH_SIZE)
        
        # Create the model
        self.model = AEModelWithFactor(data_dim=DATA_DIM, seed=self.seed)
        
        # Setup optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def compute_loss(self, inputs, outputs):
        l1_regularizer = tf.keras.regularizers.l1(0.001)
        regularization_penalty = l1_regularizer(self.model.scaling_factor)
        reconstruction_loss = tf.sqrt(tf.reduce_mean(tf.square(inputs - outputs)))
        return reconstruction_loss + regularization_penalty

    @tf.function
    def train_step(self, data):
        with tf.GradientTape() as tape:
            decoded, encoded = self.model(data, training=True)
            loss = self.compute_loss(data, decoded)
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
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
            encoded_feature = encoded_feature.numpy()
            encoded_feature = np.squeeze(encoded_feature)
            all_encoded_features.append(encoded_feature)
            all_labels.append(label)
            num_batches += 1
        
        # 在所有批次数据累积后训练分类器
        all_encoded_features = np.vstack(all_encoded_features)
        all_labels = np.hstack(all_labels)
        self.classifier.fit(all_encoded_features, all_labels)
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        print(f'classifier Epoch {self.epoch_count + 1} completed, duration: {epoch_duration:.2f} seconds')
        self.test_classifier()

    def test_classifier(self):
        all_encoded_features = []
        all_labels = []
        label_count = {}
        label_correct = {}
        self.test_iterations = self.test_length // BATCH_SIZE
        
        for step in range(self.test_iterations):
            batch = self.get_a_test_batch(step)
            data, label = self.getBatch_data_label(batch)
            
            _, encoded_feature = self.model(data, training=False)
            encoded_feature = encoded_feature.numpy()
            encoded_feature = np.squeeze(encoded_feature)
            all_encoded_features.append(encoded_feature)
            all_labels.append(label)
        
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
        epoch_start_time = time.time()
        
        # 随机打乱训练数据
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
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        print(f'Epoch {self.epoch_count + 1} completed, average loss: {average_loss:.6f}, duration: {epoch_duration:.2f} seconds')
        
        ''' 早停策略 每次都要'''
        keyFeatureNums = K
        scaling_factor_value = self.model.scaling_factor.numpy()
        max_indices = np.argsort(scaling_factor_value.flatten())[::-1][:keyFeatureNums]
        max_set = set(max_indices)
        self.intersection_sets.append(max_set)
        
        if len(self.intersection_sets) > self.ES_THRESHOLD:
            self.intersection_sets.pop(0)
        if len(self.intersection_sets) > 0:
            intersection = set.intersection(*self.intersection_sets)
            print(f"Epoch {self.epoch_count + 1}, Intersection size: {len(intersection)}")
            if len(intersection) == K and self.epoch_count > self.ES_THRESHOLD:
                self.earlyStop = True
            self.TSMRecord.append(len(intersection))
        
        return average_loss

    def getBatch_data_label(self, batch):
        data = np.delete(batch, -1, axis=1)
        data = data.astype(np.float32)
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
                if char in('None','') :
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
                if char in('None','') :
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
