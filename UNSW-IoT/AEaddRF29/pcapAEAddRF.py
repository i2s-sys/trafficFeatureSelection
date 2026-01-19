# TensorFlow 2.9.0 compatible AutoEncoder + RandomForest implementation for UNSW-IoT
import time
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from tensorflow import keras
from tensorflow.keras import layers, Sequential, Model
from sklearn.metrics import f1_score
import numpy as np
import csv, random

# Enable eager execution (default in TF 2.x)
tf.config.run_functions_eagerly(True)

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
OUTPUT_DIM = 29
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

class AEModel(Model):
    def __init__(self, data_dim, seed=None):
        super(AEModel, self).__init__()
        self.data_dim = data_dim
        self.seed = seed
        
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
        # Encoder
        encoded = self.encoder1(inputs)
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
        self.micro_F1List = []
        self.macro_F1List = []
        self.intersection_sets = []
        self.TSMRecord = []
        self.batch_size = BATCH_SIZE
        self.earlyStop = False
        self.learning_rate = LEARNING_RATE
        self.epoch_count = 0
        
        self.init_data()
        self.total_iterations_per_epoch = self.train_length // BATCH_SIZE
        
        # Create the model
        self.model = AEModel(data_dim=DATA_DIM, seed=self.seed)
        
        # Setup optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def compute_loss(self, inputs, outputs):
        return tf.sqrt(tf.reduce_mean(tf.square(inputs - outputs)))

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
        print(f'duration: {epoch_duration:.2f} seconds')
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
        
        average_loss = total_loss / self.train_length
        self.loss_history.append(average_loss)
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        print(f'Epoch {self.epoch_count + 1} completed, average loss: {average_loss:.6f}, duration: {epoch_duration:.2f} seconds')
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
        x_tensor = tf.constant([x_feature], dtype=tf.float32)
        _, encoded = self.model(x_tensor, training=False)
        encoded_feature = encoded.numpy()
        encoded_feature = np.squeeze(encoded_feature)
        predictions = self.classifier.predict([encoded_feature])
        top_k_indices = np.argsort(predictions)[-k:][::-1]
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
        return macro_f1, micro_f1
