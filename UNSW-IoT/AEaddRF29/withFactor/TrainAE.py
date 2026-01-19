# TensorFlow 2.9.0 compatible training script for AEWithFactor
import os
import time
import numpy as np
import tensorflow as tf
from AEWithFactor import AE

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def configure_gpu():
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU配置成功: {len(gpus)} 个GPU")
        else:
            print("未检测到GPU，将使用CPU")
    except Exception as e:
        print(f"GPU配置失败: {e}")


def main():
    configure_gpu()
    SEED = 25
    TRAIN_EPOCH = 30
    curr_time = time.strftime("%Y%m%d%H%M%S", time.localtime())

    ae = AE(seed=SEED)
    print('start training...')

    start_time = time.time()
    for epoch in range(TRAIN_EPOCH):
        ae.train()
        ae.epoch_count += 1
        if ae.earlyStop:
            print(f"Early stopping at epoch {epoch + 1}")
            break
    end_time = time.time()
    print("ae_loss_history", ae.loss_history)
    print("ae_TSMRecord", ae.TSMRecord)

    # 保存 scaling_factor
    scaling_path = os.path.join(os.path.dirname(__file__), "scaling_factor.npy")
    np.save(scaling_path, ae.model.scaling_factor.numpy())
    print(f"scaling_factor saved to {scaling_path}")

    print('start testing...')
    ae.train_classifier()


if __name__ == "__main__":
    main()
