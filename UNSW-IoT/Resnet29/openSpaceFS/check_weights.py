"""
快速校验权重文件能否成功加载，避免跑完整测试后才发现维度不匹配。

逻辑：
- 针对每个权重文件，尝试用输出维度20和25各构建一次模型并load_weights。
- 若20或25其中之一成功，则记录“可加载的输出维度”。
- 若都失败，打印报错信息。

输入：
- 默认检查项目中列出的8个权重；可通过 --weights 参数手动指定（逗号分隔）。

用法示例：
    cd UNSW-IoT/Resnet29/openSpaceFS
    python check_weights.py
    python check_weights.py --weights checkpoints/fpa_resnet2_k32_20260113133007.weights.h5
"""

import argparse
import os
import tensorflow as tf
from tensorflow.keras import layers, Sequential, Model
import numpy as np

from pcapResnetPacketSeed import BasicBlock

# 模型输入维度固定为 32（topk=32），与训练保持一致
INPUT_DIM = 32
TRY_OUTPUT_DIMS = [20, 25]

DEFAULT_WEIGHTS = [
    "checkpoints/ow_resnet2_k32_20260114033635.weights.h5",
    "checkpoints/factorOW2_resnet2_k32_20260113140851.weights.h5",
    "checkpoints/ow_resnet2_k32_20260114034001.weights.h5",
    "checkpoints/PSOOW2_resnet2_k32_20260113133445.weights.h5",
    "checkpoints/ow_resnet2_k32_20260114033852.weights.h5",
    "checkpoints/fpaOW2_resnet2_k32_20260113133155.weights.h5",
    "checkpoints/infOW1_resnet2_k32_20260113133623.weights.h5",
    "checkpoints/infOW2_resnet2_k32_20260113133550.weights.h5",
]


class ResNetModelVar(Model):
    def __init__(self, dim, output_dim, seed=25):
        super().__init__()
        self.dim = dim
        self.seed = seed

        self.conv_layer = layers.Conv2D(64, kernel_size=(3, 3), strides=1, padding="same",
                                        kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.seed))
        self.stm = Sequential([
            self.conv_layer,
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPool2D(pool_size=(2, 2), strides=1, padding="same"),
        ])
        layer_dims = [2, 2, 2, 2]
        self.layer1 = self.build_resblock(64, layer_dims[0])
        self.layer2 = self.build_resblock(128, layer_dims[1], stride=2)
        self.layer3 = self.build_resblock(256, layer_dims[2], stride=2)
        self.layer4 = self.build_resblock(512, layer_dims[3], stride=2)
        self.avgpool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(output_dim)

    def build_resblock(self, filter_num, blocks, stride=1):
        res_blocks = Sequential()
        res_blocks.add(BasicBlock(filter_num, stride, seed=self.seed))
        for _ in range(1, blocks):
            res_blocks.add(BasicBlock(filter_num, stride=1, seed=self.seed))
        return res_blocks

    def call(self, inputs, training=None):
        x = tf.reshape(inputs, [tf.shape(inputs)[0], self.dim, 1, 1])
        x = self.stm(x, training=training)
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)
        x = self.avgpool(x)
        return self.fc(x)


def try_load(weight_path, out_dim):
    model = ResNetModelVar(dim=INPUT_DIM, output_dim=out_dim, seed=25)
    _ = model(tf.zeros((1, INPUT_DIM), dtype=tf.float32), training=False)
    model.load_weights(weight_path)
    return True


def main():
    parser = argparse.ArgumentParser(description="Check weight files for output-dim compatibility (20/25).")
    parser.add_argument("--weights", type=str, default=None,
                        help="逗号分隔的权重路径列表，留空则使用内置列表。")
    args = parser.parse_args()

    weights = DEFAULT_WEIGHTS
    if args.weights:
        weights = [w.strip() for w in args.weights.split(",") if w.strip()]

    for w in weights:
        if not os.path.exists(w):
            print(f"[MISS] {w}")
            continue
        ok_dims = []
        last_err = None
        for odim in TRY_OUTPUT_DIMS:
            try:
                try_load(w, odim)
                ok_dims.append(odim)
            except Exception as e:  # noqa: BLE001
                last_err = e
                continue
        if ok_dims:
            print(f"[OK] {w} -> compatible output_dim: {ok_dims}")
        else:
            print(f"[FAIL] {w} -> cannot load with dims {TRY_OUTPUT_DIMS}. last_error={last_err}")


if __name__ == "__main__":
    main()
