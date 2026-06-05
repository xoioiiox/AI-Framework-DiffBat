import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.data import Dataset as tfds
from gaussian_diffusion import GaussianDiffusion
from diffusion_model import DiffusionModel
from unet import build_model
from postproc_utils import PostProcess
import platform
import subprocess

def is_m_chip():
    """
    Checks if the current machine is a Mac with an M-chip (Apple Silicon).

    Returns:
        bool: True if the machine is a Mac with an M-chip, False otherwise.
    """
    # Check if the operating system is macOS
    if platform.system() != 'Darwin':
        return False

    # Check if the architecture is arm64
    try:
        # 'uname -m' returns 'arm64' for Apple Silicon
        chip_type = subprocess.check_output(['uname', '-m']).decode('utf-8').strip()
        return chip_type == 'arm64'
    except Exception as e:
        print(f"Error during check: {e}")
        return False

class TrainingProgressCallback(keras.callbacks.Callback):
    """打印更直观的训练进度，避免长时间训练时看不到 batch 级反馈。"""

    def __init__(self, train_steps=None, log_every=10):
        super().__init__()
        self.train_steps = train_steps
        self.log_every = log_every

    def on_train_begin(self, logs=None):
        if self.train_steps is None:
            print("开始训练：数据集步数未知，将每隔若干 batch 打印一次 loss。")
        else:
            print(f"开始训练：每个 epoch 约 {self.train_steps} 个 batch。")

    def on_epoch_begin(self, epoch, logs=None):
        print(f"\nEpoch {epoch + 1}/{self.params.get('epochs', '?')} 开始")

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        current_step = batch + 1
        should_log = current_step % self.log_every == 0
        if self.train_steps is not None:
            should_log = should_log or current_step == self.train_steps

        if should_log:
            loss = logs.get("loss")
            loss_text = f"{loss:.6f}" if loss is not None else "N/A"
            if self.train_steps is None:
                print(f"  batch {current_step}: loss={loss_text}")
            else:
                progress = current_step / self.train_steps * 100
                print(f"  batch {current_step}/{self.train_steps} ({progress:.1f}%): loss={loss_text}")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get("loss")
        val_loss = logs.get("val_loss")
        loss_text = f"{loss:.6f}" if loss is not None else "N/A"
        val_loss_text = f"{val_loss:.6f}" if val_loss is not None else "N/A"
        print(f"Epoch {epoch + 1} 结束：loss={loss_text}, val_loss={val_loss_text}")

# GPU 显存按需增长，避免 TensorFlow 一启动就占满全部显存。
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

tf.config.set_visible_devices(gpus, 'GPU')

# 选择要训练的电池数据集，对应 data/{battery_dataset}_train_ds。
battery_dataset = 'matr_1'

# DDPM 与训练超参数。
num_epochs = 500
total_timesteps = 1000
learning_rate = 1e-3

# U-Net 结构超参数。sequence_length=256 表示模型直接生成 256 点的 SOH 曲线。
sequence_length = 256
first_conv_channels = 8
channel_multiplier = [1, 2, 4, 8]
widths = [first_conv_channels * mult for mult in channel_multiplier]
has_attention = [False, False, True, True]
num_res_blocks = 2  # Number of residual blocks
# classifier-free guidance 训练时丢弃条件的概率。
p_uncond = 0.2

# 构建当前训练网络和 EMA 网络，两者结构完全一致。
network = build_model(
    sequence_length=sequence_length,
    widths=widths,
    has_attention=has_attention,
    first_conv_channels = first_conv_channels,
    num_res_blocks=num_res_blocks,
)
ema_network = build_model(
    sequence_length=sequence_length,
    widths=widths,
    has_attention=has_attention,
    first_conv_channels = first_conv_channels,
    num_res_blocks=num_res_blocks,
)
ema_network.set_weights(network.get_weights())  # Initially the weights are the same

# 扩散过程工具类，负责 q_sample 和 p_sample 等公式。
gdf_util = GaussianDiffusion(timesteps=total_timesteps)

# DiffusionModel 封装 Keras 自定义 train_step/test_step。
model = DiffusionModel(
    network=network,
    ema_network=ema_network,
    gdf_util=gdf_util,
    timesteps=total_timesteps,
    p_uncond=p_uncond,
)

# 读取 TensorFlow Dataset snapshot 格式的数据。
# 每条样本在模型中会被解包为 images, _, protocol。
train_ds = tfds.load(f'./data/{battery_dataset}_train_ds')
test_ds = tfds.load(f'./data/{battery_dataset}_test_ds')

# 尝试获取每个 epoch 的 batch 数，用于显示百分比；未知时只显示 batch 序号。
train_cardinality = tf.data.experimental.cardinality(train_ds).numpy()
train_steps = None if train_cardinality < 0 else int(train_cardinality)

# Apple Silicon 上用 legacy.Adam，其它平台用标准 Adam。
if is_m_chip():
    keras_optimizer = keras.optimizers.legacy.Adam(learning_rate=learning_rate)
else:
    keras_optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

model.compile(
    loss=keras.losses.MeanSquaredError(),
    optimizer=keras_optimizer,
    weighted_metrics=[]
)

# 保存验证集 loss 最低的一组权重。
checkpoint_filepath = f'./checkpoints/{battery_dataset}_checkpoint.weights.h5'
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    verbose=1,
    monitor='val_loss',
    save_freq = 'epoch',
    mode='min',
    save_best_only=True)

progress_callback = TrainingProgressCallback(train_steps=train_steps, log_every=10)

# 开始训练。当前 num_epochs=1 只是演示，正式复现实验需要增加 epoch。
hist = model.fit(
    train_ds,
    epochs=num_epochs,
    validation_data=test_ds,
    callbacks=[progress_callback, model_checkpoint_callback],
    verbose=1,
    validation_freq=1,
)

# 加载最佳 checkpoint，并分别保存普通网络和 EMA 网络。
model.load_weights(checkpoint_filepath)
network.save(f'./checkpoints/{battery_dataset}_network', save_format='tf')
ema_network.save(f'./checkpoints/{battery_dataset}_ema_network', save_format='tf')
