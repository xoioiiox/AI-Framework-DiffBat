import numpy as np
import tensorflow as tf
from tensorflow import keras

class DiffusionModel(keras.Model):
    """封装训练、EMA 和反向采样逻辑。

    network 是正在训练的噪声预测网络，ema_network 是其指数滑动平均版本。
    推理生成时使用 ema_network，通常比直接使用训练网络更稳定。
    """

    def __init__(self, network, ema_network, timesteps, gdf_util, ema=0.999, p_uncond=0.0, first_channels=8):
        super().__init__()
        self.network = network
        self.ema_network = ema_network
        self.timesteps = timesteps
        self.gdf_util = gdf_util
        self.ema = ema
        self.p_uncond = p_uncond
        self.first_channels = first_channels

    @tf.function    
    def bernoulli(self, shape):
        # classifier-free guidance 的训练技巧：
        # 按概率 p_uncond 把条件 mask 置 0，让网络也学会“无条件”预测。
        c = tf.random.uniform(shape, minval=0, maxval=1, dtype=tf.float32)
        c = tf.where(c < self.p_uncond, 0.0, 1.0)
        return c

    @tf.function
    def train_step(self, images):
        images, _, protocol = images
        # images: 真实 SOH 曲线，protocol: 条件容量/工况矩阵。
        # 1. 获取 batch size。
        batch_size = tf.shape(images)[0]

        # 2. 为 batch 中每条曲线随机采样一个扩散时间步 t。
        t = tf.random.uniform(
            minval=0, maxval=self.timesteps, shape=(batch_size,), dtype=tf.int64
        )

        # 生成条件 mask，并扩展到和条件 embedding 相同的通道数。
        c_mask = self.bernoulli(shape=(batch_size,))
        c_mask = tf.tile(c_mask[...,None], [1, self.first_channels*4])
        
        with tf.GradientTape() as tape:
            # 3. 采样真实噪声 epsilon。
            noise = tf.random.normal(shape=tf.shape(images), dtype=images.dtype)

            # 4. 用 q_sample 得到加噪后的 x_t。
            images_t = self.gdf_util.q_sample(images, t, noise)

            # 5. 网络根据 x_t、t 和条件矩阵预测噪声。
            pred_noise = self.network([images_t, t, protocol, c_mask], training=True)

            # 6. DDPM 常用目标：预测噪声和真实噪声之间的 MSE。
            loss = self.loss(noise, pred_noise)

        # 7. 计算梯度。
        gradients = tape.gradient(loss, self.network.trainable_weights)

        # 8. 更新当前训练网络。
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        # 9. 更新 EMA 网络权重：ema_weight <- ema * ema_weight + (1 - ema) * weight。
        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)

        # 10. 返回 Keras 训练日志需要的 loss。
        return {"loss": loss}
    
    @tf.function
    def test_step(self, images):
        images, _, protocol = images
        # 验证阶段不随机丢条件，直接使用全 1 mask。
        # 1. 获取 batch size。
        batch_size = tf.shape(images)[0]

        # 2. 随机采样验证用扩散时间步。
        t = tf.random.uniform(
            minval=0, maxval=self.timesteps, shape=(batch_size,), dtype=tf.int64
        )

        c_mask = tf.ones(shape=(batch_size, self.first_channels*4))

        # 3. 加噪并预测噪声，计算验证 MSE。
        noise = tf.random.normal(shape=tf.shape(images), dtype=images.dtype)

        images_t = self.gdf_util.q_sample(images, t, noise)

        pred_noise = self.network([images_t, t, protocol, c_mask], training=False)

        loss = self.loss(noise, pred_noise)

        return {"loss": loss}
    
    @tf.function
    def generate(self, samples, tt, capacity_matrices, guide_w):
        # 同时做条件预测和无条件预测，然后按 guidance 权重组合：
        # eps = (1 + w) * eps_cond - w * eps_uncond
        ones = tf.ones((len(samples), self.first_channels*4))
        zeros = tf.zeros((len(samples), self.first_channels*4))

        pred_noise1 = self.ema_network([samples, tt, capacity_matrices, ones], training=False)
        pred_noise2 = self.ema_network([samples, tt, capacity_matrices, zeros], training=False)
        pred_noise = (1+guide_w)*pred_noise1 - guide_w*pred_noise2
        samples = self.gdf_util.p_sample(
            pred_noise, samples, tt, clip_denoised=False
        )
        return samples
    
    def generate_samples(self, capacity_matrices, guide_w = 0.0, record_samples=False):
        # 1. 从标准高斯噪声开始，作为反向扩散链的起点 x_T。
        num_images = len(capacity_matrices)
        samples = tf.random.normal(
            shape=(num_images, 256, 1), dtype=tf.float32
        )
        capacity_matrices = tf.cast(capacity_matrices, dtype=tf.float32)

        record = []
        record.append(samples)
        # 2. 从 T-1 到 0 逐步去噪，最终得到生成的 SOH 曲线。
        for t in reversed(range(0, self.timesteps)):
            tt = tf.cast(tf.fill(num_images, t), dtype=tf.int64)
            samples = self.generate(samples, tt, capacity_matrices, guide_w)
            if record_samples:
                record.append(samples)
        # 3. 可选返回每一步采样轨迹，便于可视化扩散过程。
        if record_samples:
            return samples, record
        else:
            return samples
        
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'network': self.network,
            'ema_network': self.ema_network,
            'timesteps': self.timesteps,
            'gdf_util': self.gdf_util,
            'ema': self.ema,
            'p_uncond': self.p_uncond,
            'first_channels': self.first_channels,
        })
        return config
