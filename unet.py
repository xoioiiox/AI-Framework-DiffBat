import math
import tensorflow as tf
from tensorflow import keras

# Keras 中使用的方差缩放初始化器。MindSpore 复现时可用类似 He/Xavier 初始化替代。
def kernel_init(scale):
    scale = max(scale, 1e-10)
    return keras.initializers.VarianceScaling(
        scale, mode="fan_avg", distribution="uniform"
    )

class AttentionBlock(keras.layers.Layer):
    """Applies self-attention.

    这里的注意力作用在 1D 序列长度维度上：
    输入形状为 [batch, length, channels]，用于建模不同循环位置之间的关系。

    Args:
        units: Number of units in the dense layers
        groups: Number of groups to be used for GroupNormalization layer
    """

    def __init__(self, units, groups=8, **kwargs):
        self.units = units
        self.groups = groups
        super().__init__(**kwargs)

        self.norm = keras.layers.GroupNormalization(groups=groups)
        self.query = keras.layers.Dense(units, kernel_initializer=kernel_init(1.0))
        self.key = keras.layers.Dense(units, kernel_initializer=kernel_init(1.0))
        self.value = keras.layers.Dense(units, kernel_initializer=kernel_init(1.0))
        self.proj = keras.layers.Dense(units, kernel_initializer=kernel_init(0.0))

    def call(self, inputs):
        scale = tf.cast(self.units, tf.float32) ** (-0.5)

        # 先做 GroupNorm，再生成 Q/K/V。
        inputs = self.norm(inputs)
        q = self.query(inputs)
        k = self.key(inputs)
        v = self.value(inputs)

        # 注意力矩阵形状为 [batch, length, length]。
        attn_score = tf.einsum("blc, bLc->blL", q, k) * scale
        attn_score = tf.nn.softmax(attn_score, -1)

        # 将注意力权重作用到 V 上，并用残差连接保留原输入信息。
        proj = tf.einsum("blL,bLc->blc", attn_score, v)
        proj = self.proj(proj)
        return inputs + proj


class TimeEmbedding(keras.layers.Layer):
    """把离散扩散步 t 编码成正弦/余弦位置向量。"""

    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.half_dim = dim // 2
        self.emb = math.log(10000) / (self.half_dim - 1)
        self.emb = tf.exp(tf.range(self.half_dim, dtype=tf.float32) * -self.emb)

    def call(self, inputs):
        inputs = tf.cast(inputs, dtype=tf.float32)
        # 与 Transformer 位置编码类似，用不同频率的 sin/cos 表示时间步。
        emb = inputs[:, None] * self.emb[None, :]
        emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=-1)
        return emb

class PositionalEncoding1D(keras.layers.Layer):
    """给 1D SOH 序列添加循环位置编码。"""

    def __init__(self, channels: int, dtype=tf.float32):
        super(PositionalEncoding1D, self).__init__()
        
        self.channels = int(2 * tf.math.ceil(channels / 2))
        self.inv_freq = 1 / tf.math.pow(10000, tf.range(0, self.channels, 2, dtype=dtype) / self.channels)
        self.cached_penc = None

    @tf.function
    def get_emb(self, sin_inp):
        """
        Gets a base embedding for one dimension with sin and cos intertwined
        """
        emb = tf.stack((tf.sin(sin_inp), tf.cos(sin_inp)), -1)
        emb = tf.reshape(emb, (*emb.shape[:-2], -1))
        return emb

    @tf.function
    def call(self, inputs):
        """
        :param tensor: A 3d tensor of size (batch_size, x, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, ch)
        """
        if len(inputs.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")

        if self.cached_penc is not None and self.cached_penc.shape == inputs.shape:
            return self.cached_penc

        self.cached_penc = None
        _, x, org_channels = inputs.shape

        dtype = self.inv_freq.dtype
        pos_x = tf.range(x, dtype=dtype)
        sin_inp_x = tf.einsum("i,j->ij", pos_x, self.inv_freq)
        emb = self.get_emb(sin_inp_x)

        cached_penc = tf.repeat(
            emb[None, :, :org_channels], tf.shape(inputs)[0], axis=0
        )
        return cached_penc

def ResidualBlock_Down(width, groups=8, activation_fn=keras.activations.swish):
    """下采样路径使用的残差块，只用时间步 embedding 条件化。"""

    def apply(inputs):
        x, t = inputs
        input_width = x.shape[-1]

        # 如果通道数不同，用 1x1 Conv 对 residual 分支对齐通道。
        if input_width == width:
            residual = x
        else:
            residual = keras.layers.Conv1D(
                width, kernel_size=1, kernel_initializer=kernel_init(1.0)
            )(x)

        # 将时间 embedding 投影到当前通道数，并广播到序列长度维度。
        temb = activation_fn(t)
        temb = keras.layers.Dense(width, kernel_initializer=kernel_init(1.0))(temb)[:, None, :]

        # 主分支：Norm -> Activation -> Conv1D。
        x = keras.layers.GroupNormalization(groups=groups)(x)
        x = activation_fn(x)
        x = keras.layers.Conv1D(
            width, kernel_size=3, padding="same", kernel_initializer=kernel_init(1.0)
        )(x)

        # 在中间特征上加入时间条件。
        x = keras.layers.Add()([x, temb])
        x = keras.layers.GroupNormalization(groups=groups)(x)
        x = activation_fn(x)

        x = keras.layers.Conv1D(
            width, kernel_size=3, padding="same", kernel_initializer=kernel_init(0.0)
        )(x)
        x = keras.layers.Add()([x, residual])
        return x

    return apply

def ResidualBlock_Up(width, groups=8, activation_fn=keras.activations.swish):
    """上采样路径使用的残差块，同时使用时间 embedding 和条件矩阵 embedding。"""

    def apply(inputs):
        x, t, c = inputs
        input_width = x.shape[-1]

        # residual 分支保持和主分支输出通道一致。
        if input_width == width:
            residual = x
        else:
            residual = keras.layers.Conv1D(
                width, kernel_size=1, kernel_initializer=kernel_init(1.0)
            )(x)

        # 时间条件：决定当前反向扩散步的信息。
        temb = activation_fn(t)
        temb = keras.layers.Dense(width, kernel_initializer=kernel_init(1.0))(temb)[:, None, :]
            
        # 条件矩阵 embedding：来自 protocol/capacity matrix，用于指导生成曲线。
        cemb = activation_fn(c)
        cemb = keras.layers.Dense(width, kernel_initializer=kernel_init(1.0))(cemb)[:, None, :]

        x = keras.layers.GroupNormalization(groups=groups)(x)
        x = activation_fn(x)
        x = keras.layers.Conv1D(
            width, kernel_size=3, padding="same", kernel_initializer=kernel_init(1.0)
        )(x)

        # 条件化方式：cemb 作为乘性调制，temb 作为加性偏置。
        x = keras.layers.Add()([cemb * x, temb])
        x = keras.layers.GroupNormalization(groups=groups)(x)
        x = activation_fn(x)

        x = keras.layers.Conv1D(
            width, kernel_size=3, padding="same", kernel_initializer=kernel_init(0.0)
        )(x)
        x = keras.layers.Add()([x, residual])
        return x

    return apply


def DownSample(width):
    """用 stride=2 的 Conv1D 将序列长度减半。"""

    def apply(x):
        x = keras.layers.Conv1D(
            width,
            kernel_size=3,
            strides=2,
            padding="same",
            kernel_initializer=kernel_init(1.0),
        )(x)
        return x

    return apply


def UpSample(width, interpolation="nearest"):
    """先最近邻上采样，再用 Conv1D 融合特征。"""

    def apply(x):
        x = keras.layers.UpSampling1D(size=2)(x)
        x = keras.layers.Conv1D(
            width, kernel_size=3, padding="same", kernel_initializer=kernel_init(1.0)
        )(x)
        return x

    return apply


def TimeMLP(units, activation_fn=keras.activations.swish):
    """进一步变换时间 embedding，输出维度与条件 embedding 对齐。"""

    def apply(inputs):
        temb = keras.layers.Dense(
            units, activation=activation_fn, kernel_initializer=kernel_init(1.0)
        )(inputs)
        temb = keras.layers.Dense(units, kernel_initializer=kernel_init(1.0))(temb)
        return temb

    return apply
    
def TransformerEncoder(units, activation_fn=keras.activations.swish):
    """把 100x100 的条件矩阵编码为一个全局条件向量。"""

    def apply(inputs):
        # 将 [100, 100, 1] reshape 为长度为 100、特征维为 100 的序列。
        pemb = keras.layers.Reshape((-1, 100))(inputs)
        pemb = keras.layers.Dense(units, kernel_initializer=kernel_init(1.0))(pemb)

        # 用自注意力提取条件矩阵内部关系，再全局平均池化成一个向量。
        pemb = AttentionBlock(units, groups=8)(pemb)
        pemb = keras.layers.GlobalAveragePooling1D()(pemb)
        pemb = keras.layers.Dense(units, kernel_initializer=kernel_init(1.0))(pemb)
        return pemb
    
    return apply

def build_model(
    sequence_length,
    widths,
    has_attention,
    capacity_matrix_size=(100,100,1),
    first_conv_channels=8,
    num_res_blocks=2,
    norm_groups=8,
    interpolation="nearest",
    activation_fn=keras.activations.gelu,
):
    
    # 主输入：加噪后的 SOH 曲线，形状 [batch, 256, 1]。
    image_input = keras.layers.Input(shape=(sequence_length, 1), name="input")
    # 扩散时间步 t，形状 [batch]。
    time_input = keras.Input(shape=(), dtype=tf.int64, name="time_input")
    # 条件输入：电池容量/工况矩阵，默认形状 [batch, 100, 100, 1]。
    capacity_matrix_input = keras.Input(shape=capacity_matrix_size, name="capacity_matrix_input")
    # classifier-free guidance 的条件 mask，全 0 表示无条件分支，全 1 表示有条件分支。
    condition_mask = keras.Input(shape=(first_conv_channels * 4,), name="mask_input")

    # 第一层卷积把单通道曲线映射到 first_conv_channels 个特征通道。
    x = keras.layers.Conv1D(
        first_conv_channels,
        kernel_size=7,
        padding="same",
        kernel_initializer=kernel_init(1.0),
    )(image_input)

    # 准备三类 embedding：序列位置、扩散时间步、条件矩阵。
    posemb = PositionalEncoding1D(first_conv_channels)(x)
    temb = TimeEmbedding(dim=first_conv_channels * 4)(time_input)
    temb = TimeMLP(units=first_conv_channels * 4, activation_fn=activation_fn)(temb)
    cemb = TransformerEncoder(units=first_conv_channels * 4, activation_fn=activation_fn)(capacity_matrix_input)
    
    # 把原始卷积特征和位置编码拼接，作为 U-Net 起点。
    x = keras.layers.Concatenate()([x, posemb])
    skips = [x]

    # DownBlock：逐级提取更抽象的局部/全局特征，并保存 skip connection。
    for i in range(len(widths)):
        for _ in range(num_res_blocks):
            x = ResidualBlock_Down(
                widths[i], groups=norm_groups, activation_fn=activation_fn
            )([x, temb])
            if has_attention[i]:
                x = AttentionBlock(widths[i], groups=norm_groups)(x)
            skips.append(x)

        if widths[i] != widths[-1]:
            # 不是最后一级时，下采样进入更低分辨率。
            x = DownSample(widths[i])(x)
            skips.append(x)

    # MiddleBlock：U-Net bottleneck，分辨率最低、通道数最高。
    x = ResidualBlock_Down(widths[-1], groups=norm_groups, activation_fn=activation_fn)(
        [x, temb]
    )
    x = AttentionBlock(widths[-1], groups=norm_groups)(x)
    x = ResidualBlock_Down(widths[-1], groups=norm_groups, activation_fn=activation_fn)(
        [x, temb]
    )

    # UpBlock：逐级恢复序列长度，并融合对应的 skip connection。
    for i in reversed(range(len(widths))):
        for _ in range(num_res_blocks + 1):
            x = keras.layers.Concatenate(axis=-1)([x, skips.pop()])
            x = ResidualBlock_Up(
                widths[i], groups=norm_groups, activation_fn=activation_fn
            )([x, temb, condition_mask * cemb])
            if has_attention[i]:
                x = AttentionBlock(widths[i], groups=norm_groups)(x)

        if i != 0:
            # 除最后一级外，每一级结束后都上采样恢复长度。
            x = UpSample(widths[i], interpolation=interpolation)(x)

    # End block：把特征映射回单通道噪声预测，形状与输入 SOH 曲线一致。
    x = keras.layers.GroupNormalization(groups=norm_groups)(x)
    x = activation_fn(x)
    x = keras.layers.Conv1D(1, kernel_size=7, padding="same", kernel_initializer=kernel_init(0.0))(x)
    return keras.Model([image_input, time_input, capacity_matrix_input, condition_mask], x, name="unet")
