import math
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor


def _make_group_count(channels, requested_groups):
    groups = min(requested_groups, channels)
    while channels % groups != 0:
        groups -= 1
    return max(groups, 1)


class Conv1dLast(nn.Cell):
    """Conv1d wrapper for tensors in [batch, length, channels] layout."""

    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, weight_init=None
    ):
        super().__init__()
        if weight_init is None:
            weight_init = "normal"
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            pad_mode="same",
            weight_init=weight_init,
        )

    def construct(self, x):
        x = ops.transpose(x, (0, 2, 1))
        x = self.conv(x)
        return ops.transpose(x, (0, 2, 1))


class GroupNormLast(nn.Cell):
    """GroupNorm wrapper for tensors in [batch, length, channels] layout."""

    def __init__(self, channels, groups=8):
        super().__init__()
        self.norm = nn.GroupNorm(_make_group_count(channels, groups), channels)

    def construct(self, x):
        x = ops.transpose(x, (0, 2, 1))
        x = self.norm(x)
        return ops.transpose(x, (0, 2, 1))


class AttentionBlock(nn.Cell):
    def __init__(self, units, groups=8):
        super().__init__()
        self.units = units
        self.norm = GroupNormLast(units, groups)
        self.query = nn.Dense(units, units)
        self.key = nn.Dense(units, units)
        self.value = nn.Dense(units, units)
        self.proj = nn.Dense(units, units, weight_init="zeros")
        self.softmax = nn.Softmax(axis=-1)

    def construct(self, x):
        x = self.norm(x)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        scale = Tensor(self.units ** -0.5, ms.float32)
        attn = ops.matmul(q, ops.transpose(k, (0, 2, 1))) * scale
        attn = self.softmax(attn)
        proj = ops.matmul(attn, v)
        proj = self.proj(proj)
        return x + proj


class TimeEmbedding(nn.Cell):
    def __init__(self, dim):
        super().__init__()
        half_dim = dim // 2
        emb = math.log(10000.0) / (half_dim - 1)
        emb = np.exp(np.arange(half_dim, dtype=np.float32) * -emb)
        self.emb = Tensor(emb, ms.float32)

    def construct(self, timesteps):
        timesteps = ops.cast(timesteps, ms.float32)
        emb = timesteps[:, None] * self.emb[None, :]
        return ops.concat((ops.sin(emb), ops.cos(emb)), axis=-1)


class PositionalEncoding1D(nn.Cell):
    def __init__(self, sequence_length, channels):
        super().__init__()
        channels_even = int(2 * math.ceil(channels / 2))
        inv_freq = 1.0 / np.power(
            10000.0, np.arange(0, channels_even, 2, dtype=np.float32) / channels_even
        )
        pos = np.arange(sequence_length, dtype=np.float32)
        sin_inp = np.einsum("i,j->ij", pos, inv_freq)
        emb = np.stack((np.sin(sin_inp), np.cos(sin_inp)), axis=-1)
        emb = emb.reshape(sequence_length, -1)[:, :channels]
        self.encoding = Tensor(emb[None, :, :], ms.float32)

    def construct(self, x):
        return ops.tile(self.encoding, (x.shape[0], 1, 1))


class TimeMLP(nn.Cell):
    def __init__(self, units, activation):
        super().__init__()
        self.net = nn.SequentialCell(
            nn.Dense(units, units),
            activation,
            nn.Dense(units, units),
        )

    def construct(self, x):
        return self.net(x)


class TransformerEncoder(nn.Cell):
    def __init__(self, units, capacity_matrix_size=(100, 100, 1), groups=8):
        super().__init__()
        self.feature_dim = capacity_matrix_size[1] * capacity_matrix_size[2]
        self.proj_in = nn.Dense(self.feature_dim, units)
        self.attention = AttentionBlock(units, groups=groups)
        self.proj_out = nn.Dense(units, units)

    def construct(self, x):
        x = ops.reshape(x, (x.shape[0], -1, self.feature_dim))
        x = self.proj_in(x)
        x = self.attention(x)
        x = ops.mean(x, axis=1)
        return self.proj_out(x)


class ResidualBlockDown(nn.Cell):
    def __init__(
        self, in_channels, out_channels, cond_channels, groups=8, activation=None
    ):
        super().__init__()
        self.activation = activation if activation is not None else nn.GELU()
        self.shortcut = (
            Conv1dLast(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )
        self.temb_proj = nn.Dense(cond_channels, out_channels)
        self.norm1 = GroupNormLast(in_channels, groups)
        self.conv1 = Conv1dLast(in_channels, out_channels, kernel_size=3)
        self.norm2 = GroupNormLast(out_channels, groups)
        self.conv2 = Conv1dLast(
            out_channels, out_channels, kernel_size=3, weight_init="zeros"
        )

    def construct(self, x, temb):
        residual = x if self.shortcut is None else self.shortcut(x)
        cond = self.temb_proj(self.activation(temb))[:, None, :]
        x = self.conv1(self.activation(self.norm1(x)))
        x = x + cond
        x = self.conv2(self.activation(self.norm2(x)))
        return x + residual


class ResidualBlockUp(nn.Cell):
    def __init__(self, in_channels, out_channels, cond_channels, groups=8, activation=None):
        super().__init__()
        self.activation = activation if activation is not None else nn.GELU()
        self.shortcut = (
            Conv1dLast(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )
        self.temb_proj = nn.Dense(cond_channels, out_channels)
        self.cemb_proj = nn.Dense(cond_channels, out_channels)
        self.norm1 = GroupNormLast(in_channels, groups)
        self.conv1 = Conv1dLast(in_channels, out_channels, kernel_size=3)
        self.norm2 = GroupNormLast(out_channels, groups)
        self.conv2 = Conv1dLast(
            out_channels, out_channels, kernel_size=3, weight_init="zeros"
        )

    def construct(self, x, temb, cemb):
        residual = x if self.shortcut is None else self.shortcut(x)
        temb = self.temb_proj(self.activation(temb))[:, None, :]
        cemb = self.cemb_proj(self.activation(cemb))[:, None, :]
        x = self.conv1(self.activation(self.norm1(x)))
        x = cemb * x + temb
        x = self.conv2(self.activation(self.norm2(x)))
        return x + residual


class DownSample(nn.Cell):
    def __init__(self, channels):
        super().__init__()
        self.conv = Conv1dLast(channels, channels, kernel_size=3, stride=2)

    def construct(self, x):
        return self.conv(x)


class UpSample(nn.Cell):
    def __init__(self, channels):
        super().__init__()
        self.conv = Conv1dLast(channels, channels, kernel_size=3)

    def construct(self, x):
        x = ops.repeat_interleave(x, repeats=2, axis=1)
        return self.conv(x)


class DiffBattUNet(nn.Cell):
    def __init__(
        self,
        sequence_length,
        widths,
        has_attention,
        capacity_matrix_size=(100, 100, 1),
        first_conv_channels=8,
        num_res_blocks=2,
        norm_groups=8,
    ):
        super().__init__()
        self.sequence_length = sequence_length
        self.widths = list(widths)
        self.has_attention = list(has_attention)
        self.first_conv_channels = first_conv_channels
        self.num_res_blocks = num_res_blocks
        self.cond_channels = first_conv_channels * 4
        activation = nn.GELU()

        self.input_conv = Conv1dLast(1, first_conv_channels, kernel_size=7)
        self.pos_encoding = PositionalEncoding1D(sequence_length, first_conv_channels)
        self.time_embedding = TimeEmbedding(self.cond_channels)
        self.time_mlp = TimeMLP(self.cond_channels, activation)
        self.condition_encoder = TransformerEncoder(
            self.cond_channels, capacity_matrix_size=capacity_matrix_size
        )

        down_blocks = []
        down_attn = []
        down_samples = []
        skip_channels = [first_conv_channels * 2]
        in_channels = first_conv_channels * 2
        for i, width in enumerate(self.widths):
            level_blocks = []
            level_attn = []
            for _ in range(num_res_blocks):
                level_blocks.append(
                    ResidualBlockDown(
                        in_channels,
                        width,
                        self.cond_channels,
                        norm_groups,
                        activation,
                    )
                )
                level_attn.append(
                    AttentionBlock(width, norm_groups) if has_attention[i] else nn.Identity()
                )
                in_channels = width
                skip_channels.append(in_channels)
            down_blocks.append(nn.CellList(level_blocks))
            down_attn.append(nn.CellList(level_attn))
            if width != self.widths[-1]:
                down_samples.append(DownSample(width))
                skip_channels.append(width)
            else:
                down_samples.append(nn.Identity())

        self.down_blocks = nn.CellList(down_blocks)
        self.down_attn = nn.CellList(down_attn)
        self.down_samples = nn.CellList(down_samples)

        self.mid_block1 = ResidualBlockDown(
            in_channels,
            self.widths[-1],
            self.cond_channels,
            norm_groups,
            activation,
        )
        self.mid_attn = AttentionBlock(self.widths[-1], norm_groups)
        self.mid_block2 = ResidualBlockDown(
            self.widths[-1],
            self.widths[-1],
            self.cond_channels,
            norm_groups,
            activation,
        )
        in_channels = self.widths[-1]

        up_blocks = []
        up_attn = []
        up_samples = []
        for i in reversed(range(len(self.widths))):
            level_blocks = []
            level_attn = []
            for _ in range(num_res_blocks + 1):
                in_with_skip = in_channels + skip_channels.pop()
                level_blocks.append(
                    ResidualBlockUp(
                        in_with_skip,
                        self.widths[i],
                        self.cond_channels,
                        norm_groups,
                        activation,
                    )
                )
                level_attn.append(
                    AttentionBlock(self.widths[i], norm_groups)
                    if has_attention[i]
                    else nn.Identity()
                )
                in_channels = self.widths[i]
            up_blocks.append(nn.CellList(level_blocks))
            up_attn.append(nn.CellList(level_attn))
            up_samples.append(UpSample(in_channels) if i != 0 else nn.Identity())

        self.up_blocks = nn.CellList(up_blocks)
        self.up_attn = nn.CellList(up_attn)
        self.up_samples = nn.CellList(up_samples)
        self.out_norm = GroupNormLast(in_channels, norm_groups)
        self.out_conv = Conv1dLast(in_channels, 1, kernel_size=7, weight_init="zeros")
        self.out_activation = activation

    def construct(self, x, timesteps, capacity_matrix, condition_mask):
        x = self.input_conv(x)
        x = ops.concat((x, self.pos_encoding(x)), axis=-1)
        skips = [x]

        temb = self.time_mlp(self.time_embedding(timesteps))
        cemb = self.condition_encoder(capacity_matrix)
        cemb = condition_mask * cemb

        for i in range(len(self.widths)):
            for j in range(self.num_res_blocks):
                x = self.down_blocks[i][j](x, temb)
                x = self.down_attn[i][j](x)
                skips.append(x)
            if self.widths[i] != self.widths[-1]:
                x = self.down_samples[i](x)
                skips.append(x)

        x = self.mid_block1(x, temb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, temb)

        for i in range(len(self.up_blocks)):
            for j in range(self.num_res_blocks + 1):
                x = ops.concat((x, skips.pop()), axis=-1)
                x = self.up_blocks[i][j](x, temb, cemb)
                x = self.up_attn[i][j](x)
            x = self.up_samples[i](x)

        x = self.out_activation(self.out_norm(x))
        return self.out_conv(x)


def build_model(
    sequence_length=256,
    widths=None,
    has_attention=None,
    capacity_matrix_size=(100, 100, 1),
    first_conv_channels=8,
    num_res_blocks=2,
    norm_groups=8,
):
    if widths is None:
        widths = [first_conv_channels * mult for mult in [1, 2, 4, 8]]
    if has_attention is None:
        has_attention = [False, False, True, True]
    return DiffBattUNet(
        sequence_length=sequence_length,
        widths=widths,
        has_attention=has_attention,
        capacity_matrix_size=capacity_matrix_size,
        first_conv_channels=first_conv_channels,
        num_res_blocks=num_res_blocks,
        norm_groups=norm_groups,
    )
